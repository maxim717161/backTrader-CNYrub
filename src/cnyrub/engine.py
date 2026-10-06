"""Минутный шаг позиции. Общий для прогона на истории и для живого счёта.

За минуту позиция меняется не больше чем на FILL_PER_MINUTE контрактов,
каждый кусок по цене закрытия этой минуты. Решение на закрытии исполняется
со следующей минуты. Стоп и последний бар ряда могут поставить цель до куска
этой же минуты.
"""

from __future__ import annotations

from datetime import date

MULTIPLIER = 1000.0
COMMISSION = 1.0
# За минуту позиция меняется не больше чем на столько контрактов.
FILL_PER_MINUTE = 10
CLOCK_DAYS = 5
# Медиана объёма предыдущих 300 минут. Меньше 60 минуток — медианы ещё нет.
SURGE_BARS = 300
SURGE_MIN_PERIODS = 60
# От одной до двух медиан пробоя доля размера держится на 50%.
LONG_BREAKOUT_LOW = 1.0
LONG_BREAKOUT_HIGH = 2.0
LONG_SCALE_FLOOR = 0.5


class _FillBook:
    """Текущая позиция и цель. К цели идём кусками по закрытию минуты."""

    def __init__(self, secid: str, equity: float | None) -> None:
        self.secid = secid
        self.equity = equity
        self.held = 0
        self.target = 0
        self.avg = 0.0
        self.stop_dist: float | None = None
        self.stop_px: float | None = None
        self.trail = False
        self.entry_i: int | None = None
        self.reason = ""
        # Пробой, из-за которого открыта цель. Живёт, пока позиция не закрыта.
        self.entry = ""
        self.opened = 0
        self.gross = 0.0
        self.commission = 0.0
        self.cooldown_until = 0
        self.last_signed = 0
        self.trades: list[dict[str, object]] = []
        self.base = 0
        self.best: float | None = None
        self.unit = 0.0
        self.scale_level = 0
        self.scaled = False
        self.peak = 0

    def stop_hit(self, opened: float, high: float, low: float) -> bool:
        if self.held == 0 or self.stop_px is None:
            return False
        if self.held > 0:
            return opened <= self.stop_px or low <= self.stop_px
        return opened >= self.stop_px or high >= self.stop_px

    def move(self, price: float, index: int, limit: int) -> int:
        """Сдвинуть позицию к цели не больше чем на limit контрактов."""
        self.last_signed = 0
        delta = self.target - self.held
        if delta == 0 or limit <= 0:
            return 0
        step = max(-limit, min(limit, delta))
        if self.held > 0:
            step = max(step, -self.held)
        elif self.held < 0:
            step = min(step, -self.held)
        if step == 0:
            return 0
        self._apply(step, price, index)
        self.last_signed = step
        return step

    def _sync_stop(self) -> None:
        if self.held == 0 or self.stop_dist is None:
            self.stop_px = None
            return
        base = self.avg - self.stop_dist if self.held > 0 else self.avg + self.stop_dist
        if self.stop_px is None or not self.trail:
            self.stop_px = base
        elif self.held > 0:
            self.stop_px = max(self.stop_px, base)
        else:
            self.stop_px = min(self.stop_px, base)

    def _apply(self, step: int, price: float, index: int) -> None:
        if self.held == 0:
            self.avg = price
            self.held = step
            self.entry_i = index
            self.opened = abs(step)
            self.peak = abs(step)
            self.gross = 0.0
            self.commission = COMMISSION * abs(step)
            self.reason = ""
            self.best = price
            self._sync_stop()
            return
        if step * self.held > 0:
            total = abs(self.held) + abs(step)
            self.avg = (self.avg * abs(self.held) + price * abs(step)) / total
            self.held += step
            self.opened += abs(step)
            self.peak = max(self.peak, abs(self.held))
            self.commission += COMMISSION * abs(step)
            self._sync_stop()
            return
        sign = 1 if self.held > 0 else -1
        lots = abs(step)
        self.gross += (price - self.avg) * sign * lots * MULTIPLIER
        self.commission += COMMISSION * lots
        self.held += step
        if self.held != 0:
            return
        pnlcomm = self.gross - self.commission
        self.trades.append(
            {
                "secid": self.secid,
                "direction": "long" if sign > 0 else "short",
                "lots": self.peak or self.opened,
                "pnl": self.gross,
                "pnlcomm": pnlcomm,
                "bars": index - int(self.entry_i),
                "reason": self.reason,
                "scaled": self.scaled,
            }
        )
        if self.equity is not None:
            self.equity += pnlcomm
        self.avg = 0.0
        self.stop_dist = None
        self.stop_px = None
        self.entry_i = None
        self.reason = ""
        self.entry = ""
        self.opened = 0
        self.gross = 0.0
        self.commission = 0.0
        self.base = 0
        self.best = None
        self.unit = 0.0
        self.scale_level = 0
        self.scaled = False
        self.peak = 0


def marked_equity(book: _FillBook, close: float) -> float:
    """Счёт с открытой позицией по этой цене: уже закрытый результат плюс текущий."""
    settled = 0.0 if book.equity is None else book.equity
    return settled + book.gross - book.commission + (close - book.avg) * book.held * MULTIPLIER


def _resize_for_pullback(
    book: _FillBook,
    close: float,
    step: float,
    floor: float,
    restore: float,
) -> None:
    """Уменьшить цель, когда цена ушла против лучшей цены сделки, и вернуть, когда она вернулась.

    step и restore — медианы минутного диапазона, зафиксированные на входе.
    На step и дальше цель равна доле floor от исходного размера. На restore
    и ближе к лучшей цене цель снова полная. Между ними цель не меняется,
    чтобы откат туда-сюда не крутил позицию. До нуля цель не падает.
    """
    if book.held == 0 or book.base == 0 or not book.unit > 0 or not step > 0:
        return
    if restore >= step:
        restore = step / 2
    if book.best is None:
        book.best = close
    if book.held > 0:
        book.best = max(book.best, close)
        pullback = book.best - close
    else:
        book.best = min(book.best, close)
        pullback = close - book.best
    ranges = pullback / book.unit if pullback > 0 else 0.0
    if ranges >= step:
        book.scale_level = 1
    elif ranges <= restore:
        book.scale_level = 0
    if book.scale_level:
        book.scaled = True
    fraction = floor if book.scale_level else 1.0
    desired = int(abs(book.base) * fraction)
    if desired < 1:
        desired = 1
    book.target = (1 if book.base > 0 else -1) * desired


def step_minute(
    book: _FillBook,
    index: int,
    *,
    opened: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    day: date,
    next_day: date | None,
    last_day: date,
    prior_high: float,
    prior_low: float,
    prior_vol: float,
    prior_range: float,
    exit_high: float,
    exit_low: float,
    clock_vol: float,
    entry_ready: float,
    clearance: float,
    cooldown: int,
    clock_volume: bool,
    clock_cap: float | None,
    size_mode: str,
    stop_mult: float | None,
    loss_bars: int | None,
    risk_fraction: float | None,
    margin: float,
    stop_rub: float | None,
    breakout_span: float | None,
    trade_from: date | None,
    trail: bool,
    scale_step: float | None = None,
    scale_floor: float = LONG_SCALE_FLOOR,
    scale_back: float | None = None,
    fill_per_minute: int | None = None,
    drift: float = float("nan"),
    path: float = float("nan"),
    surge_vol: float = float("nan"),
    eff_low: float | None = None,
    eff_high: float | None = None,
    surge_cap: float | None = None,
    leverage: float | None = None,
) -> None:
    """Одна минута: сначала кусок по её закрытию, потом решение на следующие."""
    if day == last_day and (book.held != 0 or book.target != 0):
        book.target = 0
        if book.held != 0 and not book.reason:
            book.reason = "expiry"
    elif book.stop_hit(opened, high, low):
        book.target = 0
        book.reason = "stop"
        if cooldown:
            book.cooldown_until = index + cooldown

    pace = FILL_PER_MINUTE if fill_per_minute is None else fill_per_minute
    limit = abs(book.target - book.held) if next_day is None else pace
    book.move(close, index, limit)
    if trail and book.held != 0 and book.stop_dist is not None and book.stop_px is not None:
        if book.held > 0:
            book.stop_px = max(book.stop_px, close - book.stop_dist)
        else:
            book.stop_px = min(book.stop_px, close + book.stop_dist)

    if day == last_day or next_day is None:
        return
    if book.held != 0 and book.target != 0:
        if (
            loss_bars
            and book.entry_i is not None
            and index - book.entry_i >= loss_bars
            and (close - book.avg) * book.held < 0
        ):
            book.target = 0
            book.reason = "time"
            return
        if exit_low == exit_low and book.held > 0 and close < exit_low:
            book.target = 0
            book.reason = "channel"
            return
        if exit_high == exit_high and book.held < 0 and close > exit_high:
            book.target = 0
            book.reason = "channel"
            return
        if scale_step:
            restore = scale_step / 2 if scale_back is None else scale_back
            _resize_for_pullback(book, close, scale_step, scale_floor, restore)
        return
    if book.held != 0 or book.target != 0 or next_day == last_day or index < book.cooldown_until:
        return
    if trade_from is not None and next_day < trade_from:
        return
    if entry_ready < 1 or prior_high != prior_high:
        return
    if clock_volume:
        if clock_vol != clock_vol:
            return
        typical = clock_vol
    else:
        typical = prior_vol
    if typical == typical and typical > 0 and volume < typical:
        return
    if clock_cap is not None and clock_vol == clock_vol and clock_vol > 0 and volume > clock_cap * clock_vol:
        return
    room = clearance * prior_range if prior_range == prior_range else 0.0
    if close > prior_high + room:
        side = 1
    elif prior_low == prior_low and close < prior_low - room:
        side = -1
    else:
        return
    if eff_low is not None and eff_high is not None:
        if not path > 0 or drift != drift:
            return
        efficiency = side * drift / path
        if not eff_low <= efficiency <= eff_high:
            return
    if (
        surge_cap is not None
        and surge_vol == surge_vol
        and surge_vol > 0
        and surge_vol != float("inf")
        and volume > surge_cap * surge_vol
    ):
        return
    if close <= 0:
        return
    if leverage is not None and leverage > 0:
        if book.equity is None:
            raise ValueError("Для плеча нужен текущий счёт")
        lots = int(leverage * book.equity // (close * MULTIPLIER))
        # Плечо считает контракты от цены. Залог не даёт взять больше, чем покрывает счёт.
        if margin > 0:
            lots = min(lots, int(book.equity // margin))
        if lots < 1:
            return
        book.stop_dist = None if stop_mult is None else stop_mult * prior_range
    elif breakout_span:
        if prior_range == prior_range and prior_range > 0:
            beyond = (close - prior_high) / prior_range if side > 0 else (prior_low - close) / prior_range
        else:
            beyond = float("inf")
        if book.equity is None:
            raise ValueError("Для доли пробоя нужен текущий счёт")
        sized = breakout_lots(book.equity, margin, beyond, breakout_span)
        if sized is None:
            return
        lots = sized
        book.stop_dist = None if stop_mult is None else stop_mult * prior_range
    elif risk_fraction:
        if book.equity is None:
            raise ValueError("Для доли риска нужен текущий счёт")
        sized_risk = _risk_size(
            book.equity,
            risk_fraction,
            margin,
            0.0 if stop_rub is None else stop_rub,
        )
        if sized_risk is None:
            return
        lots, book.stop_dist = sized_risk
    else:
        book.stop_dist = None if stop_mult is None else stop_mult * prior_range
        lots = entry_lots(size_mode, side, close, prior_high, prior_low, prior_range)
    book.reason = ""
    book.entry = "up" if side > 0 else "down"
    book.base = side * lots
    book.unit = float(prior_range) if prior_range == prior_range and prior_range > 0 else 0.0
    book.best = None
    book.scale_level = 0
    book.scaled = False
    book.target = side * lots


def _risk_size(
    equity: float,
    risk_fraction: float,
    margin: float,
    stop_rub: float,
) -> tuple[int, float] | None:
    """Контракты так, чтобы стоп stop_rub плюс комиссия забирали не больше доли счёта."""
    if stop_rub <= 0:
        return None
    unit = stop_rub + 2 * COMMISSION
    by_risk = int((risk_fraction * equity) // unit)
    by_margin = int(equity // (margin + COMMISSION))
    contracts = min(by_risk, by_margin)
    if contracts < 1:
        return None
    return contracts, stop_rub / MULTIPLIER


def breakout_fraction(
    beyond: float,
    span: float,
    low: float = LONG_BREAKOUT_LOW,
    high: float = LONG_BREAKOUT_HIGH,
) -> float:
    """0 медиан пробоя — 1, от low до high — 0.5, на span и дальше — 0.

    До одной медианы доля падает со 100% до 50%. Между одной и двумя держится
    на 50%. Дальше линейно сходит к нулю на span.
    """
    if span <= 0 or low <= 0 or high <= low or high >= span or beyond != beyond or beyond == float("inf"):
        return 0.0
    if beyond <= 0:
        return 1.0
    if beyond >= span:
        return 0.0
    if beyond < low:
        return 1.0 - 0.5 * (beyond / low)
    if beyond <= high:
        return 0.5
    return 0.5 * (span - beyond) / (span - high)


def breakout_lots(equity: float, margin: float, beyond: float, span: float) -> int | None:
    """Контракты: доля пробоя от максимума, который пускает залог."""
    fraction = breakout_fraction(beyond, span)
    if fraction <= 0:
        return None
    maximum = int(equity // (margin + COMMISSION))
    contracts = int(maximum * fraction)
    if contracts < 1:
        return None
    return contracts


def entry_lots(
    size_mode: str,
    side: int,
    close: float,
    prior_high: float,
    prior_low: float,
    prior_range: float,
) -> int:
    """1 лот, либо 3/2/1 по тому, насколько закрытие ушло за канал."""
    if size_mode != "inverse":
        return 1
    if not prior_range > 0:
        return 1
    beyond = (close - prior_high) / prior_range if side > 0 else (prior_low - close) / prior_range
    if beyond < 1.0:
        return 3
    if beyond < 2.0:
        return 2
    return 1


_BOOK_FIELDS = (
    "secid",
    "equity",
    "held",
    "target",
    "avg",
    "stop_dist",
    "stop_px",
    "trail",
    "entry_i",
    "reason",
    "entry",
    "opened",
    "gross",
    "commission",
    "cooldown_until",
    "last_signed",
    "base",
    "best",
    "unit",
    "scale_level",
    "scaled",
    "peak",
)


def export_book(book: _FillBook) -> dict[str, object]:
    """Снимок книги для хранения и для отката неудачной заявки."""
    data: dict[str, object] = {name: getattr(book, name) for name in _BOOK_FIELDS}
    data["trades"] = [dict(trade) for trade in book.trades]
    return data


def load_book(data: dict[str, object] | None, secid: str, equity: float | None) -> _FillBook:
    """Собрать книгу из снимка. Пустой снимок — плоская позиция."""
    book = _FillBook(secid, equity)
    if not data:
        return book
    for name in _BOOK_FIELDS:
        if name in data:
            setattr(book, name, data[name])
    book.held = int(book.held)
    book.target = int(book.target)
    book.base = int(book.base)
    book.opened = int(book.opened)
    book.peak = int(book.peak)
    book.cooldown_until = int(book.cooldown_until)
    book.last_signed = int(book.last_signed)
    book.scale_level = int(book.scale_level)
    book.trail = bool(book.trail)
    book.scaled = bool(book.scaled)
    if book.entry_i is not None:
        book.entry_i = int(book.entry_i)
    book.trades = [dict(trade) for trade in data.get("trades") or []]
    return book


def restore_book(book: _FillBook, data: dict[str, object]) -> None:
    """Вернуть книгу к снимку, не меняя объект, на который уже есть ссылки."""
    fresh = load_book(data, book.secid, book.equity)
    book.__dict__.clear()
    book.__dict__.update(fresh.__dict__)


def reprice_fill(book: _FillBook, before: dict[str, object], close: float, executed: float) -> None:
    """Заменить цену последнего куска с закрытия свечи на цену сделки.

    Книга уже сдвинута по close. before — снимок до этого шага.
    """
    step = int(book.last_signed)
    if step == 0 or executed == close:
        return
    before_held = int(before["held"])
    if before_held == 0:
        book.avg = executed
        if book.best == close:
            book.best = executed
        book._sync_stop()
        return
    sign = 1 if before_held > 0 else -1
    lots = abs(step)
    if step * before_held > 0:
        total = abs(before_held) + lots
        book.avg = (float(before["avg"]) * abs(before_held) + executed * lots) / total
        book._sync_stop()
        return
    delta = (executed - close) * sign * lots * MULTIPLIER
    if book.held != 0:
        book.gross += delta
        return
    if book.trades:
        trade = book.trades[-1]
        trade["pnl"] = float(trade["pnl"]) + delta
        trade["pnlcomm"] = float(trade["pnlcomm"]) + delta
    if book.equity is not None:
        book.equity += delta
