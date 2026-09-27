"""Канал по закрытиям минуток каждого квартального CNY/RUB.

Склеенного ряда нет: у каждого контракта свой прогон. Сигнал — закрытие
минуты за пределами предыдущих N максимумов или минимумов. Позиция меняется
не быстрее 10 контрактов в минуту, каждый кусок по закрытию этой минуты.
Первый кусок — на следующей минуте после сигнала, если объём минуты сигнала
не ниже медианы этих N минут. Выход — обратный пробой более короткого окна
или стоп в нескольких медианах минутного диапазона: закрытие тоже идёт
по 10 контрактов в минуту. В последний день новая сделка не открывается,
открытая так же разбирается по минутам.

Сделки идут только в истекающем контракте, со следующего дня после экспирации
предыдущего. Месяц до этого дня уже лежит в файле и прогревает окно, но
позиция в нём не открывается. После экспирации следующий контракт торгуется
сразу: окно к этому дню уже собрано.

В работе два окна, и правила у них разные. Короткое окно 525 минут
закрывается каналом той же длины. Минута сигнала не громче пяти медиан
той же минуты суток за пять дней. Счёт 100 000 руб., залог 1 000 руб.
Стоп — 285 руб. на контракт. Контрактов столько, чтобы этот стоп забирал
не больше 10% счёта, и число растёт вместе со счётом. Если через 1 450 минут
сделка всё ещё в минусе, она закрывается.
Длинное окно 12 420 минут не выходит по каналу и держит стоп в 22 медианы
минутного диапазона. Ноль медиан пробоя — это 100% контрактов, которые
пускает залог, от одной до двух медиан — 50%, двенадцать медиан — 0%. Счёт 100 000 руб.,
залог 1 000 руб. Пока сделка открыта, запоминается лучшая цена закрытия.
Откат на 100 медиан минутного диапазона уменьшает позицию до половины
исходного размера. Когда откат сжимается до 50 медиан, размер возвращается.
Обычный день — это около 50 таких медиан. Куски те же: 10 контрактов в минуту.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import backtrader as bt
import numpy as np
import pandas as pd

COARSE_FROM = 480
COARSE_TO = 14_400
COARSE_STEP = 480
REFINE_STEP = 120
REFINE_RADIUS = 480
# Середина двух соседних лучших окон при уже выбранных выходах.
# Короткое 525 — между 520 и 530: выход той же длины, стопа нет.
# Длинное 12420 — между 12240 и 12600: выход по каналу выключен, стоп 22.
SHORT_WINDOW = 525
LONG_WINDOW = 12_420
LONG_STOP = 22.0
LONG_CASH = 100_000.0
LONG_MARGIN = 1_000.0
# Ноль медиан пробоя — 100% контрактов по залогу, столько медиан — 0%.
# От LOW до HIGH медиан доля держится на 50%, как раньше два контракта.
LONG_BREAKOUT_SPAN = 12.0
LONG_BREAKOUT_LOW = 1.0
LONG_BREAKOUT_HIGH = 2.0
# Откат от лучшей цены сделки, в медианах минутного диапазона на входе.
# 100 медиан — около двух обычных дней: позиция уменьшается до половины.
# 50 медиан — около одного дня: размер возвращается. До нуля не режем,
# чтобы ту же сделку можно было набрать обратно.
LONG_SCALE_STEP = 100.0
LONG_SCALE_BACK = 50.0
LONG_SCALE_FLOOR = 0.5
SHORT_CLOCK_CAP = 5.0
LONG_CANDIDATES = (12_480, 12_960)


# Короткое окно: счёт 100 000, залог 1 000. Стоп 285 руб. на контракт.
# Контрактов столько, чтобы этот стоп забирал не больше 10% счёта.
# Сделка, которая через 1 450 минут всё ещё в минусе, закрывается.
SHORT_CASH = 100_000.0
SHORT_MARGIN = 1_000.0
SHORT_RISK = 0.10
SHORT_STOP_RUB = 285.0
SHORT_LOSS_BARS = 1_450


class Window(NamedTuple):
    """Свои вход, выход, стоп, потолок объёма и размер позиции."""

    channel: int
    exit_channel: int
    stop_mult: float | None
    clock_cap: float | None = None
    size_mode: str = "flat"
    loss_bars: int | None = None
    risk_fraction: float | None = None
    cash: float = 1_000_000.0
    margin: float = 20_000.0
    stop_rub: float | None = None
    breakout_span: float | None = None
    scale_step: float | None = None
    scale_back: float | None = None
    scale_floor: float = LONG_SCALE_FLOOR


WINDOWS = (
    Window(
        SHORT_WINDOW,
        SHORT_WINDOW,
        None,
        SHORT_CLOCK_CAP,
        "flat",
        SHORT_LOSS_BARS,
        SHORT_RISK,
        SHORT_CASH,
        SHORT_MARGIN,
        SHORT_STOP_RUB,
    ),
    Window(
        LONG_WINDOW,
        0,
        LONG_STOP,
        None,
        "span",
        None,
        None,
        LONG_CASH,
        LONG_MARGIN,
        None,
        LONG_BREAKOUT_SPAN,
        LONG_SCALE_STEP,
        LONG_SCALE_BACK,
        LONG_SCALE_FLOOR,
    ),
)
CHANNEL = SHORT_WINDOW
STOP_MULT = 2.5
# Длина выхода = N * scale / 4. Ноль выключает выход по каналу.
EXIT_SCALES = (0, 1, 2, 4, 8)
STOP_GRID = (1.5, 2.0, 2.5, 3.5, 5.0, 8.0)
MIN_TRADES_PER_CONTRACT = 4
CLEARANCE = 0.5
CLOCK_DAYS = 5
PARTS = (
    ("CRM2", "CRU2", "CRZ2", "CRH3"),
    ("CRM3", "CRU3", "CRZ3", "CRH4"),
    ("CRM4", "CRU4", "CRZ4", "CRH5"),
    ("CRM5", "CRU5", "CRZ5", "CRH6"),
    ("CRM6", "CRU6", "CRZ6"),
)
MULTIPLIER = 1000.0
COMMISSION = 1.0
# За минуту позиция меняется не больше чем на столько контрактов.
FILL_PER_MINUTE = 10
MARGIN = 20_000.0
START_CASH = 1_000_000.0
BARS_DIR = Path("data/bars")


class SignalData(bt.feeds.PandasData):
    lines = (
        "prior_high",
        "prior_low",
        "prior_vol",
        "prior_range",
        "exit_high",
        "exit_low",
        "entry_ready",
        "exit_ready",
        "clock_vol",
    )
    params = (
        ("prior_high", -1),
        ("prior_low", -1),
        ("prior_vol", -1),
        ("prior_range", -1),
        ("exit_high", -1),
        ("exit_low", -1),
        ("entry_ready", -1),
        ("exit_ready", -1),
        ("clock_vol", -1),
    )


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

    limit = abs(book.target - book.held) if next_day is None else FILL_PER_MINUTE
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
    if breakout_span:
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
    book.base = side * lots
    book.unit = float(prior_range) if prior_range == prior_range and prior_range > 0 else 0.0
    book.best = None
    book.scale_level = 0
    book.scaled = False
    book.target = side * lots


class MinuteDonchian(bt.Strategy):
    params = dict(
        secid="",
        last_day="",
        stop_mult=STOP_MULT,
        use_stop=True,
        clock_cap=None,
        size_mode="flat",
        trade_from="",
        loss_bars=0,
        risk_fraction=0.0,
        margin=MARGIN,
        stop_rub=0.0,
        breakout_span=0.0,
        scale_step=0.0,
        scale_floor=LONG_SCALE_FLOOR,
        scale_back=0.0,
    )

    def __init__(self) -> None:
        self.book = _FillBook(str(self.p.secid), None)
        self.trades = self.book.trades
        self.values: list[tuple[object, float]] = []

    def start(self) -> None:
        self.book.equity = float(self.broker.getvalue())

    def next(self) -> None:
        book = self.book
        close = float(self.data.close[0])
        step_minute(
            book,
            len(self) - 1,
            opened=float(self.data.open[0]),
            high=float(self.data.high[0]),
            low=float(self.data.low[0]),
            close=close,
            volume=float(self.data.volume[0]),
            day=self.data.datetime.date(0),
            next_day=self._following_day(),
            last_day=self._last_day(),
            prior_high=float(self.data.prior_high[0]),
            prior_low=float(self.data.prior_low[0]),
            prior_vol=float(self.data.prior_vol[0]),
            prior_range=float(self.data.prior_range[0]),
            exit_high=float(self.data.exit_high[0]),
            exit_low=float(self.data.exit_low[0]),
            clock_vol=float(self.data.clock_vol[0]),
            entry_ready=float(self.data.entry_ready[0]),
            clearance=0.0,
            cooldown=0,
            clock_volume=False,
            clock_cap=None if self.p.clock_cap is None else float(self.p.clock_cap),
            size_mode=str(self.p.size_mode),
            stop_mult=None if not self.p.use_stop or self.p.risk_fraction else float(self.p.stop_mult),
            loss_bars=int(self.p.loss_bars) or None,
            risk_fraction=float(self.p.risk_fraction) or None,
            margin=float(self.p.margin),
            stop_rub=float(self.p.stop_rub) or None,
            breakout_span=float(self.p.breakout_span) or None,
            trade_from=self._front_day(),
            trail=False,
            scale_step=float(self.p.scale_step) or None,
            scale_floor=float(self.p.scale_floor),
            scale_back=float(self.p.scale_back) or None,
        )
        if book.last_signed:
            self._fill_broker(book.last_signed, close)
        self.broker._get_value()
        self.values.append((self.data.datetime.datetime(0), float(self.broker.getvalue())))

    def _following_day(self) -> date | None:
        if len(self.data) >= self.data.buflen():
            return None
        return self.data.datetime.date(1)

    def _fill_broker(self, signed: int, price: float) -> None:
        """Исполнить кусок по закрытию этой минуты, без второй сделки на открытии следующей."""
        order = self.buy(size=signed) if signed > 0 else self.sell(size=-signed)
        broker = self.broker
        broker.submitted.remove(order)
        order.accept()
        broker._execute(order, ago=0, price=price)

    def _last_day(self) -> date:
        value = self.p.last_day
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    def _front_day(self) -> date | None:
        return _as_date(self.p.trade_from)


def _as_feed(frame: pd.DataFrame) -> pd.DataFrame:
    feed = frame.copy()
    if not isinstance(feed.index, pd.DatetimeIndex):
        stamp = feed["datetime"] if "datetime" in feed.columns else feed.index
        feed = feed.copy()
        feed.index = pd.to_datetime(stamp)
    feed = feed.sort_index()
    columns = [name for name in ("open", "high", "low", "close", "volume") if name in feed.columns]
    return feed.loc[:, columns]


def _as_date(value: date | datetime | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def load_trade_starts(bars_dir: Path = BARS_DIR) -> dict[str, date]:
    """День, с которого контракт истекающий и в нём можно открывать сделки."""
    path = bars_dir.parent / "contracts.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    starts: dict[str, date] = {}
    for row in document["contracts"]:
        front = _as_date(row.get("window_start"))
        if front is not None:
            starts[str(row["secid"])] = front
    return starts


def load_minutes(bars_dir: Path = BARS_DIR) -> dict[str, pd.DataFrame]:
    """Минутки каждого контракта отдельно, в порядке последнего бара."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(bars_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["datetime", "open", "high", "low", "close", "volume"])
        frames[path.stem] = _as_feed(frame)
    ordered = sorted(frames, key=lambda secid: frames[secid].index[-1])
    return {secid: frames[secid] for secid in ordered}


def channel_view(frame: pd.DataFrame, channel: int, exit_channel: int | None = None) -> pd.DataFrame:
    """Уровни канала на закрытии минуты: окно не включает саму эту минуту."""
    if exit_channel is None:
        exit_channel = max(channel // 2, 1)
    high = frame["high"]
    low = frame["low"]
    view = pd.DataFrame(index=frame.index)
    view["open"] = frame["open"]
    view["high"] = high
    view["low"] = low
    view["close"] = frame["close"]
    view["volume"] = frame["volume"]
    view["prior_high"] = high.rolling(channel, min_periods=channel).max().shift(1)
    view["prior_low"] = low.rolling(channel, min_periods=channel).min().shift(1)
    view["prior_vol"] = frame["volume"].rolling(channel, min_periods=channel).median().shift(1)
    view["prior_range"] = (high - low).rolling(channel, min_periods=channel).median().shift(1)
    if exit_channel > 0:
        view["exit_high"] = high.rolling(exit_channel, min_periods=exit_channel).max().shift(1)
        view["exit_low"] = low.rolling(exit_channel, min_periods=exit_channel).min().shift(1)
        view["exit_ready"] = view["exit_low"].notna().astype(float)
    else:
        view["exit_high"] = np.nan
        view["exit_low"] = np.nan
        view["exit_ready"] = 0.0
    view["entry_ready"] = view["prior_high"].notna().astype(float)
    view["clock_vol"] = _clock_volume(pd.DatetimeIndex(view.index), frame["volume"].to_numpy(dtype=float), CLOCK_DAYS)
    return view


def _clock_volume(index: pd.DatetimeIndex, volume: np.ndarray, lookback: int) -> np.ndarray:
    """Медиана объёма той же минуты суток по предыдущим lookback наблюдениям."""
    minutes = index.hour * 60 + index.minute
    typical = np.full(len(volume), np.nan)
    history: dict[int, list[float]] = {}
    for i, minute in enumerate(minutes):
        seen = history.get(int(minute))
        if seen is not None and len(seen) >= lookback:
            typical[i] = float(np.median(seen[-lookback:]))
        history.setdefault(int(minute), []).append(float(volume[i]))
    return typical


def scaled_exit(channel: int, scale: int) -> int:
    """Длина выхода: scale 2 — это половина окна, scale 0 — выход по каналу выключен."""
    if scale <= 0:
        return 0
    return max(channel * scale // 4, 1)


def simulate(
    secid: str,
    frame: pd.DataFrame,
    channel: int,
    *,
    clearance: float = 0.0,
    cooldown: int = 0,
    clock_volume: bool = False,
    clock_cap: float | None = None,
    size_mode: str = "flat",
    stop_mult: float | None = STOP_MULT,
    exit_channel: int | None = None,
    trail: bool = False,
    trade_from: date | None = None,
    loss_bars: int | None = None,
    risk_fraction: float | None = None,
    margin: float = MARGIN,
    stop_rub: float | None = None,
    cash: float | None = None,
    view: pd.DataFrame | None = None,
    breakout_span: float | None = None,
    scale_step: float | None = None,
    scale_floor: float = LONG_SCALE_FLOOR,
    scale_back: float | None = None,
    marks: list[float] | None = None,
) -> list[dict[str, object]]:
    """Те же правила, что у стратегии в backtrader, без самого движка.

    Позиция меняется не быстрее чем на FILL_PER_MINUTE контрактов за минуту,
    каждый кусок по закрытию этой минуты.

    clearance — на сколько медиан диапазона закрытие должно пробить канал.
    cooldown — сколько минут после стопа нельзя открывать новую сделку.
    clock_volume заменяет сравнение с медианой последних N минут на медиану
    той же минуты суток за предыдущие дни.
    clock_cap — потолок: минута сигнала не громче, чем clock_cap медиан
    той же минуты суток за предыдущие дни. Пол по медиане окна остаётся.
    size_mode inverse ставит 3, 2 или 1 лот, если пробой короче одной,
    двух или больше медиан минутного диапазона.
    breakout_span — ноль медиан пробоя берёт все контракты по залогу,
    столько медиан берёт ноль.
    stop_mult None выключает стоп. trail подтягивает стоп за закрытием
    на исходную дистанцию, уже после проверки стопа на этой минуте.
    trade_from — первый день истекающего контракта. Более ранние минуты
    только собирают окно: сделка открывается не раньше открытия этого дня.
    loss_bars закрывает сделку, которая столько минут всё ещё в минусе.
    Исполнение — не больше FILL_PER_MINUTE контрактов за минуту по её закрытию.
    scale_step уменьшает позицию до доли scale_floor, когда закрытие отходит
    от лучшей цены сделки на столько медиан минутного диапазона. scale_back
    возвращает полный размер, когда откат сжимается до стольких медиан.
    marks, если передан, получает счёт после каждой минуты, с открытой позицией.
    risk_fraction — доля счёта, которую может забрать стоп stop_rub на всех контрактах.
    cash — счёт на входе,
    он растёт и уменьшается от сделки к сделке. margin — залог на контракт.
    view — уже посчитанные уровни; иначе они строятся из channel и exit_channel.
    """
    if view is None:
        view = channel_view(frame, channel, exit_channel)
    opened = view["open"].to_numpy(dtype=float)
    high = view["high"].to_numpy(dtype=float)
    low = view["low"].to_numpy(dtype=float)
    close = view["close"].to_numpy(dtype=float)
    volume = view["volume"].to_numpy(dtype=float)
    prior_high = view["prior_high"].to_numpy(dtype=float)
    prior_low = view["prior_low"].to_numpy(dtype=float)
    prior_vol = view["prior_vol"].to_numpy(dtype=float)
    prior_range = view["prior_range"].to_numpy(dtype=float)
    exit_high = view["exit_high"].to_numpy(dtype=float)
    exit_low = view["exit_low"].to_numpy(dtype=float)
    need_clock = clock_volume or clock_cap is not None
    if not need_clock:
        clock_typical = None
    elif "clock_vol" in view.columns:
        clock_typical = view["clock_vol"].to_numpy(dtype=float)
    else:
        clock_typical = _clock_volume(view.index, volume, CLOCK_DAYS)
    trade_from = _as_date(trade_from)
    days = view.index.date
    last_day = days[-1]
    ready = view["entry_ready"].to_numpy(dtype=float)
    book = _FillBook(secid, cash)
    book.trail = trail
    for i in range(len(view)):
        step_minute(
            book,
            i,
            opened=float(opened[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(volume[i]),
            day=days[i],
            next_day=None if i + 1 >= len(view) else days[i + 1],
            last_day=last_day,
            prior_high=float(prior_high[i]),
            prior_low=float(prior_low[i]),
            prior_vol=float(prior_vol[i]),
            prior_range=float(prior_range[i]),
            exit_high=float(exit_high[i]),
            exit_low=float(exit_low[i]),
            clock_vol=float("nan") if clock_typical is None else float(clock_typical[i]),
            entry_ready=float(ready[i]),
            clearance=clearance,
            cooldown=cooldown,
            clock_volume=clock_volume,
            clock_cap=clock_cap,
            size_mode=size_mode,
            stop_mult=stop_mult,
            loss_bars=loss_bars,
            risk_fraction=risk_fraction,
            margin=margin,
            stop_rub=stop_rub,
            breakout_span=breakout_span,
            trade_from=trade_from,
            trail=trail,
            scale_step=scale_step,
            scale_floor=scale_floor,
            scale_back=scale_back,
        )
        if marks is not None:
            marks.append(marked_equity(book, float(close[i])))
    return book.trades


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


def buy_and_hold(frame: pd.DataFrame, channel: int, trade_from: date | None = None) -> float:
    """Лонг с первой возможной минуты входа до открытия последнего дня."""
    if len(frame) < channel + 2:
        return 0.0
    days = frame.index.date
    last_day = days[-1]
    entry_i = channel + 1
    trade_from = _as_date(trade_from)
    if trade_from is not None:
        while entry_i < len(frame) and days[entry_i] < trade_from:
            entry_i += 1
    if entry_i >= len(frame) or days[entry_i] == last_day:
        return 0.0
    exit_i = int(np.argmax(days == last_day))
    if exit_i <= entry_i:
        return 0.0
    return (float(frame["open"].iloc[exit_i]) - float(frame["open"].iloc[entry_i])) * MULTIPLIER - 2 * COMMISSION


def coarse_windows() -> tuple[int, ...]:
    return tuple(range(COARSE_FROM, COARSE_TO + 1, COARSE_STEP))


def _profitable_parts(row: dict[str, object]) -> int:
    parts = row["parts"]
    per_contract = row["per_contract"]
    assert isinstance(parts, tuple)
    assert isinstance(per_contract, tuple)
    return sum(
        float(part) > 0 and float(count) >= MIN_TRADES_PER_CONTRACT
        for part, count in zip(parts, per_contract, strict=True)
    )


def _eligible(row: dict[str, object]) -> bool:
    parts = row["parts"]
    assert isinstance(parts, tuple)
    return _profitable_parts(row) * 2 > len(parts)


def choose_window(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Прибыльных частей больше половины, затем лучший результат худшей части."""
    eligible = [row for row in rows if _eligible(row)]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (min(float(part) for part in row["parts"]), -int(row["channel"])))


def local_peaks(rows: list[dict[str, object]]) -> list[int]:
    """Окна, которые прошли правило и не хуже прошедших соседей по худшей части."""
    ordered = sorted(rows, key=lambda row: int(row["channel"]))
    peaks: list[int] = []
    for index, row in enumerate(ordered):
        if not _eligible(row):
            continue
        worst = min(float(part) for part in row["parts"])
        neighbors = []
        if index > 0:
            neighbors.append(ordered[index - 1])
        if index + 1 < len(ordered):
            neighbors.append(ordered[index + 1])
        neighbor_worst = [
            min(float(part) for part in item["parts"]) if _eligible(item) else float("-inf") for item in neighbors
        ]
        if all(worst >= other for other in neighbor_worst):
            peaks.append(int(row["channel"]))
    return peaks


def refine_windows(rows: list[dict[str, object]]) -> tuple[int, ...]:
    """Шаг 120 вокруг прошедших пиков и вокруг краёв грубой сетки."""
    centers = set(local_peaks(rows))
    centers.update((COARSE_FROM, COARSE_TO))
    known = {int(row["channel"]) for row in rows}
    points: set[int] = set()
    for center in centers:
        low = COARSE_FROM if center == COARSE_FROM else max(COARSE_FROM, center - REFINE_RADIUS)
        high = center + REFINE_RADIUS if center == COARSE_TO else min(COARSE_TO, center + REFINE_RADIUS)
        for channel in range(low, high + 1, REFINE_STEP):
            if channel not in known:
                points.add(channel)
    return tuple(sorted(points))


def run_grid(
    frames: dict[str, pd.DataFrame],
    windows: tuple[int, ...],
    *,
    clearance: float = 0.0,
    cooldown: int = 0,
    clock_volume: bool = False,
    trade_from: dict[str, date] | None = None,
) -> list[dict[str, object]]:
    missing = [secid for part in PARTS for secid in part if secid not in frames]
    if missing:
        raise ValueError("В данных нет контрактов: " + ", ".join(missing))
    rows = []
    for channel in windows:
        print(f"окно {channel}", flush=True)
        by_secid = {
            secid: simulate(
                secid,
                frame,
                channel,
                clearance=clearance,
                cooldown=cooldown,
                clock_volume=clock_volume,
                trade_from=None if trade_from is None else trade_from.get(secid),
            )
            for secid, frame in frames.items()
        }
        trades = [trade for secid in frames for trade in by_secid[secid]]
        parts = tuple(_net(by_secid, list(part)) for part in PARTS)
        per_contract = tuple(
            sum(len(by_secid[secid]) for secid in part) / len(part) for part in PARTS
        )
        rows.append(
            {
                "channel": channel,
                "exit": channel // 2,
                "trades": len(trades),
                "net": _net(by_secid, list(frames)),
                "parts": parts,
                "per_contract": per_contract,
                "factor": _profit_factor(trades),
            }
        )
    return rows


def _net(by_secid: dict[str, list[dict[str, object]]], secids: list[str]) -> float:
    return sum(float(trade["pnlcomm"]) for secid in secids for trade in by_secid.get(secid, ()))


def score_window(
    frames: dict[str, pd.DataFrame],
    channel: int,
    *,
    stop_mult: float | None = STOP_MULT,
    exit_channel: int | None = None,
    trail: bool = False,
    clock_cap: float | None = None,
    size_mode: str = "flat",
    trade_from: dict[str, date] | None = None,
    views: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Сумма по контрактам и пять частей. Сделки идут только в истекающем контракте."""
    by_secid = {
        secid: simulate(
            secid,
            frame,
            channel,
            stop_mult=stop_mult,
            exit_channel=exit_channel,
            trail=trail,
            clock_cap=clock_cap,
            size_mode=size_mode,
            trade_from=None if trade_from is None else trade_from.get(secid),
            view=None if views is None else views[secid],
        )
        for secid, frame in frames.items()
    }
    trades = [trade for secid in frames for trade in by_secid[secid]]
    reasons: dict[str, int] = {}
    for trade in trades:
        reason = str(trade["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    recorded_exit = max(channel // 2, 1) if exit_channel is None else exit_channel
    pnls = [float(trade["pnlcomm"]) for trade in trades]
    return {
        "channel": channel,
        "exit": recorded_exit,
        "stop_mult": stop_mult,
        "worst": min(pnls) if pnls else 0.0,
        "best": max(pnls) if pnls else 0.0,
        "trades": len(trades),
        "wins": sum(float(trade["pnlcomm"]) > 0 for trade in trades),
        "net": _net(by_secid, list(frames)),
        "parts": tuple(_net(by_secid, list(part)) for part in PARTS),
        "factor": _profit_factor(trades),
        "reasons": reasons,
        "lots": sum(int(trade.get("lots", 1)) for trade in trades),
    }


def _profit_factor(trades: list[dict[str, object]]) -> float | None:
    wins = sum(float(trade["pnlcomm"]) for trade in trades if float(trade["pnlcomm"]) > 0)
    losses = sum(float(trade["pnlcomm"]) for trade in trades if float(trade["pnlcomm"]) < 0)
    if losses == 0:
        return None
    return wins / abs(losses)


def run_contract(
    secid: str,
    frame: pd.DataFrame,
    channel: int = CHANNEL,
    *,
    stop_mult: float | None = STOP_MULT,
    exit_channel: int | None = None,
    clock_cap: float | None = None,
    size_mode: str = "flat",
    trade_from: date | None = None,
    loss_bars: int | None = None,
    risk_fraction: float | None = None,
    cash: float = START_CASH,
    margin: float = MARGIN,
    stop_rub: float | None = None,
    breakout_span: float | None = None,
    scale_step: float | None = None,
    scale_floor: float = LONG_SCALE_FLOOR,
    scale_back: float | None = None,
) -> MinuteDonchian:
    view = channel_view(_as_feed(frame), channel, exit_channel)
    view["clock_vol"] = view["clock_vol"].fillna(0.0)
    last_day = pd.Timestamp(view.index[-1]).date().isoformat()
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(
        MinuteDonchian,
        secid=secid,
        last_day=last_day,
        stop_mult=STOP_MULT if stop_mult is None else stop_mult,
        use_stop=stop_mult is not None or risk_fraction is not None,
        clock_cap=clock_cap,
        size_mode=size_mode,
        trade_from="" if trade_from is None else trade_from.isoformat(),
        loss_bars=0 if loss_bars is None else loss_bars,
        risk_fraction=0.0 if risk_fraction is None else risk_fraction,
        margin=margin,
        stop_rub=0.0 if stop_rub is None else stop_rub,
        breakout_span=0.0 if breakout_span is None else breakout_span,
        scale_step=0.0 if scale_step is None else scale_step,
        scale_floor=scale_floor,
        scale_back=0.0 if scale_back is None else scale_back,
    )
    cerebro.adddata(SignalData(dataname=view))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(
        commission=COMMISSION,
        commtype=bt.CommInfoBase.COMM_FIXED,
        mult=MULTIPLIER,
        margin=margin,
        stocklike=False,
    )
    return cerebro.run()[0]


def run_all(
    frames: dict[str, pd.DataFrame],
    channel: int = CHANNEL,
    *,
    stop_mult: float | None = STOP_MULT,
    exit_channel: int | None = None,
    clock_cap: float | None = None,
    size_mode: str = "flat",
    trade_from: dict[str, date] | None = None,
) -> dict[str, object]:
    results = []
    for secid, frame in frames.items():
        print(secid, flush=True)
        strategy = run_contract(
            secid,
            frame,
            channel,
            stop_mult=stop_mult,
            exit_channel=exit_channel,
            clock_cap=clock_cap,
            size_mode=size_mode,
            trade_from=None if trade_from is None else trade_from.get(secid),
        )
        gross = sum(float(trade["pnl"]) for trade in strategy.trades)
        net = sum(float(trade["pnlcomm"]) for trade in strategy.trades)
        front = None if trade_from is None else trade_from.get(secid)
        results.append(
            {
                "secid": secid,
                "trades": strategy.trades,
                "gross": gross,
                "net": net,
                "hold": buy_and_hold(frame, channel, front),
                "values": strategy.values,
            }
        )
    trades = [trade for item in results for trade in item["trades"]]
    recorded_exit = max(channel // 2, 1) if exit_channel is None else exit_channel
    return {
        "contracts": results,
        "trades": trades,
        "channel": channel,
        "exit": recorded_exit,
        "stop_mult": stop_mult,
        "clock_cap": clock_cap,
        "size_mode": size_mode,
    }


def run_account(
    frames: dict[str, pd.DataFrame],
    window: Window,
    trade_from: dict[str, date] | None = None,
) -> dict[str, object]:
    """Один счёт на все контракты: размер следующей сделки зависит от результата предыдущей."""
    equity = window.cash
    results = []
    for secid, frame in frames.items():
        print(secid, flush=True)
        front = None if trade_from is None else trade_from.get(secid)
        strategy = run_contract(
            secid,
            frame,
            window.channel,
            stop_mult=window.stop_mult,
            exit_channel=window.exit_channel,
            clock_cap=window.clock_cap,
            size_mode=window.size_mode,
            trade_from=front,
            loss_bars=window.loss_bars,
            risk_fraction=window.risk_fraction,
            cash=equity,
            margin=window.margin,
            stop_rub=window.stop_rub,
            breakout_span=window.breakout_span,
            scale_step=window.scale_step,
            scale_floor=window.scale_floor,
            scale_back=window.scale_back,
        )
        equity = float(strategy.broker.getvalue())
        gross = sum(float(trade["pnl"]) for trade in strategy.trades)
        net = sum(float(trade["pnlcomm"]) for trade in strategy.trades)
        results.append(
            {
                "secid": secid,
                "trades": strategy.trades,
                "gross": gross,
                "net": net,
                "hold": buy_and_hold(frame, window.channel, front),
                "values": strategy.values,
            }
        )
    trades = [trade for item in results for trade in item["trades"]]
    return {
        "contracts": results,
        "trades": trades,
        "channel": window.channel,
        "exit": window.exit_channel,
        "stop_mult": window.stop_mult,
        "clock_cap": window.clock_cap,
        "size_mode": window.size_mode,
        "loss_bars": window.loss_bars,
        "risk_fraction": window.risk_fraction,
        "stop_rub": window.stop_rub,
        "breakout_span": window.breakout_span,
        "scale_step": window.scale_step,
        "scale_floor": window.scale_floor,
        "scale_back": window.scale_back,
        "cash": window.cash,
        "equity": equity,
        "margin": window.margin,
    }


def _max_drawdown(values: list[tuple[object, float]], start: float = START_CASH) -> float:
    peak = start
    worst = 0.0
    for _moment, value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def _money(value: float) -> str:
    return f"{value:,.0f}"


def _part_labels() -> str:
    return "  ".join(f"{secids[0]}–{secids[-1]}" for secids in PARTS)


def grid_report(rows: list[dict[str, object]], title: str) -> str:
    chosen = choose_window(rows)
    lines = [
        title,
        "Выход — половина окна. Объём минуты сигнала не ниже медианы окна.",
        f"Годится окно, если прибыльных частей больше половины. Часть прибыльна при результате больше нуля и не меньше {MIN_TRADES_PER_CONTRACT:.0f} сделок на контракт.",
        _part_labels(),
        "Сделки только в истекающем контракте. Месяц до него прогревает окно и не торгуется.",
        "",
        f"{'N':>6} {'СДЕЛОК':>7} {'ИТОГ':>10} "
        + " ".join(f"{'Ч' + str(index):>9}" for index in range(1, 6))
        + f" {'ХУДШАЯ':>9} {'ФАКТОР':>7}",
    ]
    for row in sorted(rows, key=lambda item: int(item["channel"])):
        parts = tuple(float(part) for part in row["parts"])
        factor = row["factor"]
        factor_text = "—" if factor is None else f"{float(factor):.2f}"
        mark = "  ←" if chosen is not None and int(row["channel"]) == int(chosen["channel"]) else ""
        part_text = " ".join(f"{_money(part):>9}" for part in parts)
        lines.append(
            f"{int(row['channel']):>6} {int(row['trades']):>7} {_money(float(row['net'])):>10} "
            f"{part_text} {_money(min(parts)):>9} {factor_text:>7}{mark}"
        )
    lines.append("")
    if chosen is None:
        lines.append("В этой сетке ни одно окно не прошло правило.")
    else:
        worst = min(float(part) for part in chosen["parts"])
        lines.append(f"Лучшее в этой сетке: N={int(chosen['channel'])}, худшая часть {_money(worst)} руб.")
    return "\n".join(lines)


def filter_report(frames: dict[str, pd.DataFrame], channel: int) -> str:
    """Три фильтра по отдельности на уже выбранном окне."""
    variants = (
        ("как есть", {}),
        ("пауза после стопа", {"cooldown": channel}),
        ("запас пробоя 0,5 диапазона", {"clearance": CLEARANCE}),
        ("объём той же минуты за 5 дней", {"clock_volume": True}),
    )
    lines = [
        f"Фильтры по одному на окне {channel}. Объём последних {channel} минут остаётся, кроме последней строки: там он заменён.",
        f"{'ФИЛЬТР':<32} {'СДЕЛОК':>7} {'ИТОГ':>10} {'ХУДШАЯ':>9} {'ПЛЮС':>5}",
    ]
    for name, options in variants:
        print(f"фильтр: {name}", flush=True)
        rows = run_grid(frames, (channel,), **options)
        row = rows[0]
        parts = tuple(float(part) for part in row["parts"])
        plus = sum(part > 0 for part in parts)
        lines.append(
            f"{name:<32} {int(row['trades']):>7} {_money(float(row['net'])):>10} {_money(min(parts)):>9} {plus:>5}/5"
        )
    return "\n".join(lines)


def report(summary: dict[str, object]) -> str:
    trades = summary["trades"]
    assert isinstance(trades, list)
    contracts = summary["contracts"]
    assert isinstance(contracts, list)
    channel = int(summary["channel"])
    exit_channel = int(summary.get("exit", channel // 2))
    stop_mult = summary.get("stop_mult", STOP_MULT)
    net = sum(float(trade["pnlcomm"]) for trade in trades)
    gross = sum(float(trade["pnl"]) for trade in trades)
    hold = sum(float(item["hold"]) for item in contracts)
    longs = [trade for trade in trades if trade["direction"] == "long"]
    shorts = [trade for trade in trades if trade["direction"] == "short"]
    wins = [trade for trade in trades if float(trade["pnlcomm"]) > 0]
    factor = _profit_factor(trades)
    factor_text = "—" if factor is None else f"{factor:.2f}"
    risk_fraction = summary.get("risk_fraction")
    breakout_span = summary.get("breakout_span")
    account = bool(risk_fraction or breakout_span) and summary.get("equity") is not None
    if account:
        stitched = [point for item in contracts for point in item["values"]]
        worst = _max_drawdown(stitched, start=float(summary.get("cash", START_CASH)))
    else:
        worst = min((_max_drawdown(item["values"]) for item in contracts), default=0.0)
    clock_cap = summary.get("clock_cap")
    size_mode = str(summary.get("size_mode", "flat"))
    stop_rub = summary.get("stop_rub")
    if risk_fraction and stop_rub:
        stop_text = f"стоп {float(stop_rub):.0f} руб. на контракт, не больше {float(risk_fraction):.0%} счёта"
    elif stop_mult is None:
        stop_text = "защитного стопа нет"
    else:
        stop_text = f"стоп {float(stop_mult):.0f} медиан минутного диапазона"
    if exit_channel <= 0:
        exit_text = "выход по каналу выключен"
    else:
        exit_text = f"выход по каналу {exit_channel} минут"
    loss_bars = summary.get("loss_bars")
    if breakout_span:
        size_text = (
            f"0 медиан пробоя — 100% контрактов по залогу {float(summary.get('margin', 0)):,.0f} руб., "
            f"{LONG_BREAKOUT_LOW:g}–{LONG_BREAKOUT_HIGH:g} медианы — 50%, {float(breakout_span):.0f} медиан — 0%, "
            f"старт {float(summary.get('cash', 0)):,.0f} руб"
        )
    elif risk_fraction:
        size_text = (
            f"контрактов столько, сколько проходит в {float(risk_fraction):.0%} счёта, "
            f"старт {float(summary.get('cash', 0)):,.0f} руб"
        )
    elif size_mode == "inverse":
        size_text = "лоты 3, 2 или 1: чем ближе пробой к границе канала, тем больше"
    else:
        size_text = "1 лот"
    if clock_cap is None:
        volume_text = "Объём минуты сигнала не ниже медианы окна."
    else:
        volume_text = (
            f"Объём минуты сигнала не ниже медианы окна и не выше {float(clock_cap):g} "
            f"медиан той же минуты за {CLOCK_DAYS} дней."
        )
    part_nets = _part_nets(trades)
    lines = [
        f"Минутный канал {channel}, {exit_text}, {stop_text}. {size_text}.",
        volume_text,
        f"Комиссия {COMMISSION:.0f} руб. за контракт за сторону. "
        f"Позиция меняется по {FILL_PER_MINUTE} контрактов в минуту, по закрытию этой минуты. "
        "В последний день новая позиция не открывается.",
        "Сделки только в истекающем контракте. Месяц до него прогревает окно и не торгуется.",
    ]
    if loss_bars:
        lines.append(f"Если через {int(loss_bars)} минут сделка всё ещё в минусе, она закрывается.")
    scale_step = summary.get("scale_step")
    if scale_step:
        scaled = sum(bool(trade.get("scaled")) for trade in trades)
        back = summary.get("scale_back")
        back_text = f"{float(scale_step) / 2:.0f}" if back is None else f"{float(back):.0f}"
        lines.append(
            f"Откат от лучшей цены сделки на {float(scale_step):.0f} медиан минутного диапазона "
            f"уменьшает позицию до {float(summary.get('scale_floor', LONG_SCALE_FLOOR)):.0%} исходного размера. "
            f"Когда откат сжимается до {back_text} медиан, размер возвращается. "
            f"Так сработало в {scaled} сделках."
        )
    if account and summary.get("equity") is not None:
        lines.append(f"Счёт в конце: {float(summary['equity']):,.0f} руб.")
    lines.extend([
        "Части: " + ", ".join(f"Ч{index} {_money(part)}" for index, part in enumerate(part_nets, start=1)),
        f"Сделок: {len(trades)}. Прибыльных: {len(wins)}.",
        _lots_line(trades),
        f"Лонгов: {len(longs)}, результат {sum(float(trade['pnlcomm']) for trade in longs):,.0f} руб.",
        f"Шортов: {len(shorts)}, результат {sum(float(trade['pnlcomm']) for trade in shorts):,.0f} руб.",
        f"Без комиссии: {gross:,.0f} руб. После комиссии: {net:,.0f} руб.",
        f"Фактор прибыли: {factor_text}. "
        + (
            f"Худшая просадка счёта: {worst:,.0f} руб."
            if account
            else f"Худшая просадка одного контракта: {worst:,.0f} руб."
        ),
        f"Просто лонг на тех же историях: {hold:,.0f} руб.",
        "",
        f"{'SECID':<8} {'СДЕЛОК':>7} {'СТРАТЕГИЯ':>12} {'ЛОНГ':>12}",
    ])
    for item in contracts:
        lines.append(
            f"{item['secid']:<8} {len(item['trades']):>7} {item['net']:>12,.0f} {item['hold']:>12,.0f}"
        )
    reasons: dict[str, int] = {}
    for trade in trades:
        reason = str(trade["reason"] or "—")
        reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        lines.append("")
        lines.append("Выходы: " + ", ".join(f"{name} {count}" for name, count in sorted(reasons.items())))
    return "\n".join(lines)


def _lots_line(trades: list[dict[str, object]]) -> str:
    if not trades:
        return "Лотов в сделке: —."
    def _avg(rows: list[dict[str, object]]) -> float:
        if not rows:
            return 0.0
        return sum(int(trade.get("lots", 1)) for trade in rows) / len(rows)

    losses = [trade for trade in trades if float(trade["pnlcomm"]) < 0]
    wins = [trade for trade in trades if float(trade["pnlcomm"]) > 0]
    return (
        f"Средний размер: {_avg(trades):.2f} лота. "
        f"У прибыльных {_avg(wins):.2f}, у убыточных {_avg(losses):.2f}."
    )


def _part_nets(trades: list[dict[str, object]]) -> tuple[float, ...]:
    by_secid: dict[str, list[dict[str, object]]] = {}
    for trade in trades:
        by_secid.setdefault(str(trade["secid"]), []).append(trade)
    return tuple(_net(by_secid, list(part)) for part in PARTS)


def _reason_text(reasons: object) -> str:
    assert isinstance(reasons, dict)
    order = ("stop", "channel", "expiry")
    parts = [f"{name} {int(reasons[name])}" for name in order if name in reasons]
    extra = [f"{name} {int(count)}" for name, count in sorted(reasons.items()) if name not in order]
    return ", ".join(parts + extra) if parts or extra else "—"


def _score_line(row: dict[str, object]) -> str:
    parts = tuple(float(part) for part in row["parts"])
    factor = row["factor"]
    factor_text = "—" if factor is None else f"{float(factor):.2f}"
    part_text = " ".join(f"{_money(part):>9}" for part in parts)
    return (
        f"{int(row['channel']):>6} {int(row['exit']):>6} {float(row['stop_mult']):>4.1f} "
        f"{int(row['trades']):>7} {int(row['wins']):>5} {_money(float(row['net'])):>10} "
        f"{part_text} {factor_text:>7}  {_reason_text(row['reasons'])}"
    )


def exit_grid(frames: dict[str, pd.DataFrame], trade_from: dict[str, date] | None = None) -> str:
    """Точная сумма длинной пары и сетка выхода на двух окнах по общему итогу."""
    print("длинная пара, выход N/2, стоп 2.5", flush=True)
    long_rows = [score_window(frames, channel, trade_from=trade_from) for channel in LONG_CANDIDATES]
    long_row = max(long_rows, key=lambda row: float(row["net"]))
    long_channel = int(long_row["channel"])
    windows = (SHORT_WINDOW, long_channel)
    header = (
        f"{'ВХОД':>6} {'ВЫХОД':>6} {'СТОП':>4} {'СДЕЛОК':>7} {'ПЛЮС':>5} {'ИТОГ':>10} "
        + " ".join(f"{'Ч' + str(index):>9}" for index in range(1, 6))
        + f" {'ФАКТОР':>7}  ВЫХОДЫ"
    )
    lines = [
        "Длинная пара при выходе N/2 и стопе 2,5. Берётся окно с большей точной суммой.",
        header,
    ]
    for row in long_rows:
        mark = "  ←" if int(row["channel"]) == long_channel else ""
        lines.append(_score_line(row) + mark)
    lines.append("")
    lines.append(
        f"Грубая сетка выхода на входах {SHORT_WINDOW} и {long_channel}. "
        "Сумма окон — не один счёт. Объём и вход не меняются."
    )
    lines.append(header)
    baselines: dict[int, dict[str, object]] = {}
    by_key: dict[tuple[int, float], dict[int, dict[str, object]]] = {}
    for channel in windows:
        for scale in EXIT_SCALES:
            exit_channel = scaled_exit(channel, scale)
            print(f"вход {channel}, выход {exit_channel}", flush=True)
            views = {
                secid: channel_view(frame, channel, exit_channel) for secid, frame in frames.items()
            }
            for stop_mult in STOP_GRID:
                row = score_window(
                    frames,
                    channel,
                    stop_mult=stop_mult,
                    exit_channel=exit_channel,
                    trade_from=trade_from,
                    views=views,
                )
                row["scale"] = scale
                lines.append(_score_line(row))
                by_key.setdefault((scale, stop_mult), {})[channel] = row
                if scale == 2 and stop_mult == STOP_MULT:
                    baselines[channel] = row
        lines.append("")
    adopted = _adopted_exit(by_key, baselines, windows)
    lines.append(_exit_decision(adopted, baselines, windows))
    return "\n".join(lines)


def _adopted_exit(
    by_key: dict[tuple[int, float], dict[int, dict[str, object]]],
    baselines: dict[int, dict[str, object]],
    windows: tuple[int, ...],
) -> tuple[int, float] | None:
    """Вариант, который поднимает итог каждого окна. При нескольких — больший из меньших приростов."""
    best_key: tuple[int, float] | None = None
    best_rank: tuple[float, float] | None = None
    for key, pair in by_key.items():
        if key == (2, STOP_MULT):
            continue
        gains = [float(pair[channel]["net"]) - float(baselines[channel]["net"]) for channel in windows]
        if min(gains) <= 0:
            continue
        rank = (min(gains), sum(gains))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_key = key
    return best_key


def _exit_decision(
    adopted: tuple[int, float] | None,
    baselines: dict[int, dict[str, object]],
    windows: tuple[int, ...],
) -> str:
    base = ", ".join(
        f"{channel}: {_money(float(baselines[channel]['net']))} руб." for channel in windows
    )
    if adopted is None:
        coarse = f"Внутри стопов 1,5–8 ни одна пара не подняла оба итога. База: {base}"
    else:
        scale, stop_mult = adopted
        coarse = (
            f"Внутри этой грубой сетки оба итога выше при выходе {scale}/4 окна и стопе {stop_mult:.1f}. "
            f"База (N/2, стоп {STOP_MULT}): {base}"
        )
    locked = ", ".join(
        "вход {channel}, выход {exit_channel}, стоп {stop}".format(
            channel=window.channel,
            exit_channel=window.exit_channel,
            stop="нет" if window.stop_mult is None else f"{float(window.stop_mult):g}",
        )
        for window in WINDOWS
    )
    return coarse + " Рабочие выходы после более широкой проверки: " + locked + "."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Канал по закрытиям минуток CNY/RUB")
    parser.add_argument("--grid", action="store_true", help="Прогнать сетку окон и выбрать N")
    parser.add_argument("--exits", action="store_true", help="Сравнить выходы на двух окнах по общему итогу")
    parser.add_argument("--bars", type=Path, default=BARS_DIR, help="Каталог parquet по контрактам")
    args = parser.parse_args(argv)
    frames = load_minutes(args.bars)
    starts = load_trade_starts(args.bars)
    if args.exits:
        print(exit_grid(frames, starts))
        return 0
    if args.grid:
        around_short = tuple(range(480, 1440 + 1, 20))
        around_long = tuple(range(11_520, 13_440 + 1, 120))
        short = run_grid(frames, around_short, trade_from=starts)
        print(grid_report(short, "Вокруг 960: 480–1440, шаг 20."))
        print()
        long = run_grid(frames, around_long, trade_from=starts)
        print(grid_report(long, "Вокруг 12480: 11520–13440, шаг 120."))
        chosen = choose_window(short + long)
        print()
        if chosen is None:
            print("Ни одно окно не прошло правило: прибыльных частей должно быть больше половины.")
            return 0
        print(
            f"Выбрано N={int(chosen['channel'])}: прибыльных частей {_profitable_parts(chosen)} из 5, "
            f"худшая часть {_money(min(float(part) for part in chosen['parts']))} руб."
        )
        return 0
    for window in WINDOWS:
        if window.risk_fraction or window.breakout_span:
            print(report(run_account(frames, window, starts)))
        else:
            print(
                report(
                    run_all(
                        frames,
                        window.channel,
                        stop_mult=window.stop_mult,
                        exit_channel=window.exit_channel,
                        clock_cap=window.clock_cap,
                        size_mode=window.size_mode,
                        trade_from=starts,
                    )
                )
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
