"""Стратегии видят чужую многодневную позицию и пропускают пробой против неё.

Пять окон из backtest.py остаются раздельными счетами. Меняется только вход:
если другая стратегия уже держит позицию несколько торговых дней и плюс по ней
растёт, новый сигнал в обратную сторону считается ложным и не открывается.
Выходы, стопы и размер своих сделок не трогаются. Живой счёт этот модуль не
читает и не меняет.

День считается торговым: это смена даты в минутках контракта, без выходных.
«Держит N дней» значит, что с дня первого исполнения прошёл по крайней мере
N таких дней. «Плюс» — открытый результат в рублях больше нуля. «Выше вчера» —
он больше закрытия прошлого торгового дня. «Растёт» мягче: на закрытии прошлого дня плюс уже был больше позавчерашнего,
а сегодня позиция всё ещё в плюсе. Пробой, который снимает часть плюса,
но не уводит её в минус, всё ещё гасится.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    PARTS,
    WINDOWS,
    Window,
    _money,
    _part_nets,
    _profit_factor,
    channel_view,
    load_minutes,
    load_trade_starts,
)
from cnyrub.engine import MULTIPLIER, _FillBook, marked_equity, step_minute


NAMES = ("short", "long", "thirty", "fortyfive", "sixty")


@dataclass(frozen=True)
class ConsensusRule:
    """Кого слушаем и когда чужая позиция гасит обратный вход."""

    id: str
    title: str
    hold_days: int
    growing: str = "no"
    source: str = "any"
    shielded: frozenset[str] = field(default_factory=frozenset)
    min_ranges: float = 0.0
    enabled: bool = True


RULES: tuple[ConsensusRule, ...] = (
    ConsensusRule("base", "без фильтра", 0, enabled=False),
    ConsensusRule("d1-grows", "1 день, плюс растёт", 1, "soft"),
    ConsensusRule("d2-plus", "2 дня, только плюс", 2, "no"),
    ConsensusRule("d2-grows", "2 дня, плюс растёт", 2, "soft"),
    ConsensusRule("d3-plus", "3 дня, только плюс", 3, "no"),
    ConsensusRule("d3-strict", "3 дня, плюс выше вчера", 3, "strict"),
    ConsensusRule("d3-grows", "3 дня, плюс растёт", 3, "soft"),
    ConsensusRule("d5-grows", "5 дней, плюс растёт", 5, "soft"),
    ConsensusRule("long-d3", "глушит только окно 12420", 3, "soft", source="long"),
    ConsensusRule("slower-d3", "глушит только старшее окно", 3, "soft", source="slower"),
    ConsensusRule(
        "fast-d3",
        "глушит только 30/45/60",
        3,
        "soft",
        shielded=frozenset({"short", "long"}),
    ),
    ConsensusRule("wide-d3", "3 дня, плюс и 20 диапазонов", 3, "soft", min_ranges=20.0),
)


@dataclass(frozen=True)
class OpenState:
    """Чужая позиция на закрытии этой минуты, уже после стопа и выхода."""

    name: str
    channel: int
    side: int
    held_days: int
    pnl: float
    prev_pnl: float | None
    prev2_pnl: float | None
    ranges: float


@dataclass
class StrategyStats:
    name: str
    trades: list[dict[str, object]]
    blocked_bars: int
    blocked_setups: int
    veto_minutes: int
    end_equity: float
    drawdown: float
    cash: float

    @property
    def net(self) -> float:
        return sum(float(trade["pnlcomm"]) for trade in self.trades)

    @property
    def factor(self) -> float | None:
        return _profit_factor(self.trades)


@dataclass
class ConsensusResult:
    rule: ConsensusRule
    strategies: list[StrategyStats]
    combined_drawdown: float
    source_setups: dict[str, int]

    @property
    def trades(self) -> list[dict[str, object]]:
        return [trade for stats in self.strategies for trade in stats.trades]

    @property
    def net(self) -> float:
        return sum(stats.net for stats in self.strategies)

    @property
    def blocked_setups(self) -> int:
        return sum(stats.blocked_setups for stats in self.strategies)


def open_pnl(book: _FillBook, close: float) -> float:
    """Открытый результат позиции в рублях, без уже списанной комиссии."""
    if book.held == 0:
        return 0.0
    return (close - book.avg) * book.held * MULTIPLIER


def still_in(book: _FillBook) -> bool:
    """Позиция ещё нужна стратегии: не стоп, не выход и не добор в ноль."""
    if book.held == 0 or book.target == 0:
        return False
    return (book.held > 0) == (book.target > 0)


def position_state(
    name: str,
    channel: int,
    book: _FillBook,
    index: int,
    day_index: np.ndarray,
    close: float,
    prev_pnl: float | None,
    prev2_pnl: float | None,
) -> OpenState | None:
    """Снимок подтверждающей позиции. Новая цель без исполнения сюда не входит."""
    if not still_in(book) or book.entry_i is None:
        return None
    side = 1 if book.held > 0 else -1
    favor = (close - book.avg) * side
    ranges = favor / book.unit if book.unit > 0 else 0.0
    return OpenState(
        name=name,
        channel=channel,
        side=side,
        held_days=int(day_index[index] - day_index[int(book.entry_i)]),
        pnl=open_pnl(book, close),
        prev_pnl=prev_pnl,
        prev2_pnl=prev2_pnl,
        ranges=ranges,
    )


def _profit_grew(state: OpenState, growing: str) -> bool:
    """Строго — плюс выше вчера. Мягко — вчера он уже рос, а сегодня позиция ещё в плюсе."""
    if growing == "no":
        return True
    if growing not in {"strict", "soft"}:
        raise ValueError(f"неизвестный режим плюса: {growing}")
    if state.prev_pnl is None:
        return False
    if state.pnl > state.prev_pnl:
        return True
    if growing == "soft" and state.prev2_pnl is not None:
        return state.prev_pnl > state.prev2_pnl
    return False


def confirms(state: OpenState, rule: ConsensusRule) -> bool:
    """Позиция достаточно старая, в плюсе и, если правило просит, этот плюс растёт."""
    if state.held_days < rule.hold_days or not state.pnl > 0:
        return False
    if state.ranges < rule.min_ranges:
        return False
    return _profit_grew(state, rule.growing)


def _source_ok(state: OpenState, rule: ConsensusRule, name: str, channel: int) -> bool:
    if state.name == name:
        return False
    if rule.source == "long":
        return state.name == "long"
    if rule.source == "slower":
        return state.channel > channel
    if rule.source != "any":
        raise ValueError(f"неизвестный источник правила: {rule.source}")
    return True


def veto_side(states: list[OpenState], rule: ConsensusRule, name: str, channel: int) -> int:
    """1 — не покупать, -1 — не продавать, 2 — не открывать ни туда ни туда, 0 — пускать."""
    if not rule.enabled or name in rule.shielded:
        return 0
    block_long = False
    block_short = False
    for state in states:
        if not _source_ok(state, rule, name, channel) or not confirms(state, rule):
            continue
        if state.side > 0:
            block_short = True
        else:
            block_long = True
    if block_long and block_short:
        return 2
    if block_long:
        return 1
    if block_short:
        return -1
    return 0


def veto_sources(states: list[OpenState], rule: ConsensusRule, name: str, channel: int, side: int) -> list[str]:
    """Имена стратегий, из-за которых этот вход в сторону side считается ложным."""
    found: list[str] = []
    for state in states:
        if not _source_ok(state, rule, name, channel) or not confirms(state, rule):
            continue
        if side > 0 and state.side < 0:
            found.append(state.name)
        elif side < 0 and state.side > 0:
            found.append(state.name)
    return found


def cancel_entry(book: _FillBook) -> None:
    """Снять цель, которую эта минута только что поставила. Исполнения ещё не было."""
    book.target = 0
    book.base = 0
    book.entry = ""
    book.stop_dist = None
    book.stop_px = None
    book.unit = 0.0
    book.best = None
    book.scale_level = 0
    book.scaled = False
    book.reason = ""


def suppress_entry(book: _FillBook, veto: int) -> bool:
    """Убрать свежий вход, если запрет смотрит в его сторону. Открытую позицию не трогать."""
    if veto == 0 or book.held != 0 or book.target == 0:
        return False
    side = 1 if book.target > 0 else -1
    if veto != 2 and veto != side:
        return False
    cancel_entry(book)
    return True


def _day_index(days: np.ndarray) -> np.ndarray:
    index = np.empty(len(days), dtype=np.int32)
    if len(days) == 0:
        return index
    current = 0
    index[0] = 0
    for i in range(1, len(days)):
        if days[i] != days[i - 1]:
            current += 1
        index[i] = current
    return index


class _Drawdown:
    def __init__(self, start: float) -> None:
        self.peak = start
        self.worst = 0.0

    def add(self, value: float) -> None:
        if value > self.peak:
            self.peak = value
        drop = value - self.peak
        if drop < self.worst:
            self.worst = drop


def _arrays(view: pd.DataFrame) -> dict[str, np.ndarray]:
    def column(name: str) -> np.ndarray:
        if name in view.columns:
            return view[name].to_numpy(dtype=float)
        return np.full(len(view), np.nan)

    return {
        "open": view["open"].to_numpy(dtype=float),
        "high": view["high"].to_numpy(dtype=float),
        "low": view["low"].to_numpy(dtype=float),
        "close": view["close"].to_numpy(dtype=float),
        "volume": view["volume"].to_numpy(dtype=float),
        "prior_high": view["prior_high"].to_numpy(dtype=float),
        "prior_low": view["prior_low"].to_numpy(dtype=float),
        "prior_vol": view["prior_vol"].to_numpy(dtype=float),
        "prior_range": view["prior_range"].to_numpy(dtype=float),
        "exit_high": view["exit_high"].to_numpy(dtype=float),
        "exit_low": view["exit_low"].to_numpy(dtype=float),
        "clock_vol": column("clock_vol"),
        "entry_ready": view["entry_ready"].to_numpy(dtype=float),
        "drift": column("drift"),
        "path": column("path"),
        "surge_vol": column("surge_vol"),
    }


def build_views(
    frames: dict[str, pd.DataFrame],
    names: tuple[str, ...] = NAMES,
    windows: tuple[Window, ...] = WINDOWS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Уровни каналов один раз: правила отличаются запретом, а не окном."""
    views: dict[str, dict[str, pd.DataFrame]] = {}
    for secid, frame in frames.items():
        print(f"окна {secid}", flush=True)
        views[secid] = {
            name: channel_view(frame, window.channel, window.exit_channel)
            for name, window in zip(names, windows, strict=True)
        }
    return views


def run_consensus(
    frames: dict[str, pd.DataFrame],
    rule: ConsensusRule,
    trade_from: dict[str, object] | None = None,
    *,
    names: tuple[str, ...] = NAMES,
    windows: tuple[Window, ...] = WINDOWS,
    views: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> ConsensusResult:
    """Один проход по контрактам. Счета не смешиваются, запрет общий на минуту."""
    if len(names) != len(windows):
        raise ValueError("имён стратегий и окон должно быть поровну")
    channels = {name: window.channel for name, window in zip(names, windows, strict=True)}
    equity = {name: window.cash for name, window in zip(names, windows, strict=True)}
    trades: dict[str, list[dict[str, object]]] = {name: [] for name in names}
    blocked_bars = dict.fromkeys(names, 0)
    blocked_setups = dict.fromkeys(names, 0)
    veto_minutes = dict.fromkeys(names, 0)
    drawdowns = {name: _Drawdown(window.cash) for name, window in zip(names, windows, strict=True)}
    combined = _Drawdown(sum(window.cash for window in windows))
    source_setups = dict.fromkeys(names, 0)
    if views is None:
        views = build_views(frames, names, windows)

    for secid, frame in frames.items():
        prepared = views[secid]
        sample = prepared[names[0]]
        days = sample.index.date
        day_index = _day_index(days)
        last_day = days[-1]
        packs = {name: _arrays(prepared[name]) for name in names}
        books = {name: _FillBook(secid, equity[name]) for name in names}
        prev_pnl: dict[str, float | None] = dict.fromkeys(names)
        prev2_pnl: dict[str, float | None] = dict.fromkeys(names)
        streak = dict.fromkeys(names, 0)
        front = None if trade_from is None else trade_from.get(secid)
        count = len(sample)
        for i in range(count):
            close = float(packs[names[0]]["close"][i])
            for name, window in zip(names, windows, strict=True):
                pack = packs[name]
                step_minute(
                    books[name],
                    i,
                    opened=float(pack["open"][i]),
                    high=float(pack["high"][i]),
                    low=float(pack["low"][i]),
                    close=float(pack["close"][i]),
                    volume=float(pack["volume"][i]),
                    day=days[i],
                    next_day=None if i + 1 >= count else days[i + 1],
                    last_day=last_day,
                    prior_high=float(pack["prior_high"][i]),
                    prior_low=float(pack["prior_low"][i]),
                    prior_vol=float(pack["prior_vol"][i]),
                    prior_range=float(pack["prior_range"][i]),
                    exit_high=float(pack["exit_high"][i]),
                    exit_low=float(pack["exit_low"][i]),
                    clock_vol=float(pack["clock_vol"][i]),
                    entry_ready=float(pack["entry_ready"][i]),
                    clearance=0.0,
                    cooldown=0,
                    clock_volume=False,
                    clock_cap=window.clock_cap,
                    size_mode=window.size_mode,
                    stop_mult=window.stop_mult,
                    loss_bars=window.loss_bars,
                    risk_fraction=window.risk_fraction,
                    margin=window.margin,
                    stop_rub=window.stop_rub,
                    breakout_span=window.breakout_span,
                    trade_from=front,
                    trail=False,
                    scale_step=window.scale_step,
                    scale_floor=window.scale_floor,
                    scale_back=window.scale_back,
                    drift=float(pack["drift"][i]),
                    path=float(pack["path"][i]),
                    surge_vol=float(pack["surge_vol"][i]),
                    eff_low=window.eff_low,
                    eff_high=window.eff_high,
                    surge_cap=window.surge_cap,
                    leverage=window.leverage,
                )
            states = []
            for name, window in zip(names, windows, strict=True):
                state = position_state(
                    name,
                    window.channel,
                    books[name],
                    i,
                    day_index,
                    close,
                    prev_pnl[name],
                    prev2_pnl[name],
                )
                if state is not None:
                    states.append(state)
            marked = 0.0
            for name in names:
                book = books[name]
                veto = veto_side(states, rule, name, channels[name])
                if veto:
                    veto_minutes[name] += 1
                fresh_side = 1 if book.target > 0 else -1
                if suppress_entry(book, veto):
                    blocked_bars[name] += 1
                    if streak[name] != fresh_side:
                        blocked_setups[name] += 1
                        streak[name] = fresh_side
                        for source in veto_sources(states, rule, name, channels[name], fresh_side):
                            source_setups[source] += 1
                else:
                    streak[name] = 0
                value = marked_equity(book, close)
                drawdowns[name].add(value)
                marked += value
            combined.add(marked)
            if i + 1 == count or days[i + 1] != days[i]:
                for name in names:
                    book = books[name]
                    prev2_pnl[name] = prev_pnl[name]
                    prev_pnl[name] = open_pnl(book, close) if still_in(book) else None
        for name in names:
            equity[name] = marked_equity(books[name], float(packs[name]["close"][-1]))
            trades[name].extend(books[name].trades)

    strategies = [
        StrategyStats(
            name=name,
            trades=trades[name],
            blocked_bars=blocked_bars[name],
            blocked_setups=blocked_setups[name],
            veto_minutes=veto_minutes[name],
            end_equity=equity[name],
            drawdown=drawdowns[name].worst,
            cash=window.cash,
        )
        for name, window in zip(names, windows, strict=True)
    ]
    return ConsensusResult(
        rule=rule,
        strategies=strategies,
        combined_drawdown=combined.worst,
        source_setups=source_setups,
    )


def conclusion(results: list[ConsensusResult]) -> str:
    """Короткий итог набора: где сумма выше базы и где просадка меньше."""
    base = next((item for item in results if item.rule.id == "base"), results[0])
    others = [item for item in results if item.rule.id != "base"]
    if not others:
        return "В наборе только база, сравнивать не с чем."
    best = max(others, key=lambda item: item.net)
    calmest = max(others, key=lambda item: item.combined_drawdown)
    above = [item for item in others if item.net > base.net + 0.5]
    lines = [
        f"База без фильтра: итог {_money(base.net)} руб., просадка суммы счетов {_money(base.combined_drawdown)} руб.",
    ]
    if above:
        lines.append(
            "Выше базы по сумме: "
            + "; ".join(f"«{item.rule.title}» {_signed(item.net - base.net)}" for item in above)
            + "."
        )
    else:
        lines.append("Ни одно правило не дало сумму выше базы.")
    best_parts = _part_nets(best.trades)
    base_parts = _part_nets(base.trades)
    weaker = [index for index, (left, right) in enumerate(zip(base_parts, best_parts, strict=True), start=1) if right < left - 0.5]
    weaker_text = ", ".join(f"Ч{index}" for index in weaker) if weaker else "ни в одной"
    lines.append(
        f"Наибольшая сумма — «{best.rule.title}»: {_money(best.net)} руб., {_signed(best.net - base.net)} к базе. "
        f"Части, где оно слабее базы: {weaker_text}."
    )
    lines.append(
        f"Меньше всех просела сумма при «{calmest.rule.title}»: {_money(calmest.combined_drawdown)} руб., "
        f"итог {_signed(calmest.net - base.net)} к базе."
    )
    headline = next((item for item in others if item.rule.id == "d3-grows"), None)
    if headline is not None:
        base_long = next((stats for stats in base.strategies if stats.name == "long"), None)
        head_long = next((stats for stats in headline.strategies if stats.name == "long"), None)
        if base_long is not None and head_long is not None:
            lines.append(
                f"Правило «{headline.rule.title}» даёт {_money(headline.net)} руб. ({_signed(headline.net - base.net)} к базе). "
                f"Длинное окно при нём {_signed(head_long.net - base_long.net)} руб."
            )
    return "\n".join(lines)


def _factor_text(factor: float | None) -> str:
    if factor is None:
        return "—"
    return f"{factor:.2f}"


def _signed(value: float) -> str:
    if abs(value) < 0.5:
        return "0"
    text = _money(abs(value))
    return text if value > 0 else f"-{text}"


def report(results: list[ConsensusResult]) -> str:
    """Сводка правил и разбивка по стратегиям против прогона без фильтра."""
    if not results:
        return "Нет прогонов."
    base = next((item for item in results if item.rule.id == "base"), results[0])
    base_by_name = {stats.name: stats for stats in base.strategies}
    lines = [
        "Пять счетов торгуют свои окна как раньше. Общий только запрет входа.",
        "Если чужая позиция живёт столько торговых дней и плюс по ней растёт,",
        "сигнал в обратную сторону на этой минуте не открывается.",
        "Свой стоп, канал и размер позиции не меняются. Счета не складываются в один.",
        "Запрет — это отдельная попытка входа: соседние минуты одного и того же сигнала считаются одним запретом.",
        "«Только плюс» — открытый результат больше нуля, даже если сегодня он меньше вчерашнего.",
        "«Выше вчера» — плюс больше закрытия прошлого торгового дня.",
        "«Растёт» — вчера плюс был больше позавчерашнего, а сегодня позиция всё ещё в плюсе.",
        "",
        f"{'ПРАВИЛО':<42} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'ФАКТОР':>7} {'ПРОСАДКА':>14} {'К БАЗЕ':>14}",
    ]
    for item in results:
        delta = item.net - base.net
        dd_delta = item.combined_drawdown - base.combined_drawdown
        lines.append(
            f"{item.rule.title:<42} {len(item.trades):>7} {item.blocked_setups:>7} "
            f"{_money(item.net):>14} {_signed(delta):>14} {_factor_text(_profit_factor(item.trades)):>7} "
            f"{_money(item.combined_drawdown):>14} {_signed(dd_delta):>14}"
        )
    lines.extend(["", conclusion(results), ""])
    lines.append("Просадка — худшее падение суммы пяти счетов от её пика. К базе: больше нуля — просадка меньше.")
    lines.append("Части истории, сумма пяти счетов: " + "  ".join(f"{secids[0]}–{secids[-1]}" for secids in PARTS))
    lines.append(f"{'ПРАВИЛО':<42} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    for item in results:
        parts = _part_nets(item.trades)
        lines.append(f"{item.rule.title:<42} " + " ".join(f"{_money(part):>12}" for part in parts))

    for item in results:
        if item.rule.id == "base":
            continue
        lines.extend(["", item.rule.title])
        lines.append(
            f"{'СТРАТЕГИЯ':<12} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'МИНУТ ЗАПР.':>12} {'ИТОГ':>14} {'К БАЗЕ':>14} "
            f"{'ФАКТОР':>7} {'ПРОСАДКА':>14} {'СЧЁТ':>14}"
        )
        for stats in item.strategies:
            previous = base_by_name.get(stats.name)
            previous_net = 0.0 if previous is None else previous.net
            lines.append(
                f"{stats.name:<12} {len(stats.trades):>7} {stats.blocked_setups:>7} {stats.veto_minutes:>12} "
                f"{_money(stats.net):>14} {_signed(stats.net - previous_net):>14} "
                f"{_factor_text(stats.factor):>7} {_money(stats.drawdown):>14} {_money(stats.end_equity):>14}"
            )
        heard = [f"{name} {count}" for name, count in item.source_setups.items() if count]
        lines.append("Минуты запрета — сколько минут обратный вход уже нельзя было открыть, даже без сигнала.")
        lines.append("Запреты поставили: " + (", ".join(heard) if heard else "ни одного"))
        gap = [
            f"{stats.name} {_signed(stats.end_equity - stats.cash - stats.net)}"
            for stats in item.strategies
            if abs(stats.end_equity - stats.cash - stats.net) >= 1
        ]
        if gap:
            lines.append("Счёт и сумма сделок разошлись: " + ", ".join(gap))
    return "\n".join(lines)


def rules_by_id(selected: str | None) -> tuple[ConsensusRule, ...]:
    if not selected:
        return RULES
    wanted = [item.strip() for item in selected.split(",") if item.strip()]
    known = {rule.id: rule for rule in RULES}
    missing = [item for item in wanted if item not in known]
    if missing:
        raise SystemExit("нет правил: " + ", ".join(missing))
    return tuple(known[item] for item in wanted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Запрет ложного пробоя по чужой многодневной позиции")
    parser.add_argument("--bars", type=Path, default=Path("data/bars"), help="Каталог parquet по контрактам")
    parser.add_argument("--output", type=Path, help="Куда записать текст отчёта")
    parser.add_argument("--rules", help="Идентификаторы через запятую, иначе весь набор")
    args = parser.parse_args(argv)
    frames = load_minutes(args.bars)
    starts = load_trade_starts(args.bars)
    chosen = rules_by_id(args.rules)
    views = build_views(frames)
    results = []
    for rule in chosen:
        print(f"правило {rule.id}", flush=True)
        results.append(run_consensus(frames, rule, starts, views=views))
    text = report(results)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
