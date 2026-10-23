"""Восемь счетов смотрят друг на друга и не входят против очевидного тренда.

Счета те же, что в живой книге: семь окон и моментум, каждый со своих
100 000 руб. Моментум — дневной знак хода за 20 сессий, удержание 5 дней,
плечо 3, пауза после шока и широкого дня. Окна считаются как в consensus.py.

«Очевидный тренд» — несколько чужих позиций в одну сторону. Свежий вход против
них снимается до исполнения. Своя позиция не переворачивается: если запрет
смотрит против, счёт только закрывается. Длинное окно не глушится коротким,
пока в правиле включено «только старшее».

Команда по умолчанию подбирает правило окна 1 760 и пишет trend.txt,
trend_equity.csv и trend_equity.png. Остальные роли фиксированы. Сетка окна
60 — --family sixty, короткое сравнение принятых ролей — --family kept,
полная сетка ролей — --family roles, сетка общего правила — --family grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from backtest import PARTS, WINDOWS, _money, _part_nets, _profit_factor, load_minutes, load_trade_starts
from consensus import (
    ADOPTED_RULE,
    NAMES,
    OpenState,
    StrategyStats,
    _Drawdown,
    _arrays,
    _day_index,
    _profit_grew,
    _signed,
    build_views,
    open_pnl,
    position_state,
    still_in,
    suppress_entry,
    veto_side,
    veto_sources,
)
from cnyrub.engine import _FillBook, marked_equity, step_minute
from cnyrub.live.momentum import sign_for_today, target_lots


MOMENTUM = "momentum"
EIGHT = NAMES + (MOMENTUM,)
MOMENTUM_CASH = 100_000.0
MOMENTUM_MARGIN = 1_000.0
MOMENTUM_LEVERAGE = 3.0
START_SUM = MOMENTUM_CASH + sum(window.cash for window in WINDOWS)


@dataclass(frozen=True)
class TrendSpec:
    """Как восемь счетов решают, что тренд уже очевиден."""

    id: str
    title: str
    legacy: bool = False
    enabled: bool = False
    quorum: int = 2
    hold_days: int = 1
    growing: str = "no"
    slower_only: bool = True
    profit: bool = True
    exclusive: bool = True
    shield: frozenset[str] = frozenset()
    momentum_votes: bool = True
    block_momentum: bool = True
    min_channel: int = 0
    price_days: int = 0
    voters: frozenset[str] | None = None


def _spec(spec_id: str, title: str, **overrides: object) -> TrendSpec:
    return TrendSpec(spec_id, title, **overrides)  # type: ignore[arg-type]


# Сетка вокруг одной мысли: вход против нескольких согласных позиций — ложный.
# «Старшее» не даёт короткому окну глушить длинное. Моментум голосует всегда,
# когда momentum_votes, даже если его канал короче всех.
SPECS: tuple[TrendSpec, ...] = (
    _spec("base", "без взаимодействия"),
    _spec("legacy", "старшее, 2 дня, только плюс", legacy=True),
    _spec("q2-d0", "двое старших, плюс сразу", enabled=True, quorum=2, hold_days=0),
    _spec("q2-d1", "двое старших, 1 день в плюсе", enabled=True, quorum=2, hold_days=1),
    _spec("q2-d2", "двое старших, 2 дня в плюсе", enabled=True, quorum=2, hold_days=2),
    _spec("q2-d5", "двое старших, 5 дней в плюсе", enabled=True, quorum=2, hold_days=5),
    _spec("q3-d1", "трое старших, 1 день в плюсе", enabled=True, quorum=3, hold_days=1),
    _spec("q4-d1", "четверо старших, 1 день в плюсе", enabled=True, quorum=4, hold_days=1),
    _spec("q2-soft", "двое старших, 1 день, плюс растёт", enabled=True, quorum=2, hold_days=1, growing="soft"),
    _spec("q2-hold", "двое старших, просто держат день", enabled=True, quorum=2, hold_days=1, profit=False),
    _spec("q2-px10", "двое старших и ход 10 дней", enabled=True, quorum=2, hold_days=1, price_days=10),
    _spec("q2-px20", "двое старших и ход 20 дней", enabled=True, quorum=2, hold_days=1, price_days=20),
    _spec("q2-mid", "двое от 525, 1 день в плюсе", enabled=True, quorum=2, hold_days=1, min_channel=525),
    _spec("q2-swing", "двое от 1760, 1 день в плюсе", enabled=True, quorum=2, hold_days=1, min_channel=1760),
    _spec("leg-q2-d0", "старшее и двое, плюс сразу", legacy=True, enabled=True, quorum=2, hold_days=0),
    _spec("leg-q2-d1", "старшее и двое, 1 день", legacy=True, enabled=True, quorum=2, hold_days=1),
    _spec("leg-q2-d2", "старшее и двое, 2 дня", legacy=True, enabled=True, quorum=2, hold_days=2),
    _spec("leg-q3-d1", "старшее и трое, 1 день", legacy=True, enabled=True, quorum=3, hold_days=1),
    _spec("leg-px10", "старшее, двое и ход 10 дней", legacy=True, enabled=True, quorum=2, hold_days=1, price_days=10),
    _spec("leg-px20", "старшее, двое и ход 20 дней", legacy=True, enabled=True, quorum=2, hold_days=1, price_days=20),
    _spec("leg-mid", "старшее и двое от 525", legacy=True, enabled=True, quorum=2, hold_days=1, min_channel=525),
    _spec("leg-swing", "старшее и двое от 1760", legacy=True, enabled=True, quorum=2, hold_days=1, min_channel=1760),
    _spec(
        "any-shield",
        "любые двое, длинное молчит",
        enabled=True,
        quorum=2,
        hold_days=1,
        slower_only=False,
        shield=frozenset({"long"}),
    ),
    _spec("any-open", "любые двое, без защиты", enabled=True, quorum=2, hold_days=1, slower_only=False),
    _spec(
        "leg-fast",
        "старшее, быстрым мешают любые двое",
        legacy=True,
        enabled=True,
        quorum=2,
        hold_days=1,
        slower_only=False,
        shield=frozenset({"long", "m2160", "m1760"}),
    ),
)


@dataclass(frozen=True)
class RoleRule:
    """Запрет, который действует только на одну группу счетов."""

    mode: str = "none"
    quorum: int = 2
    hold_days: int = 1
    growing: str = "no"
    slower_only: bool = True
    profit: bool = True
    exclusive: bool = True
    momentum_votes: bool = True
    min_channel: int = 0
    price_days: int = 0
    voters: frozenset[str] | None = None


MINI = ("thirty", "fortyfive", "sixty")
MEDIUM = ("m1760", "m2160")
# Мини-окна: одно старшее окно два дня в плюсе. Короткое: четыре старших один день в плюсе.
MINI_RULE = RoleRule("legacy")
SHORT_RULE = RoleRule("quorum", quorum=4, hold_days=1, momentum_votes=True)
LONG_RULE = RoleRule("none")


@dataclass(frozen=True)
class SplitSpec:
    """Разные запреты для мини-окон, короткого, средних и моментума. Длинное не глушится."""

    id: str
    title: str
    legend: str
    roles: tuple[tuple[str, RoleRule], ...]
    medium_id: str = "none"
    momentum_id: str = "none"

    def role(self, name: str) -> RoleRule:
        for key, rule in self.roles:
            if key == name:
                return rule
        raise KeyError(name)


def split_spec(
    medium: RoleRule,
    momentum: RoleRule,
    spec_id: str,
    title: str,
    legend: str,
    medium_id: str,
    momentum_id: str,
    m2160: RoleRule | None = None,
    sixty: RoleRule | None = None,
) -> SplitSpec:
    rule_2160 = medium if m2160 is None else m2160
    rule_sixty = MINI_RULE if sixty is None else sixty
    roles = tuple(
        (name, rule)
        for name, rule in (
            ("thirty", MINI_RULE),
            ("fortyfive", MINI_RULE),
            ("sixty", rule_sixty),
            ("short", SHORT_RULE),
            ("long", LONG_RULE),
            ("m1760", medium),
            ("m2160", rule_2160),
            (MOMENTUM, momentum),
        )
    )
    return SplitSpec(spec_id, title, legend, roles, medium_id, momentum_id)


# Окно 60: одно более длинное окно уже в плюсе. Два дня ждать не нужно. Моментум не голосует.
SIXTY_RULE = RoleRule("quorum", quorum=1, hold_days=0, momentum_votes=False)
# Окно 2160 не покупает против моментума и не продаёт против него, если тот уже два дня в плюсе.
M2160_MOMENTUM_RULE = RoleRule(
    "quorum",
    quorum=1,
    hold_days=2,
    slower_only=False,
    profit=True,
    voters=frozenset({MOMENTUM}),
)


def kept_split() -> SplitSpec:
    """1760 без своего запрета. 2160 уступает моментуму, который два дня в плюсе."""
    return split_spec(
        NONE_RULE,
        NONE_RULE,
        "roles-m2160-mom2",
        "мини 2 дня, короткое четверо; 2160 — моментум 2 дня в плюсе",
        "2160 — моментум 2 дня в плюсе",
        "none",
        "none",
        m2160=M2160_MOMENTUM_RULE,
        sixty=SIXTY_RULE,
    )


@dataclass
class EightResult:
    spec: TrendSpec | SplitSpec
    strategies: list[StrategyStats]
    combined_drawdown: float
    dates: list[date]
    curve: list[float]
    sixty_blocks: dict[str, int] = field(default_factory=dict)
    m1760_blocks: dict[str, int] = field(default_factory=dict)

    @property
    def trades(self) -> list[dict[str, object]]:
        return [trade for stats in self.strategies for trade in stats.trades]

    @property
    def net(self) -> float:
        return sum(stats.net for stats in self.strategies)

    @property
    def blocked_setups(self) -> int:
        return sum(stats.blocked_setups for stats in self.strategies)


class _MomentumTape:
    """Дневные свечи одного контракта. Выходные в сессию моментума не входят."""

    def __init__(self) -> None:
        self.days: list[dict[str, object]] = []
        self.bar: dict[str, object] | None = None
        self.closes: list[float] = []

    def see(self, day: date, opened: float, high: float, low: float, close: float, volume: float) -> bool:
        """True — первая минутка новой будней сессии. Предыдущий день уже закрыт."""
        if day.weekday() >= 5:
            return False
        stamp = day.isoformat()
        if self.bar is not None and self.bar.get("d") == stamp:
            self.bar["h"] = max(float(self.bar["h"]), high)
            self.bar["l"] = min(float(self.bar["l"]), low)
            self.bar["c"] = close
            self.bar["v"] = float(self.bar["v"]) + volume
            return False
        if self.bar is not None:
            self.days.append(self.bar)
            self.closes.append(float(self.bar["c"]))
        self.bar = {"d": stamp, "o": opened, "h": high, "l": low, "c": close, "v": volume}
        return True

    def price_sign(self, lookback: int) -> int:
        """Знак хода по уже закрытым будням. Сегодняшнее закрытие сюда не входит."""
        if lookback <= 0 or len(self.closes) <= lookback:
            return 0
        delta = self.closes[-1] - self.closes[-1 - lookback]
        if delta > 0:
            return 1
        if delta < 0:
            return -1
        return 0


def _peer_votes(state: OpenState, spec: TrendSpec, name: str, channel: int, price_sign: int) -> bool:
    """Эта чужая позиция входит в кворум для стратегии name."""
    if state.name == name:
        return False
    if spec.voters is not None and state.name not in spec.voters:
        return False
    if state.name == MOMENTUM:
        if not spec.momentum_votes:
            return False
    elif name == MOMENTUM:
        if not spec.block_momentum or state.channel < spec.min_channel:
            return False
    else:
        if state.channel < spec.min_channel:
            return False
        if spec.slower_only and state.channel <= channel:
            return False
    if state.held_days < spec.hold_days:
        return False
    if spec.profit and not state.pnl > 0:
        return False
    if spec.profit and not _profit_grew(state, spec.growing):
        return False
    if spec.price_days and (price_sign == 0 or state.side != price_sign):
        return False
    return True


def obvious_veto(states: list[OpenState], spec: TrendSpec, name: str, channel: int, price_sign: int) -> int:
    """1 — не покупать, -1 — не продавать, 2 — ни туда ни туда, 0 — пускать.

    Запрет есть, когда в одну сторону набрался кворум. При exclusive вторая
    сторона с любой подтверждающей позицией делает тренд неочевидным.
    """
    if not spec.enabled or spec.quorum <= 0 or name in spec.shield:
        return 0
    longs = 0
    shorts = 0
    for state in states:
        if not _peer_votes(state, spec, name, channel, price_sign):
            continue
        if state.side > 0:
            longs += 1
        elif state.side < 0:
            shorts += 1
    if spec.exclusive and longs and shorts:
        return 0
    block_long = shorts >= spec.quorum
    block_short = longs >= spec.quorum
    if block_long and block_short:
        return 2
    if block_long:
        return 1
    if block_short:
        return -1
    return 0


def obvious_sources(
    states: list[OpenState], spec: TrendSpec, name: str, channel: int, side: int, price_sign: int
) -> list[str]:
    """Кто из кворума смотрит против входа в сторону side."""
    if obvious_veto(states, spec, name, channel, price_sign) not in {side, 2}:
        return []
    found: list[str] = []
    for state in states:
        if not _peer_votes(state, spec, name, channel, price_sign):
            continue
        if side > 0 and state.side < 0:
            found.append(state.name)
        elif side < 0 and state.side > 0:
            found.append(state.name)
    return found


def combine_veto(legacy: int, obvious: int) -> int:
    block_long = legacy in {1, 2} or obvious in {1, 2}
    block_short = legacy in {-1, 2} or obvious in {-1, 2}
    if block_long and block_short:
        return 2
    if block_long:
        return 1
    if block_short:
        return -1
    return 0


def _as_quorum(role: RoleRule) -> TrendSpec:
    return TrendSpec(
        "role",
        "role",
        enabled=True,
        quorum=role.quorum,
        hold_days=role.hold_days,
        growing=role.growing,
        slower_only=role.slower_only,
        profit=role.profit,
        exclusive=role.exclusive,
        momentum_votes=role.momentum_votes,
        block_momentum=True,
        min_channel=role.min_channel,
        price_days=role.price_days,
        voters=role.voters,
    )


def entry_veto(
    name: str,
    channel: int,
    states: list[OpenState],
    window_states: list[OpenState],
    spec: TrendSpec | SplitSpec,
    price_sign: int,
) -> int:
    """Запрет свежего входа этой стратегии. Для общего правила и для раздельных ролей."""
    if isinstance(spec, TrendSpec):
        legacy = veto_side(window_states, ADOPTED_RULE, name, channel) if spec.legacy and name != MOMENTUM else 0
        return combine_veto(legacy, obvious_veto(states, spec, name, channel, price_sign))
    role = spec.role(name)
    if role.mode == "none":
        return 0
    if role.mode == "legacy":
        return veto_side(window_states, ADOPTED_RULE, name, channel)
    if role.mode != "quorum":
        raise ValueError(f"неизвестный режим роли: {role.mode}")
    return obvious_veto(states, _as_quorum(role), name, channel, price_sign)


def _entry_sources(
    name: str,
    channel: int,
    states: list[OpenState],
    window_states: list[OpenState],
    spec: TrendSpec | SplitSpec,
    price_sign: int,
    side: int,
) -> list[str]:
    """Кто смотрит против этого входа. Пусто, если запрета нет или правило общее."""
    if not isinstance(spec, SplitSpec):
        return []
    role = spec.role(name)
    if role.mode == "legacy":
        return veto_sources(window_states, ADOPTED_RULE, name, channel, side)
    if role.mode == "quorum":
        return obvious_sources(states, _as_quorum(role), name, channel, side, price_sign)
    return []


def apply_momentum_target(book: _FillBook, desired: int, veto: int) -> bool:
    """Поставить дневную цель моментума. Против запрета позиция только закрывается.

    Возвращает True, если свежий вход или переворот сняты.
    """
    if desired == 0:
        book.target = 0
        if book.held:
            book.reason = "flat"
        return False
    side = 1 if desired > 0 else -1
    blocked = veto == 2 or veto == side
    if book.held == 0:
        if blocked:
            book.target = 0
            return True
        book.target = desired
        book.reason = ""
        book.entry = "up" if side > 0 else "down"
        return False
    held_side = 1 if book.held > 0 else -1
    if side == held_side or not blocked:
        book.target = desired
        return False
    book.target = 0
    book.reason = "flat"
    return True


def _step_window(book: _FillBook, window, pack: dict[str, np.ndarray], i: int, days: np.ndarray, last_day: date, front: date | None) -> None:
    count = len(days)
    step_minute(
        book,
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
        cooldown=window.cooldown,
        min_stop=window.min_stop,
        skip_flat=window.skip_flat,
    )


def _step_momentum(book: _FillBook, i: int, pack: dict[str, np.ndarray], days: np.ndarray, last_day: date, front: date | None) -> None:
    count = len(days)
    step_minute(
        book,
        i,
        opened=float(pack["open"][i]),
        high=float(pack["high"][i]),
        low=float(pack["low"][i]),
        close=float(pack["close"][i]),
        volume=float(pack["volume"][i]),
        day=days[i],
        next_day=None if i + 1 >= count else days[i + 1],
        last_day=last_day,
        prior_high=float("nan"),
        prior_low=float("nan"),
        prior_vol=float("nan"),
        prior_range=float("nan"),
        exit_high=float("nan"),
        exit_low=float("nan"),
        clock_vol=float("nan"),
        entry_ready=0.0,
        clearance=0.0,
        cooldown=0,
        clock_volume=False,
        clock_cap=None,
        size_mode="flat",
        stop_mult=None,
        loss_bars=None,
        risk_fraction=None,
        margin=0.0,
        stop_rub=None,
        breakout_span=None,
        trade_from=front,
        trail=False,
    )


def run_eight(
    frames: dict[str, pd.DataFrame],
    spec: TrendSpec,
    trade_from: dict[str, date] | None = None,
    *,
    views: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> EightResult:
    """Один проход. Восемь книг не смешиваются, запрет общий на минуту."""
    if len(NAMES) != len(WINDOWS):
        raise ValueError("имён окон и окон должно быть поровну")
    channels = {name: window.channel for name, window in zip(NAMES, WINDOWS, strict=True)}
    channels[MOMENTUM] = 0
    cash = {name: window.cash for name, window in zip(NAMES, WINDOWS, strict=True)}
    cash[MOMENTUM] = MOMENTUM_CASH
    equity = dict(cash)
    trades: dict[str, list[dict[str, object]]] = {name: [] for name in EIGHT}
    blocked_bars = dict.fromkeys(EIGHT, 0)
    blocked_setups = dict.fromkeys(EIGHT, 0)
    veto_minutes = dict.fromkeys(EIGHT, 0)
    sixty_blocks: dict[str, int] = {}
    m1760_blocks: dict[str, int] = {}
    drawdowns = {name: _Drawdown(cash[name]) for name in EIGHT}
    combined = _Drawdown(sum(cash.values()))
    curve_dates: list[date] = []
    curve: list[float] = []
    if views is None:
        views = build_views(frames)

    for secid, frame in frames.items():
        prepared = views[secid]
        sample = prepared[NAMES[0]]
        days = sample.index.date
        day_index = _day_index(days)
        last_day = days[-1]
        packs = {name: _arrays(prepared[name]) for name in NAMES}
        price = packs[NAMES[0]]
        books = {name: _FillBook(secid, equity[name]) for name in EIGHT}
        tape = _MomentumTape()
        prev_pnl: dict[str, float | None] = dict.fromkeys(EIGHT)
        prev2_pnl: dict[str, float | None] = dict.fromkeys(EIGHT)
        streak = dict.fromkeys(EIGHT, 0)
        front = None if trade_from is None else trade_from.get(secid)
        count = len(sample)
        for i in range(count):
            day = days[i]
            close = float(price["close"][i])
            for name, window in zip(NAMES, WINDOWS, strict=True):
                _step_window(books[name], window, packs[name], i, days, last_day, front)
            new_session = tape.see(
                day,
                float(price["open"][i]),
                float(price["high"][i]),
                float(price["low"][i]),
                close,
                float(price["volume"][i]),
            )
            states: list[OpenState] = []
            for name in EIGHT:
                state = position_state(
                    name,
                    channels[name],
                    books[name],
                    i,
                    day_index,
                    close,
                    prev_pnl[name],
                    prev2_pnl[name],
                )
                if state is not None:
                    states.append(state)
            window_states = [state for state in states if state.name != MOMENTUM]
            shared_sign = tape.price_sign(spec.price_days) if isinstance(spec, TrendSpec) else 0
            for name in NAMES:
                book = books[name]
                sign = tape.price_sign(spec.role(name).price_days) if isinstance(spec, SplitSpec) else shared_sign
                veto = entry_veto(name, channels[name], states, window_states, spec, sign)
                if veto:
                    veto_minutes[name] += 1
                fresh_side = 1 if book.target > 0 else -1
                if suppress_entry(book, veto):
                    blocked_bars[name] += 1
                    if streak[name] != fresh_side:
                        blocked_setups[name] += 1
                        streak[name] = fresh_side
                        block_log = {"sixty": sixty_blocks, "m1760": m1760_blocks}.get(name)
                        if block_log is not None:
                            sources = _entry_sources(
                                name, channels[name], states, window_states, spec, sign, fresh_side
                            )
                            label = "покупка" if fresh_side > 0 else "продажа"
                            key = label + " " + ("+".join(sources) if sources else "без имени")
                            block_log[key] = block_log.get(key, 0) + 1
                else:
                    streak[name] = 0
            if new_session and day != last_day and (front is None or day >= front) and tape.closes:
                sign = sign_for_today(tape.days)
                desired = target_lots(
                    sign,
                    marked_equity(books[MOMENTUM], tape.closes[-1]),
                    tape.closes[-1],
                    MOMENTUM_MARGIN,
                    MOMENTUM_LEVERAGE,
                )
                price_sign = tape.price_sign(spec.role(MOMENTUM).price_days) if isinstance(spec, SplitSpec) else shared_sign
                veto = entry_veto(MOMENTUM, 0, states, window_states, spec, price_sign)
                if veto:
                    veto_minutes[MOMENTUM] += 1
                if apply_momentum_target(books[MOMENTUM], desired, veto):
                    blocked_bars[MOMENTUM] += 1
                    fresh_side = 1 if desired > 0 else -1
                    if streak[MOMENTUM] != fresh_side:
                        blocked_setups[MOMENTUM] += 1
                        streak[MOMENTUM] = fresh_side
                else:
                    streak[MOMENTUM] = 0
            if day.weekday() < 5 or day == last_day:
                _step_momentum(books[MOMENTUM], i, price, days, last_day, front)
            marked = 0.0
            for name in EIGHT:
                value = marked_equity(books[name], close)
                drawdowns[name].add(value)
                marked += value
            combined.add(marked)
            trading = front is None or day >= front
            if trading and (i + 1 == count or days[i + 1] != day):
                curve_dates.append(day)
                curve.append(marked)
            if i + 1 == count or days[i + 1] != day:
                for name in EIGHT:
                    book = books[name]
                    prev2_pnl[name] = prev_pnl[name]
                    prev_pnl[name] = open_pnl(book, close) if still_in(book) else None
        for name in EIGHT:
            equity[name] = marked_equity(books[name], float(price["close"][-1]))
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
            cash=cash[name],
        )
        for name in EIGHT
    ]
    return EightResult(spec, strategies, combined.worst, curve_dates, curve, sixty_blocks, m1760_blocks)


def _by_name(result: EightResult) -> dict[str, StrategyStats]:
    return {stats.name: stats for stats in result.strategies}


def _weaker_parts(result: EightResult, base: EightResult) -> int:
    left = _part_nets(base.trades)
    right = _part_nets(result.trades)
    return sum(1 for before, after in zip(left, right, strict=True) if after < before - 0.5)


def choose_spec(results: list[EightResult]) -> EightResult:
    """Правило исследования: выше базы, длинное окно не хуже, меньше слабых частей.

    Сначала отбрасываются прогоны, где длинное окно теряет к базе. Среди
    оставшихся берётся тот, у кого меньше частей слабее базы, а при равенстве —
    большая сумма. Если таких нет, остаётся наибольшая сумма вообще.
    """
    base = next(item for item in results if item.spec.id == "base")
    base_long = _by_name(base)["long"].net
    pool = [item for item in results if item.spec.id not in {"base", "legacy"}]
    if not pool:
        raise ValueError("в наборе нет правил исследования")
    safe = [item for item in pool if _by_name(item)["long"].net >= base_long - 0.5]
    ranked = safe or pool

    def rank(item: EightResult) -> tuple[int, float, float]:
        return (-_weaker_parts(item, base), item.net, item.combined_drawdown)

    return max(ranked, key=rank)


def report(results: list[EightResult]) -> str:
    """Сетка, выбор правила и три строки, которые идут на график."""
    base = next(item for item in results if item.spec.id == "base")
    legacy = next((item for item in results if item.spec.id == "legacy"), None)
    chosen = choose_spec(results)
    base_by_name = _by_name(base)
    lines = [
        "Восемь счетов, каждый стартует со 100 000 руб. Сумма старта "
        f"{_money(START_SUM)} руб. Счета не складываются в один.",
        "Семь окон — как в живой книге. Моментум: ход за 20 будней, держит 5 дней, плечо 3,",
        "следующий день без позиции после хода больше 2σ, день с объёмом или диапазоном",
        "больше двух медиан не открывает следующие 5 дней. Залог в этом прогоне 1 000 руб.,",
        "как у окон. Выходные в сессию моментума не входят. Месяц до истекающего контракта",
        "прогревает сигнал и не торгуется.",
        "Запрет снимает только свежий вход. Своя позиция против запрета закрывается и не переворачивается.",
        "«Старшее» — голос только у более длинного окна. Моментум голосует отдельно, его канал 0.",
        "«Ход N дней» — закрытия будней контракта уже прошли в ту же сторону, что и кворум.",
        "",
        f"{'ПРАВИЛО':<42} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'ФАКТОР':>7} {'ПРОСАДКА':>14}",
    ]
    for item in results:
        mark = "  ←" if item.spec.id == chosen.spec.id else ""
        lines.append(
            f"{item.spec.title:<42} {len(item.trades):>7} {item.blocked_setups:>7} "
            f"{_money(item.net):>14} {_signed(item.net - base.net):>14} "
            f"{_factor(item):>7} {_money(item.combined_drawdown):>14}{mark}"
        )
    lines.extend(["", _decision(base, legacy, chosen), ""])
    lines.append("Части истории, сумма восьми счетов: " + "  ".join(f"{secids[0]}–{secids[-1]}" for secids in PARTS))
    lines.append(f"{'ПРАВИЛО':<42} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    shown: list[EightResult] = []
    seen: set[str] = set()
    for item in (base, legacy, chosen):
        if item is None or item.spec.id in seen:
            continue
        seen.add(item.spec.id)
        shown.append(item)
    for item in shown:
        parts = _part_nets(item.trades)
        lines.append(f"{item.spec.title:<42} " + " ".join(f"{_money(part):>12}" for part in parts))
    for item in shown:
        lines.extend(["", item.spec.title])
        lines.append(
            f"{'СТРАТЕГИЯ':<12} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'ПРОСАДКА':>14} {'СЧЁТ':>14}"
        )
        for stats in item.strategies:
            previous = base_by_name[stats.name].net
            lines.append(
                f"{stats.name:<12} {len(stats.trades):>7} {stats.blocked_setups:>7} "
                f"{_money(stats.net):>14} {_signed(stats.net - previous):>14} "
                f"{_money(stats.drawdown):>14} {_money(stats.end_equity):>14}"
            )
    lines.append("")
    lines.append(
        "На графике три суммы: без взаимодействия, уже существующее правило и правило этого прогона. "
        "Сумма — восемь счетов вместе с открытым результатом."
    )
    return "\n".join(lines)


def _factor(item: EightResult) -> str:
    factor = _profit_factor(item.trades)
    if factor is None:
        return "—"
    return f"{factor:.2f}"


def _decision(base: EightResult, legacy: EightResult | None, chosen: EightResult) -> str:
    base_long = _by_name(base)["long"].net
    chosen_long = _by_name(chosen)["long"].net
    weaker = _weaker_parts(chosen, base)
    text = (
        f"Правило этого прогона — «{chosen.spec.title}»: итог {_money(chosen.net)} руб., "
        f"{_signed(chosen.net - base.net)} к базе без взаимодействия. "
        f"Длинное окно {_signed(chosen_long - base_long)}. "
        f"Частей слабее базы: {weaker} из 5. "
        f"Просадка суммы {_money(chosen.combined_drawdown)} руб. "
        f"({_signed(chosen.combined_drawdown - base.combined_drawdown)} к базе, больше нуля — просадка меньше)."
    )
    if legacy is None:
        return text
    gap = chosen.net - legacy.net
    if gap > 0.5:
        relation = f"Оно выше уже существующего правила на {_money(gap)} руб."
    elif gap < -0.5:
        relation = (
            f"Уже существующее правило остаётся выше на {_money(-gap)} руб. "
            "На графике третья линия — всё равно лучшее из этой сетки, не подмена старого правила."
        )
    else:
        relation = "С существующим правилом сумма совпала."
    return text + " " + relation


def plot_equity(results: list[EightResult], output: Path) -> Path:
    """Три линии: база, существующее правило, правило этого прогона."""
    base = next(item for item in results if item.spec.id == "base")
    legacy = next(item for item in results if item.spec.id == "legacy")
    chosen = choose_spec(results)
    if not (len(base.dates) == len(legacy.dates) == len(chosen.dates) == len(base.curve)):
        raise ValueError("у трёх прогонов разная длина кривой")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_equity, ax_gap) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7.2),
        height_ratios=(3, 1.4),
        layout="constrained",
    )
    dates = [pd.Timestamp(day) for day in base.dates]
    series = (
        (base, "#6b6258", 1.6),
        (legacy, "#1f4e79", 1.6),
        (chosen, "#9a3412", 1.8),
    )
    base_curve = np.asarray(base.curve, dtype=float)
    for item, color, width in series:
        ax_equity.plot(dates, item.curve, color=color, linewidth=width, label=item.spec.title)
        if item.spec.id != "base":
            ax_gap.plot(dates, np.asarray(item.curve) - base_curve, color=color, linewidth=width, label=item.spec.title)
    ax_gap.axhline(0.0, color="#6b6258", linewidth=0.8)
    ax_equity.set_title("Сумма восьми счетов")
    ax_equity.set_ylabel("руб.")
    ax_gap.set_ylabel("к базе, руб.")
    ax_gap.set_xlabel("день истекающего контракта")
    ax_equity.legend(frameon=False, loc="upper left")
    ax_equity.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_equity.xaxis.set_major_locator(mdates.YearLocator())
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_curve(results: list[EightResult], output: Path) -> Path:
    """Три столбца суммы счетов, одна строка на торговый день."""
    base = next(item for item in results if item.spec.id == "base")
    legacy = next(item for item in results if item.spec.id == "legacy")
    chosen = choose_spec(results)
    frame = pd.DataFrame(
        {
            "date": [day.isoformat() for day in base.dates],
            "base": base.curve,
            "legacy": legacy.curve,
            "research": chosen.curve,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


NONE_RULE = RoleRule("none")
MEDIUM_OPTIONS: tuple[tuple[str, str, RoleRule], ...] = (
    ("none", "сами", NONE_RULE),
    ("leg", "старшее, 2 дня", RoleRule("legacy")),
    ("q2-d0", "двое старших, плюс сразу", RoleRule("quorum", quorum=2, hold_days=0)),
    ("q2-d1", "двое старших, 1 день", RoleRule("quorum", quorum=2, hold_days=1, momentum_votes=False)),
    ("q2-d1-mom", "двое старших и моментум, 1 день", RoleRule("quorum", quorum=2, hold_days=1)),
    ("q2-d2", "двое старших, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, momentum_votes=False)),
    ("q2-d2-mom", "двое старших и моментум, 2 дня", RoleRule("quorum", quorum=2, hold_days=2)),
    ("long-d2", "только длинное, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d5", "только длинное, 5 дней", RoleRule("quorum", quorum=1, hold_days=5, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-mom", "длинное и моментум, 1 день", RoleRule("quorum", quorum=2, hold_days=1, voters=frozenset({"long", MOMENTUM}))),
    ("mom-d2", "только моментум, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, voters=frozenset({MOMENTUM}))),
    ("px20", "двое старших и ход 20 дней", RoleRule("quorum", quorum=2, hold_days=1, price_days=20)),
)
MOMENTUM_OPTIONS: tuple[tuple[str, str, RoleRule], ...] = (
    ("none", "сам", NONE_RULE),
    ("long-d1", "длинное, 1 день", RoleRule("quorum", quorum=1, hold_days=1, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d2", "длинное, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d5", "длинное, 5 дней", RoleRule("quorum", quorum=1, hold_days=5, momentum_votes=False, voters=frozenset({"long"}))),
    ("swing-d1", "двое от 1760, 1 день", RoleRule("quorum", quorum=2, hold_days=1, min_channel=1760, momentum_votes=False)),
    ("swing-d2", "двое от 1760, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, min_channel=1760, momentum_votes=False)),
    ("med-d1", "оба средних, 1 день", RoleRule("quorum", quorum=2, hold_days=1, slower_only=False, momentum_votes=False, voters=frozenset({"m1760", "m2160"}))),
    ("med-d2", "оба средних, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, slower_only=False, momentum_votes=False, voters=frozenset({"m1760", "m2160"}))),
    ("from525-d2", "двое от 525, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, min_channel=525, momentum_votes=False)),
    ("short-d2", "короткое, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, momentum_votes=False, voters=frozenset({"short"}))),
    ("px20-long", "длинное и ход 20 дней", RoleRule("quorum", quorum=1, hold_days=1, price_days=20, momentum_votes=False, voters=frozenset({"long"}))),
    ("any-d2", "любое окно, 2 дня", RoleRule("legacy")),
)
# Окно 60 отдельно. Остальные роли — как в kept_split: 30 и 45 на двух днях,
# короткое на четырёх старших, 2160 уступает моментуму.
SIXTY_OPTIONS: tuple[tuple[str, str, RoleRule], ...] = (
    ("none", "само, без запрета", NONE_RULE),
    ("leg", "старшее, 2 дня, только плюс", RoleRule("legacy")),
    ("d0", "старшее, плюс сразу", RoleRule("quorum", quorum=1, hold_days=0, momentum_votes=False)),
    ("d1", "старшее, 1 день", RoleRule("quorum", quorum=1, hold_days=1, momentum_votes=False)),
    ("d3", "старшее, 3 дня", RoleRule("quorum", quorum=1, hold_days=3, momentum_votes=False)),
    ("d5", "старшее, 5 дней", RoleRule("quorum", quorum=1, hold_days=5, momentum_votes=False)),
    ("soft", "старшее, 2 дня, плюс растёт", RoleRule("quorum", quorum=1, hold_days=2, growing="soft", momentum_votes=False)),
    ("swing-d1", "от 1760, 1 день", RoleRule("quorum", quorum=1, hold_days=1, min_channel=1760, momentum_votes=False)),
    ("swing-d2", "от 1760, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, min_channel=1760, momentum_votes=False)),
    ("long-d1", "только длинное, 1 день", RoleRule("quorum", quorum=1, hold_days=1, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d2", "только длинное, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d5", "только длинное, 5 дней", RoleRule("quorum", quorum=1, hold_days=5, momentum_votes=False, voters=frozenset({"long"}))),
    ("short-d2", "только короткое, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, momentum_votes=False, voters=frozenset({"short"}))),
    ("q2-d0", "двое старших, плюс сразу", RoleRule("quorum", quorum=2, hold_days=0, momentum_votes=False)),
    ("q2-d1", "двое старших, 1 день", RoleRule("quorum", quorum=2, hold_days=1, momentum_votes=False)),
    ("q2-d2", "двое старших, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, momentum_votes=False)),
    ("q2-d1-mom", "двое, моментум может голосовать, 1 день", RoleRule("quorum", quorum=2, hold_days=1)),
    ("q3-d2", "трое старших, 2 дня", RoleRule("quorum", quorum=3, hold_days=2, momentum_votes=False)),
    ("mom-d2", "только моментум, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, voters=frozenset({MOMENTUM}))),
    ("med-d2", "оба средних, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, slower_only=False, momentum_votes=False, voters=frozenset({"m1760", "m2160"}))),
    ("px20", "старшее, 2 дня и ход 20 дней", RoleRule("quorum", quorum=1, hold_days=2, price_days=20, momentum_votes=False)),
)


def sixty_spec(option_id: str, label: str, rule: RoleRule) -> SplitSpec:
    """Те же роли, что сейчас, и другое правило только у окна 60."""
    base = kept_split()
    roles = tuple((name, rule if name == "sixty" else old) for name, old in base.roles)
    return SplitSpec(f"sixty-{option_id}", f"окно 60: {label}", f"60 — {label}", roles)


# Окно 1760. Старше него только 2160 и 12420. Моментум голосует лишь там, где это сказано.
M1760_OPTIONS: tuple[tuple[str, str, RoleRule], ...] = (
    ("none", "само, без запрета", NONE_RULE),
    ("leg", "старшее, 2 дня, только плюс", RoleRule("legacy")),
    ("d0", "старшее, плюс сразу", RoleRule("quorum", quorum=1, hold_days=0, momentum_votes=False)),
    ("d1", "старшее, 1 день", RoleRule("quorum", quorum=1, hold_days=1, momentum_votes=False)),
    ("d5", "старшее, 5 дней", RoleRule("quorum", quorum=1, hold_days=5, momentum_votes=False)),
    ("soft", "старшее, 2 дня, плюс растёт", RoleRule("quorum", quorum=1, hold_days=2, growing="soft", momentum_votes=False)),
    ("long-d1", "только длинное, 1 день", RoleRule("quorum", quorum=1, hold_days=1, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d2", "только длинное, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, momentum_votes=False, voters=frozenset({"long"}))),
    ("long-d5", "только длинное, 5 дней", RoleRule("quorum", quorum=1, hold_days=5, momentum_votes=False, voters=frozenset({"long"}))),
    ("wide-d1", "только 2160, 1 день", RoleRule("quorum", quorum=1, hold_days=1, slower_only=False, momentum_votes=False, voters=frozenset({"m2160"}))),
    ("wide-d2", "только 2160, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, momentum_votes=False, voters=frozenset({"m2160"}))),
    ("mom-d1", "только моментум, 1 день", RoleRule("quorum", quorum=1, hold_days=1, slower_only=False, voters=frozenset({MOMENTUM}))),
    ("mom-d2", "только моментум, 2 дня", RoleRule("quorum", quorum=1, hold_days=2, slower_only=False, voters=frozenset({MOMENTUM}))),
    ("q2-d0", "2160 и длинное, плюс сразу", RoleRule("quorum", quorum=2, hold_days=0, momentum_votes=False)),
    ("q2-d1", "2160 и длинное, 1 день", RoleRule("quorum", quorum=2, hold_days=1, momentum_votes=False)),
    ("q2-d2", "2160 и длинное, 2 дня", RoleRule("quorum", quorum=2, hold_days=2, momentum_votes=False)),
    ("q2-d1-mom", "двое, моментум может голосовать, 1 день", RoleRule("quorum", quorum=2, hold_days=1)),
    ("long-mom", "длинное и моментум, 1 день", RoleRule("quorum", quorum=2, hold_days=1, voters=frozenset({"long", MOMENTUM}))),
    ("px20", "старшее, 2 дня и ход 20 дней", RoleRule("quorum", quorum=1, hold_days=2, price_days=20, momentum_votes=False)),
)


def m1760_spec(option_id: str, label: str, rule: RoleRule) -> SplitSpec:
    """Те же роли, что сейчас, и другое правило только у окна 1760."""
    base = kept_split()
    roles = tuple((name, rule if name == "m1760" else old) for name, old in base.roles)
    return SplitSpec(f"m1760-{option_id}", f"окно 1760: {label}", f"1760 — {label}", roles)


def _named_option(table: tuple[tuple[str, str, RoleRule], ...], key: str) -> tuple[str, str, RoleRule]:
    for item in table:
        if item[0] == key:
            return item
    raise KeyError(key)


def make_split(medium_id: str, momentum_id: str) -> SplitSpec:
    """Мини-окна и короткое фиксированы. Меняются только средние и моментум."""
    _, medium_label, medium = _named_option(MEDIUM_OPTIONS, medium_id)
    _, momentum_label, momentum = _named_option(MOMENTUM_OPTIONS, momentum_id)
    if medium_id == "none" and momentum_id == "none":
        spec_id = "roles-fixed"
        legend = "мини — 2 дня, короткое — четверо"
    else:
        spec_id = f"roles-{medium_id}-{momentum_id}"
        legend = f"средние: {medium_label}; моментум: {momentum_label}"
    title = f"мини 2 дня, короткое четверо; средние: {medium_label}; моментум: {momentum_label}"
    return split_spec(medium, momentum, spec_id, title, legend, medium_id, momentum_id)


def _role_safe(item: EightResult, base: EightResult, anchor: EightResult) -> bool:
    """Длинное, оба средних и моментум не хуже опоры, где у них ещё нет своего запрета."""
    names = _by_name(item)
    anchor_names = _by_name(anchor)
    if names["long"].net < _by_name(base)["long"].net - 0.5:
        return False
    for name in (*MEDIUM, MOMENTUM):
        if names[name].net < anchor_names[name].net - 0.5:
            return False
    return True


def _role_rank(item: EightResult, base: EightResult) -> tuple[int, float, float]:
    return (-_weaker_parts(item, base), item.net, item.combined_drawdown)


def choose_roles(results: list[EightResult]) -> EightResult:
    """Раздельное правило: сумма выше, а средние и моментум не платят за чужую прибавку."""
    base = next(item for item in results if item.spec.id == "base")
    anchor = next(item for item in results if item.spec.id == "roles-fixed")
    pool = [item for item in results if isinstance(item.spec, SplitSpec) and item.spec.id != "roles-fixed"]
    eligible = [item for item in pool if _role_safe(item, base, anchor)]
    if not eligible:
        return anchor
    best = max(eligible, key=lambda item: _role_rank(item, base))
    if best.net <= anchor.net + 0.5:
        return anchor
    return best


def _top_ids(runs: list[EightResult], base: EightResult, anchor: EightResult, field: str) -> list[str]:
    safe = [item for item in runs if _role_safe(item, base, anchor)]
    ordered = sorted(safe, key=lambda item: _role_rank(item, base), reverse=True)
    found: list[str] = []
    for item in ordered:
        if not isinstance(item.spec, SplitSpec):
            continue
        ident = item.spec.medium_id if field == "medium_id" else item.spec.momentum_id
        if ident not in found:
            found.append(ident)
        if len(found) == 3:
            break
    return found or ["none"]


def _legend(spec: TrendSpec | SplitSpec) -> str:
    if isinstance(spec, SplitSpec):
        return spec.legend
    return spec.title


def report_roles(results: list[EightResult]) -> str:
    """Сетка раздельных ролей и три линии графика."""
    base = next(item for item in results if item.spec.id == "base")
    legacy = next(item for item in results if item.spec.id == "legacy")
    uniform = next(item for item in results if item.spec.id == "q4-d1")
    anchor = next(item for item in results if item.spec.id == "roles-fixed")
    chosen = choose_roles(results)
    base_names = _by_name(base)
    width = max(len(item.spec.title) for item in results)
    lines = [
        "Восемь счетов, каждый со 100 000 руб. Сумма старта "
        f"{_money(START_SUM)} руб. Счета не складываются в один.",
        "Мини-окна 30, 45 и 60: «старшее, 2 дня, только плюс». Одно более длинное окно,",
        "которое уже два торговых дня в плюсе, не пускает их во вход против себя.",
        "Короткое окно 525: «четверо старших, 1 день в плюсе». Это 1 760, 2 160, 12 420 и моментум.",
        "Средние окна 1 760 и 2 160 подбираются отдельно. Моментум подбирается отдельно.",
        "Длинное окно никто не глушит. Своя позиция против запрета закрывается и не переворачивается.",
        "Правило годится, если длинное, оба средних и моментум не хуже прогона, где у средних",
        "и моментума своего запрета ещё нет, и сумма восьми счетов от этого растёт.",
        "",
        f"{'ПРАВИЛО':<{width}} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'ФАКТОР':>7} {'ПРОСАДКА':>14}",
    ]
    for item in results:
        mark = "  ←" if item.spec.id == chosen.spec.id else ""
        lines.append(
            f"{item.spec.title:<{width}} {len(item.trades):>7} {item.blocked_setups:>7} "
            f"{_money(item.net):>14} {_signed(item.net - base.net):>14} "
            f"{_factor(item):>7} {_money(item.combined_drawdown):>14}{mark}"
        )
    lines.extend(["", _roles_decision(base, legacy, uniform, anchor, chosen, results), ""])
    lines.append("Части истории, сумма восьми счетов: " + "  ".join(f"{secids[0]}–{secids[-1]}" for secids in PARTS))
    lines.append(f"{'ПРАВИЛО':<{width}} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    shown: list[EightResult] = []
    seen: set[str] = set()
    for item in (base, legacy, uniform, anchor, chosen):
        if item.spec.id in seen:
            continue
        seen.add(item.spec.id)
        shown.append(item)
    for item in shown:
        parts = _part_nets(item.trades)
        lines.append(f"{item.spec.title:<{width}} " + " ".join(f"{_money(part):>12}" for part in parts))
    for item in shown:
        lines.extend(["", item.spec.title])
        lines.append(
            f"{'СТРАТЕГИЯ':<12} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'ПРОСАДКА':>14} {'СЧЁТ':>14}"
        )
        for stats in item.strategies:
            lines.append(
                f"{stats.name:<12} {len(stats.trades):>7} {stats.blocked_setups:>7} "
                f"{_money(stats.net):>14} {_signed(stats.net - base_names[stats.name].net):>14} "
                f"{_money(stats.drawdown):>14} {_money(stats.end_equity):>14}"
            )
    lines.append("")
    lines.append("На графике три суммы: без взаимодействия, общее правило четырёх старших и раздельные роли.")
    return "\n".join(lines)


def _above_but_unsafe(results: list[EightResult], anchor: EightResult, base: EightResult, side: str) -> EightResult | None:
    """Лучшая сумма на этой стороне, которая всё же делает хуже защищённый счёт."""
    pool: list[EightResult] = []
    for item in results:
        spec = item.spec
        if not isinstance(spec, SplitSpec) or spec.id == anchor.spec.id:
            continue
        medium_side = spec.momentum_id == "none" and spec.medium_id != "none"
        momentum_side = spec.medium_id == "none" and spec.momentum_id != "none"
        if side == "medium" and not medium_side:
            continue
        if side == "momentum" and not momentum_side:
            continue
        if item.net <= anchor.net + 0.5 or _role_safe(item, base, anchor):
            continue
        pool.append(item)
    if not pool:
        return None
    return max(pool, key=lambda item: item.net)


def _hurt_text(item: EightResult, anchor: EightResult) -> str:
    names = _by_name(item)
    anchor_names = _by_name(anchor)
    hurt = [
        name
        for name in ("long", *MEDIUM, MOMENTUM)
        if names[name].net < anchor_names[name].net - 0.5
    ]
    return ", ".join(f"{name} {_signed(names[name].net - anchor_names[name].net)}" for name in hurt)


def _roles_decision(
    base: EightResult,
    legacy: EightResult,
    uniform: EightResult,
    anchor: EightResult,
    chosen: EightResult,
    results: list[EightResult],
) -> str:
    names = _by_name(chosen)
    anchor_names = _by_name(anchor)
    bits = [
        f"Раздельные роли — «{chosen.spec.title}»: итог {_money(chosen.net)} руб., "
        f"{_signed(chosen.net - base.net)} к базе без взаимодействия, "
        f"{_signed(chosen.net - uniform.net)} к общему правилу четырёх старших, "
        f"{_signed(chosen.net - legacy.net)} к общему «старшее, 2 дня».",
        f"Против опоры, где средние и моментум ещё сами по себе: {_signed(chosen.net - anchor.net)}.",
        "Счета к этой опоре: "
        + ", ".join(
            f"{name} {_signed(names[name].net - anchor_names[name].net)}"
            for name in ("m1760", "m2160", MOMENTUM, "short", "thirty", "fortyfive", "sixty", "long")
        )
        + ".",
        f"Частей слабее базы: {_weaker_parts(chosen, base)} из 5. "
        f"Просадка суммы {_money(chosen.combined_drawdown)} руб.",
    ]
    if chosen.spec.id == anchor.spec.id:
        bits.append("Отдельный запрет, который не ухудшает длинное, оба средних и моментум, сумму не поднял.")
        for side, label in (("medium", "Средние"), ("momentum", "Моментум")):
            rejected = _above_but_unsafe(results, anchor, base, side)
            if rejected is None:
                continue
            bits.append(
                f"{label}: выше опоры, но не взято, «{rejected.spec.title}», "
                f"{_signed(rejected.net - anchor.net)}. Хуже {_hurt_text(rejected, anchor)}."
            )
    return " ".join(bits)


def plot_roles(results: list[EightResult], output: Path) -> Path:
    """Три линии: база, общее правило четырёх старших, раздельные роли."""
    base = next(item for item in results if item.spec.id == "base")
    uniform = next(item for item in results if item.spec.id == "q4-d1")
    chosen = choose_roles(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_equity, ax_gap) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7.2),
        height_ratios=(3, 1.4),
        layout="constrained",
    )
    dates = [pd.Timestamp(day) for day in base.dates]
    series = (
        (base, "#6b6258", 1.6),
        (uniform, "#1f4e79", 1.6),
        (chosen, "#9a3412", 1.8),
    )
    base_curve = np.asarray(base.curve, dtype=float)
    seen: set[str] = set()
    for item, color, width in series:
        if item.spec.id in seen:
            continue
        seen.add(item.spec.id)
        ax_equity.plot(dates, item.curve, color=color, linewidth=width, label=_legend(item.spec))
        if item.spec.id != "base":
            ax_gap.plot(dates, np.asarray(item.curve) - base_curve, color=color, linewidth=width, label=_legend(item.spec))
    ax_gap.axhline(0.0, color="#6b6258", linewidth=0.8)
    ax_equity.set_title("Сумма восьми счетов")
    ax_equity.set_ylabel("руб.")
    ax_gap.set_ylabel("к базе, руб.")
    ax_gap.set_xlabel("день истекающего контракта")
    ax_equity.legend(frameon=False, loc="upper left")
    ax_equity.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_equity.xaxis.set_major_locator(mdates.YearLocator())
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_roles_curve(results: list[EightResult], output: Path) -> Path:
    base = next(item for item in results if item.spec.id == "base")
    uniform = next(item for item in results if item.spec.id == "q4-d1")
    chosen = choose_roles(results)
    frame = pd.DataFrame(
        {
            "date": [day.isoformat() for day in base.dates],
            "base": base.curve,
            "uniform_q4": uniform.curve,
            "roles": chosen.curve,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def run_role_study(
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    views: dict[str, dict[str, pd.DataFrame]],
) -> list[EightResult]:
    """Сначала каждый запрет отдельно, потом крест трёх лучших средних и трёх лучших моментума."""
    results: list[EightResult] = []

    def run(spec: TrendSpec | SplitSpec) -> EightResult:
        print(f"правило {spec.id}", flush=True)
        item = run_eight(frames, spec, starts, views=views)
        results.append(item)
        return item

    for spec_id in ("base", "legacy", "q4-d1"):
        run(next(spec for spec in SPECS if spec.id == spec_id))
    anchor = run(make_split("none", "none"))
    base = results[0]
    medium_runs = [anchor]
    for medium_id, _, _ in MEDIUM_OPTIONS:
        if medium_id == "none":
            continue
        medium_runs.append(run(make_split(medium_id, "none")))
    momentum_runs = [anchor]
    for momentum_id, _, _ in MOMENTUM_OPTIONS:
        if momentum_id == "none":
            continue
        momentum_runs.append(run(make_split("none", momentum_id)))
    top_medium = _top_ids(medium_runs, base, anchor, "medium_id")
    top_momentum = _top_ids(momentum_runs, base, anchor, "momentum_id")
    print(f"крест средние {top_medium} моментум {top_momentum}", flush=True)
    for medium_id in top_medium:
        for momentum_id in top_momentum:
            if medium_id == "none" or momentum_id == "none":
                continue
            run(make_split(medium_id, momentum_id))
    return results


def _pick(results: list[EightResult], spec_id: str) -> EightResult:
    return next(item for item in results if item.spec.id == spec_id)


def report_kept(results: list[EightResult]) -> str:
    """База, прежние роли и то же самое с запретом моментума для окна 2160."""
    base = _pick(results, "base")
    anchor = _pick(results, "roles-fixed")
    kept = _pick(results, "roles-m2160-mom2")
    base_names = _by_name(base)
    anchor_names = _by_name(anchor)
    kept_names = _by_name(kept)
    shown = [base, anchor, kept]
    for spec_id in ("legacy", "q4-d1"):
        if any(item.spec.id == spec_id for item in results):
            shown.insert(-1, _pick(results, spec_id))
    width = max(len(item.spec.title) for item in shown)
    lines = [
        "Восемь счетов, каждый со 100 000 руб. Сумма старта "
        f"{_money(START_SUM)} руб. Счета не складываются в один.",
        "Мини-окна 30, 45 и 60: «старшее, 2 дня, только плюс».",
        "Короткое окно 525: «четверо старших, 1 день в плюсе» — 1 760, 2 160, 12 420 и моментум.",
        "Окно 2 160 не входит против моментума, который уже два торговых дня в плюсе.",
        "Окно 1 760, длинное окно и сам моментум этим запретом не глушатся.",
        "Своя позиция против запрета закрывается и не переворачивается.",
        "",
        f"{'ПРАВИЛО':<{width}} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'К РОЛЯМ':>14} {'ФАКТОР':>7} {'ПРОСАДКА':>14}",
    ]
    for item in shown:
        mark = "  ←" if item.spec.id == kept.spec.id else ""
        lines.append(
            f"{item.spec.title:<{width}} {len(item.trades):>7} {item.blocked_setups:>7} "
            f"{_money(item.net):>14} {_signed(item.net - base.net):>14} "
            f"{_signed(item.net - anchor.net):>14} {_factor(item):>7} "
            f"{_money(item.combined_drawdown):>14}{mark}"
        )
    lines.extend(
        [
            "",
            f"Окно 2 160 с запретом моментума: итог {_money(kept.net)} руб., "
            f"{_signed(kept.net - base.net)} к базе, {_signed(kept.net - anchor.net)} к ролям без этого запрета.",
            "Счета к ролям без запрета на 2 160: "
            + ", ".join(
                f"{name} {_signed(kept_names[name].net - anchor_names[name].net)}"
                for name in ("m2160", "m1760", MOMENTUM, "short", "thirty", "fortyfive", "sixty", "long")
            )
            + ".",
            f"Своя просадка окна 2 160: {_money(anchor_names['m2160'].drawdown)} руб. без запрета, "
            f"{_money(kept_names['m2160'].drawdown)} руб. с запретом. "
            f"Просадка суммы {_money(kept.combined_drawdown)} руб. "
            f"Частей слабее базы: {_weaker_parts(kept, base)} из 5.",
            "",
            "Части истории, сумма восьми счетов: " + "  ".join(f"{secids[0]}–{secids[-1]}" for secids in PARTS),
            f"{'ПРАВИЛО':<{width}} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)),
        ]
    )
    for item in (base, anchor, kept):
        parts = _part_nets(item.trades)
        lines.append(f"{item.spec.title:<{width}} " + " ".join(f"{_money(part):>12}" for part in parts))
    for item in (base, anchor, kept):
        lines.extend(["", item.spec.title])
        lines.append(
            f"{'СТРАТЕГИЯ':<12} {'СДЕЛОК':>7} {'ЗАПРЕТ':>7} {'ИТОГ':>14} {'К БАЗЕ':>14} {'К РОЛЯМ':>14} {'ПРОСАДКА':>14} {'СЧЁТ':>14}"
        )
        names = _by_name(item)
        for name in ("short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160", MOMENTUM):
            stats = names[name]
            lines.append(
                f"{stats.name:<12} {len(stats.trades):>7} {stats.blocked_setups:>7} "
                f"{_money(stats.net):>14} {_signed(stats.net - base_names[name].net):>14} "
                f"{_signed(stats.net - anchor_names[name].net):>14} "
                f"{_money(stats.drawdown):>14} {_money(stats.end_equity):>14}"
            )
    lines.append("")
    lines.append("На графике три суммы: без взаимодействия, роли без запрета на 2 160 и роли с этим запретом.")
    lines.append("Нижняя панель — разница запрета на 2 160 с ролями, где его ещё нет.")
    return "\n".join(lines)


def plot_kept(results: list[EightResult], output: Path) -> Path:
    """Три линии. Нижняя панель — что добавляет запрет моментума на окне 2160."""
    base = _pick(results, "base")
    anchor = _pick(results, "roles-fixed")
    kept = _pick(results, "roles-m2160-mom2")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_equity, ax_gap) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7.2),
        height_ratios=(3, 1.4),
        layout="constrained",
    )
    dates = [pd.Timestamp(day) for day in base.dates]
    series = (
        (base, "#6b6258", 1.6),
        (anchor, "#1f4e79", 1.6),
        (kept, "#9a3412", 1.8),
    )
    anchor_curve = np.asarray(anchor.curve, dtype=float)
    for item, color, width in series:
        ax_equity.plot(dates, item.curve, color=color, linewidth=width, label=_legend(item.spec))
    ax_gap.plot(
        dates,
        np.asarray(kept.curve) - anchor_curve,
        color="#9a3412",
        linewidth=1.8,
        label=_legend(kept.spec),
    )
    ax_gap.axhline(0.0, color="#6b6258", linewidth=0.8)
    ax_equity.set_title("Сумма восьми счетов")
    ax_equity.set_ylabel("руб.")
    ax_gap.set_ylabel("к ролям без запрета 2160, руб.")
    ax_gap.set_xlabel("день истекающего контракта")
    ax_equity.legend(frameon=False, loc="upper left")
    ax_equity.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.2f} млн"))
    ax_equity.xaxis.set_major_locator(mdates.YearLocator())
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_kept_curve(results: list[EightResult], output: Path) -> Path:
    base = _pick(results, "base")
    anchor = _pick(results, "roles-fixed")
    kept = _pick(results, "roles-m2160-mom2")
    frame = pd.DataFrame(
        {
            "date": [day.isoformat() for day in base.dates],
            "base": base.curve,
            "roles": anchor.curve,
            "m2160_mom": kept.curve,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def run_kept_study(
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    views: dict[str, dict[str, pd.DataFrame]],
) -> list[EightResult]:
    """База, старое общее правило, роли без запрета на 2160 и роли с ним."""
    results: list[EightResult] = []
    specs: list[TrendSpec | SplitSpec] = [
        next(spec for spec in SPECS if spec.id == "base"),
        next(spec for spec in SPECS if spec.id == "legacy"),
        next(spec for spec in SPECS if spec.id == "q4-d1"),
        make_split("none", "none"),
        kept_split(),
    ]
    for spec in specs:
        print(f"правило {spec.id}", flush=True)
        results.append(run_eight(frames, spec, starts, views=views))
    return results


def _sixty_safe(item: EightResult, anchor: EightResult) -> bool:
    """30 и 45, короткое, длинное, средние и моментум не хуже текущего правила окна 60."""
    names = _by_name(item)
    anchor_names = _by_name(anchor)
    guarded = ("thirty", "fortyfive", "short", "long", "m1760", "m2160", MOMENTUM)
    return all(names[name].net >= anchor_names[name].net - 0.5 for name in guarded)


def choose_sixty(results: list[EightResult]) -> EightResult:
    """Правило окна 60, которое поднимает его счёт и не забирает деньги у остальных."""
    anchor = _pick(results, "sixty-leg")
    pool = [item for item in results if item.spec.id.startswith("sixty-") and item.spec.id != "sixty-leg"]
    eligible = [item for item in pool if _sixty_safe(item, anchor)]
    if not eligible:
        return anchor
    best = max(eligible, key=lambda item: (_by_name(item)["sixty"].net, item.net, item.combined_drawdown))
    if _by_name(best)["sixty"].net <= _by_name(anchor)["sixty"].net + 0.5:
        return anchor
    return best


def _best_sixty(results: list[EightResult]) -> EightResult:
    """Самый высокий итог именно окна 60, даже если 30 или 45 от этого слабее."""
    pool = [item for item in results if item.spec.id.startswith("sixty-")]
    return max(pool, key=lambda item: (_by_name(item)["sixty"].net, item.net))


def _block_lines(blocks: dict[str, int]) -> list[str]:
    if not blocks:
        return ["Запрещённых входов нет."]
    total = sum(blocks.values())
    lines = [f"Запрещённых входов: {total}."]
    who: dict[str, int] = {}
    for key, count in blocks.items():
        body = key.split(" ", 1)[1] if " " in key else key
        for name in body.split("+"):
            who[name] = who.get(name, 0) + count
    lines.append(
        "Участие счетов, один вход может дать несколько имён: "
        + ", ".join(f"{name} {count}" for name, count in sorted(who.items(), key=lambda item: (-item[1], item[0])))
    )
    for key, count in sorted(blocks.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count:>4}  {key}")
    return lines


def report_sixty(results: list[EightResult]) -> str:
    """Сетка правил только для окна 60 на фоне уже выбранных ролей остальных."""
    base = _pick(results, "base")
    anchor = _pick(results, "sixty-leg")
    chosen = choose_sixty(results)
    raw = _best_sixty(results)
    base_names = _by_name(base)
    anchor_names = _by_name(anchor)
    rows = [item for item in results if item.spec.id == "base" or item.spec.id.startswith("sixty-")]
    rows.sort(key=lambda item: _by_name(item)["sixty"].net, reverse=True)
    width = max(len(item.spec.title) for item in rows)
    lines = [
        "Окно 60 при уже выбранных ролях остальных счетов.",
        "Окна 30 и 45: «старшее, 2 дня, только плюс». Короткое 525: четверо старших, 1 день в плюсе.",
        "Окно 2 160 не входит против моментума, который два дня в плюсе. Окно 1 760 и моментум сами.",
        "Длинное окно никто не глушит. Меняется только запрет окна 60.",
        "Правило годится, если счёт 60 выше, а 30, 45, короткое, длинное, оба средних и моментум не хуже.",
        "Окно 60 длиннее 30 и 45, поэтому его позиция сама глушит их по правилу двух дней.",
        "",
        f"{'ПРАВИЛО':<{width}} {'СДЕЛКИ 60':>9} {'ЗАПРЕТ':>7} {'ИТОГ 60':>12} {'К СЕЙЧАС':>12} "
        f"{'30 К СЕЙЧАС':>12} {'45 К СЕЙЧАС':>12} {'СУММА':>14} {'СУММА К СЕЙЧАС':>16} {'ПРОСАДКА 60':>12}",
    ]
    for item in rows:
        names = _by_name(item)
        sixty = names["sixty"]
        mark = ""
        if item.spec.id == chosen.spec.id:
            mark += "  ←"
        if item.spec.id == anchor.spec.id:
            mark += "  сейчас"
        if item.spec.id == raw.spec.id and raw.spec.id != chosen.spec.id:
            mark += "  максимум 60"
        lines.append(
            f"{item.spec.title:<{width}} {len(sixty.trades):>9} {sixty.blocked_setups:>7} "
            f"{_money(sixty.net):>12} {_signed(sixty.net - anchor_names['sixty'].net):>12} "
            f"{_signed(names['thirty'].net - anchor_names['thirty'].net):>12} "
            f"{_signed(names['fortyfive'].net - anchor_names['fortyfive'].net):>12} "
            f"{_money(item.net):>14} {_signed(item.net - anchor.net):>16} "
            f"{_money(sixty.drawdown):>12}{mark}"
        )
    chosen_names = _by_name(chosen)
    raw_names = _by_name(raw)
    lines.extend(
        [
            "",
            f"Сейчас у окна 60 «старшее, 2 дня, только плюс»: итог {_money(anchor_names['sixty'].net)} руб., "
            f"{_signed(anchor_names['sixty'].net - base_names['sixty'].net)} к базе без взаимодействия, "
            f"запретов {anchor_names['sixty'].blocked_setups}.",
            f"Взято «{chosen.spec.title}»: итог окна 60 {_money(chosen_names['sixty'].net)} руб., "
            f"{_signed(chosen_names['sixty'].net - anchor_names['sixty'].net)} к текущему правилу, "
            f"сумма восьми {_money(chosen.net)} руб., {_signed(chosen.net - anchor.net)} к текущим ролям, "
            f"{_signed(chosen.net - base.net)} к базе.",
            "Счета к текущему правилу окна 60: "
            + ", ".join(
                f"{name} {_signed(chosen_names[name].net - anchor_names[name].net)}"
                for name in ("sixty", "thirty", "fortyfive", "short", "m1760", "m2160", MOMENTUM, "long")
            )
            + ".",
        ]
    )
    if raw.spec.id != chosen.spec.id:
        lines.append(
            f"Самый высокий итог окна 60 даёт «{raw.spec.title}»: {_money(raw_names['sixty'].net)} руб., "
            f"{_signed(raw_names['sixty'].net - anchor_names['sixty'].net)} к текущему. "
            f"Окно 30 {_signed(raw_names['thirty'].net - anchor_names['thirty'].net)}, "
            f"окно 45 {_signed(raw_names['fortyfive'].net - anchor_names['fortyfive'].net)}, "
            f"сумма {_signed(raw.net - anchor.net)}. Оно не взято: кто-то из остальных счетов слабее."
        )
    elif chosen.spec.id == anchor.spec.id:
        lines.append("Отдельное правило, которое поднимает окно 60 и не забирает деньги у остальных, не нашлось.")
    lines.extend(["", "Кто глушит окно 60 при текущем правиле «старшее, 2 дня, только плюс»:"])
    lines.extend(_block_lines(anchor.sixty_blocks))
    if chosen.spec.id != anchor.spec.id:
        lines.extend(["", f"Кто глушит окно 60 при «{chosen.spec.title}»:"])
        lines.extend(_block_lines(chosen.sixty_blocks))
    if raw.spec.id not in {anchor.spec.id, chosen.spec.id}:
        lines.extend(["", f"Кто глушит окно 60 при «{raw.spec.title}»:"])
        lines.extend(_block_lines(raw.sixty_blocks))
    lines.extend(["", "Части истории, только счёт окна 60:"])
    lines.append(f"{'ПРАВИЛО':<{width}} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    shown = [base, anchor]
    if chosen.spec.id != anchor.spec.id:
        shown.append(chosen)
    if raw.spec.id not in {item.spec.id for item in shown}:
        shown.append(raw)
    for item in shown:
        parts = _part_nets(_by_name(item)["sixty"].trades)
        lines.append(f"{item.spec.title:<{width}} " + " ".join(f"{_money(part):>12}" for part in parts))
    lines.extend(["", "Части истории, сумма восьми счетов:"])
    lines.append(f"{'ПРАВИЛО':<{width}} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    for item in shown:
        parts = _part_nets(item.trades)
        lines.append(f"{item.spec.title:<{width}} " + " ".join(f"{_money(part):>12}" for part in parts))
    shown = _sixty_chart_item(results)
    lines.append("")
    lines.append(
        "На графике сумма восьми счетов: без взаимодействия, текущее правило окна 60 "
        f"и «{shown.spec.legend}»."
    )
    lines.append("Нижняя панель — разница этой линии с текущим правилом окна 60.")
    return "\n".join(lines)


def _sixty_chart_item(results: list[EightResult]) -> EightResult:
    """Линия графика: взятое правило, а если оно совпало с текущим — максимум окна 60."""
    anchor = _pick(results, "sixty-leg")
    chosen = choose_sixty(results)
    if chosen.spec.id != anchor.spec.id:
        return chosen
    return _best_sixty(results)


def plot_sixty(results: list[EightResult], output: Path) -> Path:
    """Три суммы. Нижняя панель — что меняет правило окна 60."""
    base = _pick(results, "base")
    anchor = _pick(results, "sixty-leg")
    shown = _sixty_chart_item(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_equity, ax_gap) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7.2),
        height_ratios=(3, 1.4),
        layout="constrained",
    )
    dates = [pd.Timestamp(day) for day in base.dates]
    series = (
        (base, "#6b6258", 1.6),
        (anchor, "#1f4e79", 1.6),
        (shown, "#9a3412", 1.8),
    )
    anchor_curve = np.asarray(anchor.curve, dtype=float)
    seen: set[str] = set()
    for item, color, width in series:
        if item.spec.id in seen:
            continue
        seen.add(item.spec.id)
        ax_equity.plot(dates, item.curve, color=color, linewidth=width, label=_legend(item.spec))
        if item.spec.id == shown.spec.id and item.spec.id != anchor.spec.id:
            ax_gap.plot(
                dates,
                np.asarray(item.curve) - anchor_curve,
                color=color,
                linewidth=width,
                label=_legend(item.spec),
            )
    ax_gap.axhline(0.0, color="#6b6258", linewidth=0.8)
    ax_equity.set_title("Сумма восьми счетов")
    ax_equity.set_ylabel("руб.")
    ax_gap.set_ylabel("к текущему правилу 60, руб.")
    ax_gap.set_xlabel("день истекающего контракта")
    ax_equity.legend(frameon=False, loc="upper left")
    ax_equity.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.2f} млн"))
    ax_equity.xaxis.set_major_locator(mdates.YearLocator())
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_sixty_curve(results: list[EightResult], output: Path) -> Path:
    base = _pick(results, "base")
    anchor = _pick(results, "sixty-leg")
    shown = _sixty_chart_item(results)
    frame = pd.DataFrame(
        {
            "date": [day.isoformat() for day in base.dates],
            "base": base.curve,
            "sixty_now": anchor.curve,
            "sixty_shown": shown.curve,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def run_sixty_study(
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    views: dict[str, dict[str, pd.DataFrame]],
) -> list[EightResult]:
    """База без взаимодействия и каждое правило окна 60 на фиксированных остальных ролях."""
    results: list[EightResult] = []
    specs: list[TrendSpec | SplitSpec] = [next(spec for spec in SPECS if spec.id == "base")]
    specs.extend(sixty_spec(option_id, label, rule) for option_id, label, rule in SIXTY_OPTIONS)
    for spec in specs:
        print(f"правило {spec.id}", flush=True)
        results.append(run_eight(frames, spec, starts, views=views))
    return results


def _m1760_safe(item: EightResult, anchor: EightResult) -> bool:
    """Остальные семь счетов не хуже прогона, где у 1760 ещё нет своего запрета."""
    names = _by_name(item)
    anchor_names = _by_name(anchor)
    guarded = ("thirty", "fortyfive", "sixty", "short", "long", "m2160", MOMENTUM)
    return all(names[name].net >= anchor_names[name].net - 0.5 for name in guarded)


def choose_m1760(results: list[EightResult]) -> EightResult:
    """Правило окна 1760, которое поднимает его счёт и не забирает деньги у остальных."""
    anchor = _pick(results, "m1760-none")
    pool = [item for item in results if item.spec.id.startswith("m1760-") and item.spec.id != "m1760-none"]
    eligible = [item for item in pool if _m1760_safe(item, anchor)]
    if not eligible:
        return anchor
    best = max(eligible, key=lambda item: (_by_name(item)["m1760"].net, item.net, item.combined_drawdown))
    if _by_name(best)["m1760"].net <= _by_name(anchor)["m1760"].net + 0.5:
        return anchor
    return best


def _best_m1760(results: list[EightResult]) -> EightResult:
    pool = [item for item in results if item.spec.id.startswith("m1760-")]
    return max(pool, key=lambda item: (_by_name(item)["m1760"].net, item.net))


def report_m1760(results: list[EightResult]) -> str:
    """Сетка правил только для окна 1760 на фоне уже выбранных ролей остальных."""
    base = _pick(results, "base")
    anchor = _pick(results, "m1760-none")
    chosen = choose_m1760(results)
    raw = _best_m1760(results)
    base_names = _by_name(base)
    anchor_names = _by_name(anchor)
    rows = [item for item in results if item.spec.id == "base" or item.spec.id.startswith("m1760-")]
    rows.sort(key=lambda item: _by_name(item)["m1760"].net, reverse=True)
    width = max(len(item.spec.title) for item in rows)
    lines = [
        "Окно 1 760 при уже выбранных ролях остальных счетов.",
        "Окна 30 и 45: «старшее, 2 дня, только плюс». Окно 60: «старшее, плюс сразу».",
        "Короткое 525: четверо старших, 1 день в плюсе. Окно 2 160 уступает моментуму после двух дней в плюсе.",
        "Длинное окно и моментум сами. Старше 1 760 только 2 160 и 12 420.",
        "Правило годится, если счёт 1 760 выше, а остальные семь счетов не хуже.",
        "Позиция 1 760 глушит окна короче себя и входит в кворум короткого окна.",
        "",
        f"{'ПРАВИЛО':<{width}} {'СДЕЛКИ':>7} {'ЗАПРЕТ':>7} {'ИТОГ 1760':>12} {'К СЕЙЧАС':>12} "
        f"{'60':>10} {'525':>10} {'30':>10} {'СУММА':>14} {'К СЕЙЧАС':>12} {'ПРОСАДКА':>12}",
    ]
    for item in rows:
        names = _by_name(item)
        window = names["m1760"]
        mark = ""
        if item.spec.id == chosen.spec.id:
            mark += "  ←"
        if item.spec.id == anchor.spec.id:
            mark += "  сейчас"
        if item.spec.id == raw.spec.id and raw.spec.id != chosen.spec.id:
            mark += "  максимум 1760"
        lines.append(
            f"{item.spec.title:<{width}} {len(window.trades):>7} {window.blocked_setups:>7} "
            f"{_money(window.net):>12} {_signed(window.net - anchor_names['m1760'].net):>12} "
            f"{_signed(names['sixty'].net - anchor_names['sixty'].net):>10} "
            f"{_signed(names['short'].net - anchor_names['short'].net):>10} "
            f"{_signed(names['thirty'].net - anchor_names['thirty'].net):>10} "
            f"{_money(item.net):>14} {_signed(item.net - anchor.net):>12} "
            f"{_money(window.drawdown):>12}{mark}"
        )
    chosen_names = _by_name(chosen)
    raw_names = _by_name(raw)
    watched = ("m1760", "m2160", "short", "sixty", "thirty", "fortyfive", MOMENTUM, "long")
    lines.extend(
        [
            "",
            f"Сейчас у окна 1 760 своего запрета нет: итог {_money(anchor_names['m1760'].net)} руб., "
            f"{_signed(anchor_names['m1760'].net - base_names['m1760'].net)} к базе без взаимодействия, "
            f"сделок {len(anchor_names['m1760'].trades)}.",
            f"Взято «{chosen.spec.title}»: итог окна 1 760 {_money(chosen_names['m1760'].net)} руб., "
            f"{_signed(chosen_names['m1760'].net - anchor_names['m1760'].net)} к прогону без его запрета, "
            f"сумма восьми {_money(chosen.net)} руб., {_signed(chosen.net - anchor.net)} к этому прогону, "
            f"{_signed(chosen.net - base.net)} к базе. "
            f"Просадка суммы {_money(chosen.combined_drawdown)} руб.",
            "Счета к прогону без запрета на 1 760: "
            + ", ".join(
                f"{name} {_signed(chosen_names[name].net - anchor_names[name].net)}" for name in watched
            )
            + ".",
        ]
    )
    if raw.spec.id != chosen.spec.id:
        lines.append(
            f"Самый высокий итог окна 1 760 даёт «{raw.spec.title}»: {_money(raw_names['m1760'].net)} руб., "
            f"{_signed(raw_names['m1760'].net - anchor_names['m1760'].net)} к прогону без запрета. "
            f"Сумма {_signed(raw.net - anchor.net)}. "
            + ", ".join(
                f"{name} {_signed(raw_names[name].net - anchor_names[name].net)}"
                for name in ("short", "sixty", "thirty", "fortyfive", "m2160", MOMENTUM, "long")
                if abs(raw_names[name].net - anchor_names[name].net) >= 0.5
            )
            + ". Оно не взято: кто-то из остальных счетов слабее."
        )
    elif chosen.spec.id == anchor.spec.id:
        lines.append("Отдельный запрет, который поднимает окно 1 760 и не забирает деньги у остальных, не нашёлся.")
    interesting = [anchor, chosen, raw, _pick(results, "m1760-leg")]
    seen: set[str] = set()
    for item in interesting:
        if item.spec.id in seen:
            continue
        seen.add(item.spec.id)
        lines.extend(["", f"Кто глушит окно 1 760 при «{item.spec.title}»:"])
        lines.extend(_block_lines(item.m1760_blocks))
    lines.extend(["", "Части истории, только счёт окна 1 760:"])
    lines.append(f"{'ПРАВИЛО':<{width}} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    shown_ids = {base.spec.id, anchor.spec.id, chosen.spec.id, raw.spec.id}
    shown = [item for item in (base, anchor, chosen, raw) if item.spec.id in shown_ids]
    unique: list[EightResult] = []
    for item in shown:
        if item.spec.id not in {kept.spec.id for kept in unique}:
            unique.append(item)
    for item in unique:
        parts = _part_nets(_by_name(item)["m1760"].trades)
        lines.append(f"{item.spec.title:<{width}} " + " ".join(f"{_money(part):>12}" for part in parts))
    lines.extend(["", "Части истории, сумма восьми счетов:"])
    lines.append(f"{'ПРАВИЛО':<{width}} " + " ".join(f"{'Ч' + str(index):>12}" for index in range(1, 6)))
    for item in unique:
        parts = _part_nets(item.trades)
        lines.append(f"{item.spec.title:<{width}} " + " ".join(f"{_money(part):>12}" for part in parts))
    chart = _m1760_chart_item(results)
    lines.append("")
    lines.append(
        "На графике сумма восьми счетов: без взаимодействия, окно 1 760 без своего запрета "
        f"и «{chart.spec.legend}»."
    )
    lines.append("Нижняя панель — разница этой линии с прогоном, где у окна 1 760 запрета нет.")
    return "\n".join(lines)


def _m1760_chart_item(results: list[EightResult]) -> EightResult:
    """Если запрет не взяли, третья линия — правило двух дней, чтобы было видно, сколько оно забирает."""
    anchor = _pick(results, "m1760-none")
    chosen = choose_m1760(results)
    if chosen.spec.id != anchor.spec.id:
        return chosen
    for item in results:
        if item.spec.id == "m1760-leg":
            return item
    return _best_m1760(results)


def plot_m1760(results: list[EightResult], output: Path) -> Path:
    """Три суммы. Нижняя панель — что меняет правило окна 1760."""
    base = _pick(results, "base")
    anchor = _pick(results, "m1760-none")
    shown = _m1760_chart_item(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_equity, ax_gap) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7.2),
        height_ratios=(3, 1.4),
        layout="constrained",
    )
    dates = [pd.Timestamp(day) for day in base.dates]
    series = (
        (base, "#6b6258", 1.6),
        (anchor, "#1f4e79", 1.6),
        (shown, "#9a3412", 1.8),
    )
    anchor_curve = np.asarray(anchor.curve, dtype=float)
    seen: set[str] = set()
    for item, color, width in series:
        if item.spec.id in seen:
            continue
        seen.add(item.spec.id)
        ax_equity.plot(dates, item.curve, color=color, linewidth=width, label=_legend(item.spec))
        if item.spec.id == shown.spec.id and item.spec.id != anchor.spec.id:
            ax_gap.plot(
                dates,
                np.asarray(item.curve) - anchor_curve,
                color=color,
                linewidth=width,
                label=_legend(item.spec),
            )
    ax_gap.axhline(0.0, color="#6b6258", linewidth=0.8)
    ax_equity.set_title("Сумма восьми счетов")
    ax_equity.set_ylabel("руб.")
    ax_gap.set_ylabel("к 1760 без запрета, руб.")
    ax_gap.set_xlabel("день истекающего контракта")
    ax_equity.legend(frameon=False, loc="upper left")
    ax_equity.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f} млн"))
    ax_gap.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1_000_000:.2f} млн"))
    ax_equity.xaxis.set_major_locator(mdates.YearLocator())
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_m1760_curve(results: list[EightResult], output: Path) -> Path:
    base = _pick(results, "base")
    anchor = _pick(results, "m1760-none")
    shown = _m1760_chart_item(results)
    frame = pd.DataFrame(
        {
            "date": [day.isoformat() for day in base.dates],
            "base": base.curve,
            "m1760_now": anchor.curve,
            "m1760_shown": shown.curve,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def run_m1760_study(
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    views: dict[str, dict[str, pd.DataFrame]],
) -> list[EightResult]:
    """База без взаимодействия и каждое правило окна 1760 на фиксированных остальных ролях."""
    results: list[EightResult] = []
    specs: list[TrendSpec | SplitSpec] = [next(spec for spec in SPECS if spec.id == "base")]
    specs.extend(m1760_spec(option_id, label, rule) for option_id, label, rule in M1760_OPTIONS)
    for spec in specs:
        print(f"правило {spec.id}", flush=True)
        results.append(run_eight(frames, spec, starts, views=views))
    return results


def specs_by_id(selected: str | None) -> tuple[TrendSpec, ...]:
    if not selected:
        return SPECS
    wanted = [item.strip() for item in selected.split(",") if item.strip()]
    known = {spec.id: spec for spec in SPECS}
    missing = [item for item in wanted if item not in known]
    if missing:
        raise SystemExit("нет правил: " + ", ".join(missing))
    picked = tuple(known[item] for item in wanted)
    if not any(spec.id == "base" for spec in picked):
        picked = (known["base"],) + picked
    if not any(spec.id == "legacy" for spec in picked):
        picked = picked[:1] + (known["legacy"],) + picked[1:]
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Восемь счетов и запрет входа против очевидного тренда")
    parser.add_argument("--bars", type=Path, default=Path("data/bars"))
    parser.add_argument("--output", type=Path, default=Path("trend.txt"))
    parser.add_argument("--chart", type=Path, default=Path("trend_equity.png"))
    parser.add_argument("--csv", type=Path, default=Path("trend_equity.csv"))
    parser.add_argument("--family", choices=("m1760", "sixty", "kept", "roles", "grid"), default="m1760")
    parser.add_argument("--rules", help="Для сетки: идентификаторы через запятую. base и legacy добавляются сами")
    args = parser.parse_args(argv)
    frames = load_minutes(args.bars)
    starts = load_trade_starts(args.bars)
    print("окна", flush=True)
    views = build_views(frames)
    if args.family == "m1760":
        results = run_m1760_study(frames, starts, views)
        text = report_m1760(results)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        plot_m1760(results, args.chart)
        write_m1760_curve(results, args.csv)
    elif args.family == "sixty":
        results = run_sixty_study(frames, starts, views)
        text = report_sixty(results)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        plot_sixty(results, args.chart)
        write_sixty_curve(results, args.csv)
    elif args.family == "kept":
        results = run_kept_study(frames, starts, views)
        text = report_kept(results)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        plot_kept(results, args.chart)
        write_kept_curve(results, args.csv)
    elif args.family == "roles":
        results = run_role_study(frames, starts, views)
        text = report_roles(results)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        plot_roles(results, args.chart)
        write_roles_curve(results, args.csv)
    else:
        results = []
        for spec in specs_by_id(args.rules):
            print(f"правило {spec.id}", flush=True)
            results.append(run_eight(frames, spec, starts, views=views))
        text = report(results)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        plot_equity(results, args.chart)
        write_curve(results, args.csv)
    print(args.chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
