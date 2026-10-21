"""Запрет обратного входа, когда чужая позиция уже несколько дней в растущем плюсе."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest import Window, simulate
from consensus import (
    RULES,
    ConsensusRule,
    OpenState,
    _FillBook,
    cancel_entry,
    confirms,
    position_state,
    report,
    run_consensus,
    suppress_entry,
    veto_side,
    veto_sources,
)
from cnyrub.engine import step_minute


def _state(
    *,
    name: str = "long",
    channel: int = 12_420,
    side: int = -1,
    held_days: int = 4,
    pnl: float = 1_000.0,
    prev_pnl: float | None = 400.0,
    prev2_pnl: float | None = 200.0,
    ranges: float = 30.0,
) -> OpenState:
    return OpenState(name, channel, side, held_days, pnl, prev_pnl, prev2_pnl, ranges)


def _rule(**overrides: object) -> ConsensusRule:
    data: dict[str, object] = dict(id="probe", title="probe", hold_days=3, growing="soft")
    data.update(overrides)
    return ConsensusRule(**data)  # type: ignore[arg-type]


def test_short_in_a_growing_profit_blocks_other_buys_only():
    state = _state(side=-1)
    rule = _rule()
    assert veto_side([state], rule, "thirty", 30) == 1
    assert veto_side([state], rule, "long", 12_420) == 0
    assert veto_sources([state], rule, "thirty", 30, 1) == ["long"]
    assert veto_sources([state], rule, "thirty", 30, -1) == []


def test_same_direction_entry_stays_open_and_a_young_position_does_not_count():
    rule = _rule()
    assert veto_side([_state(side=1)], rule, "thirty", 30) == -1
    assert veto_side([_state(held_days=2)], rule, "thirty", 30) == 0
    assert not confirms(_state(pnl=-1.0), rule)


def test_strict_growth_rejects_a_dip_that_soft_growth_still_treats_as_a_trend():
    dipped = _state(pnl=100.0, prev_pnl=400.0, prev2_pnl=200.0)
    strict = _rule(growing="strict")
    soft = _rule(growing="soft")
    plus = _rule(growing="no")
    assert not confirms(dipped, strict)
    assert confirms(dipped, soft)
    assert confirms(dipped, plus)
    stale = _state(pnl=200.0, prev_pnl=250.0, prev2_pnl=400.0)
    assert not confirms(stale, soft)
    assert confirms(stale, plus)
    assert not confirms(_state(pnl=-10.0, prev_pnl=400.0, prev2_pnl=200.0), soft)


def test_only_the_long_window_or_only_a_slower_window_may_veto():
    short = _state(name="short", channel=525, side=-1)
    slow = _state(name="long", channel=12_420, side=-1)
    thirty = _state(name="thirty", channel=30, side=-1)
    fortyfive = _state(name="fortyfive", channel=45, side=-1)
    sixty = _state(name="sixty", channel=60, side=-1)
    rule = _rule(source="slower")
    assert veto_side([short, slow], _rule(source="long"), "thirty", 30) == 1
    assert veto_side([short], _rule(source="long"), "thirty", 30) == 0
    assert veto_side([slow], _rule(shielded=frozenset({"thirty"})), "thirty", 30) == 0
    assert veto_side([thirty], rule, "fortyfive", 45) == 0
    assert veto_side([thirty, fortyfive], rule, "sixty", 60) == 0
    assert veto_side([sixty], rule, "short", 525) == 0
    assert veto_side([short], rule, "long", 12_420) == 0
    assert veto_side([fortyfive], rule, "thirty", 30) == 1
    assert veto_side([sixty], rule, "fortyfive", 45) == 1
    assert veto_side([short], rule, "sixty", 60) == 1
    assert veto_side([slow], rule, "short", 525) == 1


def test_two_profitable_sides_block_both_entries():
    states = [_state(name="long", side=-1), _state(name="short", channel=525, side=1)]
    assert veto_side(states, _rule(), "thirty", 30) == 2


def test_wide_profit_needs_enough_ranges():
    rule = _rule(min_ranges=20.0)
    assert confirms(_state(ranges=20.0), rule)
    assert not confirms(_state(ranges=19.0), rule)


def test_position_state_counts_trading_days_and_ignores_an_exit():
    book = _FillBook("CRZ5", 100_000.0)
    book.held = -10
    book.target = -10
    book.avg = 10.0
    book.entry_i = 1
    book.unit = 0.05
    day_index = np.array([0, 0, 1, 2, 3])
    state = position_state("long", 12_420, book, 4, day_index, 9.0, 4_000.0, 1_000.0)
    assert state is not None
    assert state.side == -1
    assert state.held_days == 3
    assert state.pnl == pytest.approx(10_000.0)
    assert state.ranges == pytest.approx(20.0)
    book.target = 0
    assert position_state("long", 12_420, book, 4, day_index, 9.0, None, None) is None


def test_suppress_entry_drops_a_fresh_target_and_leaves_an_open_position():
    book = _FillBook("CRZ5", 100_000.0)
    step_minute(
        book,
        10,
        opened=10.0,
        high=10.3,
        low=9.9,
        close=10.2,
        volume=100.0,
        day=datetime(2024, 6, 3).date(),
        next_day=datetime(2024, 6, 4).date(),
        last_day=datetime(2024, 9, 18).date(),
        prior_high=10.0,
        prior_low=9.5,
        prior_vol=80.0,
        prior_range=0.05,
        exit_high=float("nan"),
        exit_low=float("nan"),
        clock_vol=40.0,
        entry_ready=1.0,
        clearance=0.0,
        cooldown=0,
        clock_volume=False,
        clock_cap=5.0,
        size_mode="flat",
        stop_mult=8.0,
        loss_bars=None,
        risk_fraction=None,
        margin=1_000.0,
        stop_rub=None,
        breakout_span=None,
        trade_from=datetime(2024, 6, 1).date(),
        trail=False,
        leverage=4.0,
    )
    assert book.held == 0
    assert book.target > 0
    assert not suppress_entry(book, -1)
    assert book.target > 0
    assert suppress_entry(book, 1)
    assert book.target == 0
    assert book.base == 0
    book.held = 4
    book.target = 4
    assert not suppress_entry(book, 2)
    assert book.held == 4
    cancel_entry(book)
    assert book.target == 0


def _days(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2024, 1, 2)
    index = pd.to_datetime([start + timedelta(days=i) for i in range(len(rows))])
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)
    frame.index.name = "datetime"
    return frame


def _quiet(close: float = 10.0) -> tuple[float, float, float, float, float]:
    return (close, close + 0.05, close - 0.05, close, 1000.0)


def test_baseline_matches_the_independent_simulator():
    rows = [_quiet() for _ in range(20)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000.0))
    rows.append((10.30, 10.50, 10.20, 10.40, 1000.0))
    rows.extend(_quiet(10.40) for _ in range(4))
    rows.append((10.40, 10.45, 10.00, 10.10, 1000.0))
    rows.extend(_quiet(10.0) for _ in range(4))
    frame = _days(rows)
    window = Window(20, 10, 2.5, cash=100_000.0, margin=20_000.0)
    alone = simulate(
        "CRZ5",
        frame,
        20,
        stop_mult=2.5,
        exit_channel=10,
        cash=100_000.0,
        margin=20_000.0,
    )
    result = run_consensus(
        {"CRZ5": frame},
        RULES[0],
        names=("probe",),
        windows=(window,),
    )
    stats = result.strategies[0]
    assert stats.blocked_setups == 0
    assert len(stats.trades) == len(alone) == 1
    assert stats.trades[0]["pnlcomm"] == pytest.approx(alone[0]["pnlcomm"])
    assert stats.trades[0]["direction"] == alone[0]["direction"]
    assert stats.trades[0]["reason"] == alone[0]["reason"]
    assert stats.end_equity == pytest.approx(stats.cash + stats.net)


def _bar(close: float, volume: float = 1000.0, high: float | None = None, low: float | None = None):
    return (
        close,
        close + 0.04 if high is None else high,
        close - 0.04 if low is None else low,
        close,
        volume,
    )


def test_a_multi_day_short_rejects_the_other_strategy_long_breakout():
    # Общая просадка заводит обоих в шорт. Всплеск выбивает стоп только у второй.
    # Дальше тихие минутки: первая остаётся в растущем плюсе, вторая не перезаходит,
    # потому что объём ниже медианы. Отскок выше короткого канала, но ниже цены
    # входа шорта, без фильтра становится лонгом.
    rows = [_bar(10.0) for _ in range(8)]
    rows.append(_bar(9.30, high=10.02, low=9.20))
    rows.append(_bar(9.25, high=9.40, low=9.20))
    rows.append(_bar(9.35, high=9.70, low=9.20))
    for close in (9.10, 8.90, 8.70, 8.50, 8.30, 8.10):
        rows.append(_bar(close, volume=10.0))
    rows.append(_bar(9.00, volume=5000.0, high=9.05, low=8.05))
    rows.extend(_bar(9.00) for _ in range(2))
    frame = _days(rows)
    holder = Window(5, 0, None, cash=100_000.0, margin=1_000.0)
    other = Window(5, 0, 0.5, cash=100_000.0, margin=1_000.0)
    names = ("holder", "other")
    windows = (holder, other)
    base = run_consensus({"CRZ5": frame}, RULES[0], names=names, windows=windows)
    filtered = run_consensus(
        {"CRZ5": frame},
        ConsensusRule("d2", "2 дня, плюс растёт", 2, "soft"),
        names=names,
        windows=windows,
    )
    base_other = base.strategies[1].trades
    filtered_other = filtered.strategies[1].trades
    assert any(trade["direction"] == "long" for trade in base_other)
    assert filtered.strategies[1].blocked_setups >= 1
    assert not any(trade["direction"] == "long" for trade in filtered_other)
    assert filtered.source_setups["holder"] >= 1
    assert any(trade["direction"] == "short" for trade in filtered.strategies[0].trades)


def test_report_names_the_rule_and_the_base_gap():
    text = report([])
    assert text == "Нет прогонов."
    assert [rule.id for rule in RULES].count("base") == 1
    assert len({rule.id for rule in RULES}) == len(RULES)


def test_conclusion_names_the_stronger_sum_and_the_shallower_drawdown():
    from consensus import ConsensusResult, StrategyStats, conclusion

    names = ("short", "long", "thirty", "fortyfive", "sixty")

    def pack(rule: ConsensusRule, nets: tuple[float, ...], drawdown: float) -> ConsensusResult:
        strategies = []
        for name, net in zip(names, nets, strict=True):
            trades = [{"secid": "CRU2", "pnlcomm": net}] if net else []
            strategies.append(StrategyStats(name, trades, 0, 0, 0, 100_000.0 + net, 0.0, 100_000.0))
        return ConsensusResult(rule, strategies, drawdown, dict.fromkeys(names, 0))

    base = pack(RULES[0], (10.0, 20.0, 5.0, 5.0, 5.0), -100.0)
    grows = pack(next(rule for rule in RULES if rule.id == "d3-grows"), (10.0, 5.0, 5.0, 5.0, 5.0), -80.0)
    fast = pack(next(rule for rule in RULES if rule.id == "fast-d3"), (10.0, 20.0, 8.0, 9.0, 5.0), -90.0)
    slower = pack(
        next(rule for rule in RULES if rule.id == "slower-d3"),
        (4.0, 20.0, 4.0, 4.0, 4.0),
        -70.0,
    )
    text = conclusion([base, grows, fast, slower])
    assert "Выше базы по сумме" in text
    assert "глушит только 30/45/60" in text
    assert "3 дня, плюс растёт" in text
    assert "Длинное окно" in text
    assert "короткое окно не глушит длинное" in text
    assert "старшее, 3 дня, плюс растёт" in text
    full = report([base, grows, fast])
    assert "Выше базы по сумме" in full
