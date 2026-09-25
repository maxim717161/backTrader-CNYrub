from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from backtest import (
    WINDOWS,
    _part_nets,
    breakout_fraction,
    breakout_lots,
    buy_and_hold,
    choose_window,
    entry_lots,
    load_trade_starts,
    refine_windows,
    run_contract,
    simulate,
)


def _days(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2024, 1, 2)
    index = pd.to_datetime([start + timedelta(days=i) for i in range(len(rows))])
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)
    frame.index.name = "datetime"
    return frame


def _quiet(close: float = 10.0, volume: float = 1000.0) -> tuple[float, float, float, float, float]:
    return (close, close + 0.05, close - 0.05, close, volume)


def _stop_frame() -> pd.DataFrame:
    rows = [_quiet() for _ in range(20)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.append((10.30, 10.50, 10.20, 10.40, 1000))
    rows.extend((_quiet(10.40)[:4] + (1000,)) for _ in range(4))
    rows.append((10.40, 10.45, 10.00, 10.10, 1000))
    rows.extend(_quiet(10.0) for _ in range(4))
    return _days(rows)


def test_stop_is_measured_from_the_fill_in_backtrader_and_the_simulator():
    frame = _stop_frame()
    simulated = simulate("CRZ5", frame, 20)
    strategy = run_contract("CRZ5", frame, 20)
    assert len(simulated) == 1
    assert len(strategy.trades) == 1
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["direction"] == "long"
        assert trade["reason"] == "stop"
        assert trade["pnl"] == pytest.approx(-250.0)
        assert trade["pnlcomm"] == pytest.approx(-252.0)


def test_tighter_stop_without_a_channel_exit_matches_backtrader():
    frame = _stop_frame()
    simulated = simulate("CRZ5", frame, 20, stop_mult=2.0, exit_channel=0)
    strategy = run_contract("CRZ5", frame, 20, stop_mult=2.0, exit_channel=0)
    assert len(simulated) == 1
    assert len(strategy.trades) == 1
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["direction"] == "long"
        assert trade["reason"] == "stop"
        assert trade["pnl"] == pytest.approx(-200.0)
        assert trade["pnlcomm"] == pytest.approx(-202.0)


def test_without_a_stop_the_position_is_held_until_expiry():
    frame = _stop_frame()
    simulated = simulate("CRZ5", frame, 20, stop_mult=None, exit_channel=0)
    strategy = run_contract("CRZ5", frame, 20, stop_mult=None, exit_channel=0)
    assert len(simulated) == 1
    assert len(strategy.trades) == 1
    entry = float(frame["open"].iloc[21])
    exit_ = float(frame["open"].iloc[-1])
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["reason"] == "expiry"
        assert trade["pnl"] == pytest.approx((exit_ - entry) * 1000)
        assert trade["pnlcomm"] == pytest.approx((exit_ - entry) * 1000 - 2)


def test_open_position_is_closed_on_the_last_day_and_not_reopened():
    rows = [_quiet() for _ in range(20)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.extend((10.30, 10.50, 10.20, 10.45, 1000) for _ in range(8))
    frame = _days(rows)
    simulated = simulate("CRH6", frame, 20)
    strategy = run_contract("CRH6", frame, 20)
    assert len(simulated) == len(strategy.trades) == 1
    entry = float(frame["open"].iloc[21])
    exit_ = float(frame["open"].iloc[-1])
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["reason"] == "expiry"
        assert trade["pnl"] == (exit_ - entry) * 1000
        assert buy_and_hold(frame, 20) == trade["pnlcomm"]


def test_breakout_below_the_median_volume_is_skipped():
    rows = [_quiet() for _ in range(20)]
    rows.append((10.0, 10.40, 10.20, 10.30, 100))
    rows.extend(_quiet(10.40) for _ in range(8))
    frame = _days(rows)
    assert simulate("CRU5", frame, 20) == []
    assert run_contract("CRU5", frame, 20).trades == []


def _scored(channel: int, parts: tuple[float, ...], per: tuple[float, ...] | None = None) -> dict[str, object]:
    return {
        "channel": channel,
        "parts": parts,
        "per_contract": (4.0, 4.0, 4.0, 4.0, 4.0) if per is None else per,
    }


def test_part_nets_treat_a_contract_without_trades_as_zero():
    parts = _part_nets([{"secid": "CRU2", "pnlcomm": 10.0}])
    assert parts[0] == 10.0
    assert parts[1:] == (0.0, 0.0, 0.0, 0.0)


def test_choose_window_prefers_the_stronger_weakest_part():
    rows = [
        _scored(480, (100.0, 80.0, 70.0, 60.0, 10.0)),
        _scored(960, (50.0, 40.0, 45.0, 40.0, 40.0)),
        _scored(1440, (80.0, -1.0, 90.0, 90.0, 90.0)),
        _scored(1920, (40.0, 40.0, 40.0, 40.0, 40.0)),
        _scored(2400, (5.0, 5.0, 5.0, 5.0, 5.0), (4.0, 4.0, 1.0, 4.0, 4.0)),
    ]
    chosen = choose_window(rows)
    assert chosen is not None
    assert chosen["channel"] == 960
    assert choose_window([_scored(480, (10.0, 10.0, -1.0, -1.0, -1.0))]) is None
    assert choose_window([_scored(720, (10.0, 10.0, 10.0, -5.0, -5.0))]) is not None


def test_clearance_blocks_a_close_that_only_touches_the_channel():
    rows = [_quiet() for _ in range(5)]
    rows.append((10.0, 10.08, 9.98, 10.07, 1000))
    rows.extend(_quiet(10.07) for _ in range(3))
    frame = _days(rows)
    assert len(simulate("CRZ5", frame, 5)) == 1
    assert simulate("CRZ5", frame, 5, clearance=0.5) == []


def test_clock_volume_compares_with_the_same_minute_of_prior_days():
    start = datetime(2024, 1, 2, 10, 0)
    stamps = []
    values = []
    for day in range(5):
        stamps.append(start + timedelta(days=day))
        values.append((10.0, 10.05, 9.95, 10.0, 1000))
        stamps.append(start + timedelta(days=day, minutes=1))
        values.append((10.0, 10.05, 9.95, 10.0, 1))
    signal = start + timedelta(days=5)
    stamps.extend((signal, signal + timedelta(minutes=1), signal + timedelta(days=1)))
    values.extend(
        (
            (10.0, 10.40, 10.20, 10.30, 10),
            (10.30, 10.40, 10.20, 10.30, 1),
            (10.30, 10.40, 10.20, 10.30, 1),
        )
    )
    frame = pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"], index=pd.to_datetime(stamps))
    assert len(simulate("CRU5", frame, 1)) == 1
    assert simulate("CRU5", frame, 1, clock_volume=True) == []


def test_clock_cap_rejects_a_loud_same_minute_and_keeps_the_window_floor():
    start = datetime(2024, 1, 2, 10, 0)
    stamps = []
    values = []
    for day in range(5):
        for minute, volume in ((0, 10.0), (1, 100.0), (2, 100.0)):
            stamps.append(start + timedelta(days=day, minutes=minute))
            values.append((10.0, 10.05, 9.95, 10.0, volume))
    signal = start + timedelta(days=5)
    stamps.extend((signal, signal + timedelta(minutes=1), signal + timedelta(days=1)))
    values.extend(
        (
            (10.0, 10.40, 10.10, 10.30, 100.0),
            (10.30, 10.40, 10.20, 10.35, 100.0),
            (10.35, 10.40, 10.20, 10.35, 100.0),
        )
    )
    frame = pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"], index=pd.to_datetime(stamps))
    opened = simulate("CRU5", frame, 2, stop_mult=None, exit_channel=2)
    capped = simulate("CRU5", frame, 2, stop_mult=None, exit_channel=2, clock_cap=5)
    assert len(opened) == 1
    assert capped == []
    assert run_contract("CRU5", frame, 2, stop_mult=None, exit_channel=2, clock_cap=5).trades == []


def test_inverse_size_uses_three_lots_on_a_fresh_breakout():
    rows = [_quiet() for _ in range(5)]
    rows.append((10.0, 10.12, 10.06, 10.10, 1000))
    rows.extend(_quiet(10.20) for _ in range(3))
    frame = _days(rows)
    simulated = simulate("CRZ5", frame, 5, stop_mult=None, exit_channel=0, size_mode="inverse")
    strategy = run_contract("CRZ5", frame, 5, stop_mult=None, exit_channel=0, size_mode="inverse")
    assert len(simulated) == len(strategy.trades) == 1
    entry = float(frame["open"].iloc[6])
    exit_ = float(frame["open"].iloc[-1])
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["lots"] == 3
        assert trade["reason"] == "expiry"
        assert trade["pnl"] == pytest.approx((exit_ - entry) * 3 * 1000)
        assert trade["pnlcomm"] == pytest.approx((exit_ - entry) * 3 * 1000 - 6)


def _front_frame() -> pd.DataFrame:
    rows = [_quiet() for _ in range(5)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.extend(_quiet(10.40) for _ in range(3))
    rows.append((10.80, 10.90, 10.70, 10.80, 1000))
    return _days(rows)


def test_warmup_bars_form_the_channel_and_the_fill_opens_the_front_contract():
    frame = _front_frame()
    front = frame.index[6].date()
    simulated = simulate("CRU2", frame, 5, stop_mult=None, exit_channel=0, trade_from=front)
    strategy = run_contract("CRU2", frame, 5, stop_mult=None, exit_channel=0, trade_from=front)
    assert len(simulated) == len(strategy.trades) == 1
    entry = float(frame["open"].iloc[6])
    exit_ = float(frame["open"].iloc[-1])
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["reason"] == "expiry"
        assert trade["pnl"] == pytest.approx((exit_ - entry) * 1000)
        assert trade["pnlcomm"] == pytest.approx((exit_ - entry) * 1000 - 2)


def test_a_fill_during_the_warmup_month_is_not_a_trade():
    frame = _front_frame()
    front = frame.index[7].date()
    assert simulate("CRU2", frame, 5, stop_mult=None, exit_channel=0, trade_from=front) == []
    assert run_contract("CRU2", frame, 5, stop_mult=None, exit_channel=0, trade_from=front).trades == []


def test_trade_starts_follow_the_expiring_contract_not_the_warmup_month():
    starts = load_trade_starts()
    assert starts["CRM2"] == date(2022, 4, 21)
    assert starts["CRU2"] == date(2022, 6, 17)
    assert starts["CRZ6"] == date(2026, 9, 18)


def test_a_fixed_ruble_stop_stays_inside_ten_percent_of_the_account():
    rows = [_quiet() for _ in range(5)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.append((10.30, 10.35, 9.80, 9.90, 1000))
    rows.append(_quiet(9.90))
    frame = _days(rows)
    cash = 10_000.0
    simulated = simulate(
        "CRZ5", frame, 5, stop_mult=None, exit_channel=0,
        risk_fraction=0.10, margin=1_000.0, stop_rub=285.0, cash=cash,
    )
    strategy = run_contract(
        "CRZ5", frame, 5, stop_mult=None, exit_channel=0,
        risk_fraction=0.10, margin=1_000.0, stop_rub=285.0, cash=cash,
    )
    assert len(simulated) == len(strategy.trades) == 1
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["lots"] == 3
        assert trade["reason"] == "stop"
        assert trade["pnl"] == pytest.approx(-285.0 * 3)
        assert trade["pnlcomm"] == pytest.approx(-285.0 * 3 - 6)
        assert trade["pnlcomm"] >= -0.10 * cash


def test_a_trade_still_negative_after_the_time_limit_is_closed():
    rows = [_quiet() for _ in range(5)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.append((10.30, 10.35, 10.22, 10.25, 1000))
    rows.append((10.25, 10.28, 10.20, 10.24, 1000))
    rows.append((10.22, 10.26, 10.18, 10.24, 1000))
    rows.append(_quiet(10.24))
    frame = _days(rows)
    simulated = simulate("CRZ5", frame, 5, stop_mult=None, exit_channel=0, loss_bars=1)
    strategy = run_contract("CRZ5", frame, 5, stop_mult=None, exit_channel=0, loss_bars=1)
    assert len(simulated) == len(strategy.trades) == 1
    entry = float(frame["open"].iloc[6])
    exit_ = float(frame["open"].iloc[8])
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["reason"] == "time"
        assert trade["pnl"] == pytest.approx((exit_ - entry) * 1000)


def test_the_time_limit_wins_when_the_channel_breaks_on_the_same_bar():
    rows = [_quiet() for _ in range(5)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.append((10.30, 10.35, 10.22, 10.25, 1000))
    rows.append((10.25, 10.26, 9.40, 9.50, 1000))
    rows.append((9.50, 9.55, 9.45, 9.50, 1000))
    rows.append(_quiet(9.50))
    frame = _days(rows)
    simulated = simulate("CRZ5", frame, 5, stop_mult=None, exit_channel=5, loss_bars=1)
    strategy = run_contract("CRZ5", frame, 5, stop_mult=None, exit_channel=5, loss_bars=1)
    assert len(simulated) == len(strategy.trades) == 1
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["reason"] == "time"
        assert trade["pnl"] == pytest.approx(-800.0)


def test_breakout_percent_scales_contracts_down_from_the_margin_cap():
    assert breakout_fraction(0, 12) == 1
    assert breakout_fraction(1.5, 12) == pytest.approx(0.5)
    assert breakout_fraction(1, 12) > 0.5
    assert breakout_fraction(2, 12) < 0.5
    assert breakout_fraction(12, 12) == 0
    assert breakout_fraction(float("inf"), 12) == 0
    assert breakout_lots(100_000, 1_000, 0, 12) == 99
    assert breakout_lots(100_000, 1_000, 1.5, 12) == 49
    assert breakout_lots(100_000, 1_000, 12, 12) is None
    rows = [(10.0, 10.5, 9.5, 10.0, 1000.0) for _ in range(5)]
    rows.append((10.0, 12.0, 10.0, 12.0, 1000.0))
    rows.append((12.0, 12.1, 11.9, 12.0, 1000.0))
    rows.append((12.0, 12.1, 11.9, 12.0, 1000.0))
    frame = _days(rows)
    kwargs = dict(stop_mult=22.0, exit_channel=0, breakout_span=12.0, cash=10_000.0, margin=1_000.0)
    simulated = simulate("CRZ5", frame, 5, **kwargs)
    strategy = run_contract("CRZ5", frame, 5, **kwargs)
    assert len(simulated) == len(strategy.trades) == 1
    for trade in (simulated[0], strategy.trades[0]):
        assert trade["lots"] == 4
        assert trade["pnlcomm"] == pytest.approx(-8)
    far = rows.copy()
    far[5] = (10.0, 22.5, 10.0, 22.5, 1000.0)
    assert simulate("CRZ5", _days(far), 5, **kwargs) == []
    assert run_contract("CRZ5", _days(far), 5, **kwargs).trades == []


def test_entry_lots_step_down_as_the_breakout_grows():
    assert entry_lots("inverse", 1, 10.5, 10.0, 9.0, 1.0) == 3
    assert entry_lots("inverse", 1, 11.0, 10.0, 9.0, 1.0) == 2
    assert entry_lots("inverse", 1, 11.9, 10.0, 9.0, 1.0) == 2
    assert entry_lots("inverse", 1, 12.0, 10.0, 9.0, 1.0) == 1
    assert entry_lots("flat", 1, 10.10, 10.05, 9.95, 0.10) == 1
    assert WINDOWS[0].size_mode == "flat" and WINDOWS[0].clock_cap == 5
    assert WINDOWS[0].loss_bars == 1450 and WINDOWS[0].risk_fraction == 0.10
    assert WINDOWS[0].stop_rub == 285
    assert WINDOWS[1].size_mode == "span" and WINDOWS[1].clock_cap is None
    assert WINDOWS[1].breakout_span == 12 and WINDOWS[1].cash == 100_000
    assert WINDOWS[1].margin == 1_000


def test_refine_does_not_go_below_480_and_looks_past_the_upper_edge():
    rows = [
        _scored(480, (1.0, 1.0, 1.0, 1.0, 1.0)),
        _scored(960, (-1.0, -1.0, -1.0, 1.0, 1.0)),
        _scored(14400, (2.0, 2.0, 2.0, 2.0, 2.0)),
    ]
    extra = refine_windows(rows)
    assert 360 not in extra
    assert 600 in extra
    assert 14880 in extra
    assert 14400 not in extra
