from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import buy_and_hold, choose_window, refine_windows, run_contract, simulate


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
    assert choose_window([_scored(480, (-1.0, 10.0, 10.0, 10.0, 10.0))]) is None


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


def test_refine_does_not_go_below_480_and_looks_past_the_upper_edge():
    rows = [
        _scored(480, (1.0, 1.0, 1.0, 1.0, 1.0)),
        _scored(960, (-1.0, 1.0, 1.0, 1.0, 1.0)),
        _scored(14400, (2.0, 2.0, 2.0, 2.0, 2.0)),
    ]
    extra = refine_windows(rows)
    assert 360 not in extra
    assert 600 in extra
    assert 14880 in extra
    assert 14400 not in extra
