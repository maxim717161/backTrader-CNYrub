from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import buy_and_hold, choose_window, run_contract, simulate


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


def test_choose_window_prefers_the_stronger_weaker_half():
    rows = [
        {"channel": 15, "half1": 100.0, "half2": 10.0, "per_contract_2": 4.0},
        {"channel": 60, "half1": 50.0, "half2": 40.0, "per_contract_2": 5.0},
        {"channel": 30, "half1": 80.0, "half2": -1.0, "per_contract_2": 9.0},
        {"channel": 120, "half1": 40.0, "half2": 40.0, "per_contract_2": 4.0},
    ]
    chosen = choose_window(rows)
    assert chosen is not None
    assert chosen["channel"] == 60
    assert choose_window([{"channel": 15, "half1": -1.0, "half2": 10.0, "per_contract_2": 8.0}]) is None
