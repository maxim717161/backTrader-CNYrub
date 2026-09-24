from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import SKIP_SECIDS, buy_and_hold, run_contract


def _days(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2024, 1, 2)
    index = pd.to_datetime([start + timedelta(days=i) for i in range(len(rows))])
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)
    frame.index.name = "datetime"
    frame["flatten_on_close"] = 0.0
    frame["is_last"] = 0.0
    frame.iloc[-1, frame.columns.get_loc("is_last")] = 1.0
    frame.iloc[-2, frame.columns.get_loc("flatten_on_close")] = 1.0
    return frame


def _quiet(close: float = 10.0, volume: float = 1000.0) -> tuple[float, float, float, float, float]:
    return (close, close + 0.05, close - 0.05, close, volume)


def test_stop_is_measured_from_the_fill_and_crm2_is_not_a_traded_window():
    assert "CRM2" in SKIP_SECIDS
    rows = [_quiet() for _ in range(20)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.append((10.30, 10.50, 10.20, 10.40, 1000))
    rows.extend((_quiet(10.40)[:4] + (1000,)) for _ in range(4))
    rows.append((10.40, 10.45, 10.00, 10.10, 1000))
    rows.extend(_quiet(10.0) for _ in range(4))
    strategy = run_contract("CRZ5", _days(rows))
    assert len(strategy.trades) == 1
    trade = strategy.trades[0]
    assert trade["direction"] == "long"
    assert trade["reason"] == "stop"
    assert trade["pnl"] == pytest.approx(-250.0)
    assert trade["pnlcomm"] == pytest.approx(-252.0)


def test_open_position_is_closed_on_the_last_day_and_not_reopened():
    rows = [_quiet() for _ in range(20)]
    rows.append((10.0, 10.40, 10.20, 10.30, 1000))
    rows.extend((10.30, 10.50, 10.20, 10.45, 1000) for _ in range(8))
    frame = _days(rows)
    strategy = run_contract("CRH6", frame)
    assert len(strategy.trades) == 1
    trade = strategy.trades[0]
    assert trade["reason"] == "expiry"
    entry = float(frame["open"].iloc[21])
    exit_ = float(frame["open"].iloc[-1])
    assert trade["pnl"] == (exit_ - entry) * 1000
    assert buy_and_hold(frame) == trade["pnlcomm"]
