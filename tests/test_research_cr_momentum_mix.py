"""Стыковка моментума с минутными окнами: знак цели и сумма дневных счетов."""

from datetime import date

import numpy as np
import pandas as pd

import backtest
from research.cr_momentum_mix import _align, _replay_contract, clamp_target


def test_clamp_keeps_only_the_allowed_sign():
    assert clamp_target(4, 1) == 4
    assert clamp_target(-3, -1) == -3
    assert clamp_target(4, -1) == 0
    assert clamp_target(-3, 1) == 0
    assert clamp_target(4, 0) == 0
    assert clamp_target(0, 1) == 0
    assert clamp_target(4, None) == 4


def test_replay_nets_sum_to_the_equity_change_and_the_gate_blocks_a_short():
    index = pd.bdate_range("2024-01-02 10:00", periods=80, freq="min")
    close = np.linspace(10.0, 10.4, len(index))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": np.full(len(index), 100.0),
        },
        index=index,
    )
    window = backtest.WINDOWS[NAMES_THIRTY]
    end, rows = _replay_contract("T", frame, window, cash=100_000.0, trade_from=date(2024, 1, 2), day_sign=None)
    assert rows
    assert sum(row["net"] for row in rows) == end - 100_000.0
    _, blocked = _replay_contract(
        "T",
        frame,
        window,
        cash=100_000.0,
        trade_from=date(2024, 1, 2),
        day_sign={day: 1 for day in {stamp.date() for stamp in index}},
    )
    assert all(row["eod"] >= 0 for row in blocked)


NAMES_THIRTY = 2


def test_aligned_sum_carries_a_missing_day_without_adding_pnl():
    left = pd.DataFrame({"day": [date(2024, 1, 2), date(2024, 1, 3)], "equity": [110.0, 120.0], "net": [10.0, 10.0]})
    right = pd.DataFrame({"day": [date(2024, 1, 2), date(2024, 1, 4)], "equity": [90.0, 100.0], "net": [-10.0, 10.0]})
    total = _align([left, right], [100.0, 100.0])
    assert total["equity"].tolist() == [200.0, 210.0, 220.0]
    assert total["net"].tolist() == [0.0, 10.0, 10.0]
