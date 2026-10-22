"""Запрет нового входа против старшего окна: функция и короткий прогон."""

from datetime import date

import numpy as np
import pandas as pd

import backtest
from research.cr_momentum_mix import _replay_contract, block_new_entry
from research.cr_momentum_senior import block_momentum_opens, immediate_senior


def test_new_entry_is_blocked_only_against_an_open_senior():
    assert block_new_entry(0, 0, 5, -1) == 0
    assert block_new_entry(0, 0, -4, 1) == 0
    assert block_new_entry(0, 0, 5, 1) == 5
    assert block_new_entry(0, 0, -4, -1) == -4
    assert block_new_entry(0, 0, 5, 0) == 5
    assert block_new_entry(0, 0, 5, None) == 5
    assert block_new_entry(3, 3, 3, -1) == 3
    assert block_new_entry(0, 5, 5, -1) == 5
    assert block_new_entry(0, 0, 0, -1) == 0


def test_momentum_open_stops_against_the_previous_close_and_a_hold_does_not():
    days = np.array(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], dtype="datetime64[D]")
    position = np.array([0.0, 1.0, 1.0, -1.0])
    senior = {date(2024, 1, 2): -1, date(2024, 1, 3): -1, date(2024, 1, 4): -1}
    out, episodes, mask = block_momentum_opens(position, days, senior)
    assert out.tolist() == [0.0, 0.0, 0.0, -1.0]
    assert episodes == 1
    assert mask.tolist() == [False, True, True, False]
    held = np.array([1.0, 1.0])
    held_days = days[:2]
    continued, kept, kept_mask = block_momentum_opens(held, held_days, {date(2024, 1, 2): -1})
    assert continued.tolist() == [1.0, 1.0]
    assert kept == 0
    assert not kept_mask.any()
    flat = {day: 0 for day in senior}
    opened, none_blocked, none_mask = block_momentum_opens(position, days, flat)
    assert opened.tolist() == position.tolist()
    assert none_blocked == 0
    assert not none_mask.any()


def test_immediate_senior_is_the_next_longer_name():
    order = ("thirty", "short", "моментум", "long")
    assert immediate_senior(order, "short") == "моментум"
    assert immediate_senior(order, "моментум") == "long"
    assert immediate_senior(order, "long") is None


def test_replay_does_not_open_a_long_against_a_short_senior():
    day_one = pd.date_range("2024-01-02 10:00", periods=80, freq="min")
    day_two = pd.date_range("2024-01-03 10:00", periods=30, freq="min")
    index = day_one.append(day_two)
    close = np.concatenate([np.full(40, 10.0), np.linspace(10.05, 11.0, 40), np.linspace(11.0, 11.2, 30)])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(index), 100.0),
        },
        index=index,
    )
    window = backtest.WINDOWS[3]
    end, rows = _replay_contract("T", frame, window, cash=100_000.0, trade_from=date(2024, 1, 2), day_sign=None)
    assert rows
    assert sum(row["net"] for row in rows) == end - 100_000.0
    assert any(row["eod"] > 0 for row in rows)
    blocked: list[int] = []
    held = np.zeros(len(frame), dtype=np.int8)
    _end, senior_rows = _replay_contract(
        "T",
        frame,
        window,
        cash=100_000.0,
        trade_from=date(2024, 1, 2),
        day_sign=None,
        entry_senior=np.full(len(frame), -1, dtype=np.int8),
        held_out=held,
        blocked_box=blocked,
    )
    assert blocked == [1]
    assert all(row["eod"] == 0 for row in senior_rows)
    assert _end == 100_000.0
    same, _ = _replay_contract(
        "T",
        frame,
        window,
        cash=100_000.0,
        trade_from=date(2024, 1, 2),
        day_sign=None,
        entry_senior=np.ones(len(frame), dtype=np.int8),
    )
    assert same == end
