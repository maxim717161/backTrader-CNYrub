"""Взгляд на моментум тем же правилом, что запрет старшего окна: два дня в плюсе."""

from datetime import date

import numpy as np

from consensus import ADOPTED_RULE, confirms
from research.cr_momentum_veto import combine_veto, momentum_veto, momentum_watch


def test_momentum_blocks_only_after_two_profitable_days_and_not_on_the_entry():
    days = np.array(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"], dtype="datetime64[D]")
    nets = np.array([10.0, 5.0, -30.0, 1.0, 4.0])
    contracts = np.array([3.0, 3.0, 3.0, 0.0, -2.0])
    watch = momentum_watch(days, nets, contracts)
    assert momentum_veto(watch[date(2024, 1, 2)]) == 0
    assert momentum_veto(watch[date(2024, 1, 3)]) == 0
    assert watch[date(2024, 1, 4)].held_days == 2
    assert watch[date(2024, 1, 4)].pnl == 15.0
    assert momentum_veto(watch[date(2024, 1, 4)]) == -1
    assert date(2024, 1, 5) not in watch
    assert momentum_veto(watch[date(2024, 1, 8)]) == 0
    assert confirms(watch[date(2024, 1, 4)], ADOPTED_RULE)


def test_a_losing_episode_does_not_block_even_after_two_days():
    days = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    watch = momentum_watch(days, np.array([-10.0, -5.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert momentum_veto(watch[date(2024, 1, 4)]) == 0
    assert watch[date(2024, 1, 4)].pnl == -15.0


def test_window_veto_and_momentum_veto_combine_without_cancelling_each_other():
    assert combine_veto(0, 0) == 0
    assert combine_veto(0, -1) == -1
    assert combine_veto(-1, -1) == -1
    assert combine_veto(1, -1) == 2
    assert combine_veto(2, -1) == 2
