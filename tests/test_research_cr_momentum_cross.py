"""Свежий вход моментума из нуля снимается тем же запретом, разворот открытой позиции — нет."""

from datetime import date

import numpy as np
import pandas as pd

from consensus import ADOPTED_RULE, OpenState, confirms
from research.cr_momentum_cross import LONG_RULE, MomentumAccount, replay_momentum
from research.cr_momentum_veto import momentum_veto

import research.core as core
import research.cr_momentum_params as cp


def _state(name: str, channel: int, side: int, held_days: int, pnl: float) -> OpenState:
    return OpenState(name, channel, side, held_days, pnl, None, None, 0.0)


def test_fresh_entry_from_flat_is_blocked_and_a_reversal_is_not():
    core.configure(1000.0, 0.001)
    days = np.array(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"], dtype="datetime64[D]")
    sign = np.array([1.0, 1.0, -1.0, 0.0, 1.0])
    level = np.array([10.0, 10.1, 10.2, 10.0, 10.1])
    ret = np.array([0.1, 0.1, 0.1, -0.2, 0.1])
    roll = np.array([False, False, False, False, False])
    veto = {
        date(2024, 1, 2): 0,
        date(2024, 1, 3): 0,
        date(2024, 1, 4): -1,
        date(2024, 1, 5): 0,
        date(2024, 1, 8): 1,
    }
    account = replay_momentum(days, sign, level, ret, roll, veto.get)
    assert account.contracts[0] > 0
    assert account.contracts[2] < 0
    assert account.contracts[3] == 0
    assert account.contracts[4] == 0
    assert account.blocked == 1


def test_ungated_account_matches_the_leverage_curve():
    core.configure(1000.0, 0.001)
    days = np.array(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"], dtype="datetime64[D]")
    sign = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
    level = np.array([10.0, 10.2, 10.1, 10.4, 10.3])
    ret = np.array([0.05, 0.2, -0.1, 0.3, -0.1])
    roll = np.array([False, False, True, False, False])
    frame = pd.DataFrame({"day": pd.to_datetime(days), "position": sign})
    curve = cp.leverage_curve(frame, level, ret, roll, cp.CHOSEN_LEVERAGE, cp.ACCOUNT_CASH)
    account = MomentumAccount(len(days))
    for t in range(len(days)):
        account.step(t, float(sign[t]), float(level[t]), float(ret[t]), bool(roll[t]), 0)
    assert np.allclose(account.equity, curve["equity"].to_numpy(dtype=float))
    assert np.allclose(account.contracts, curve["contracts"].to_numpy(dtype=float))


def test_long_window_confirms_only_after_two_days_in_profit():
    fresh = _state("long", 12420, 1, 0, 0.0)
    ready = _state("long", 12420, 1, 2, 15.0)
    losing = _state("long", 12420, 1, 2, -15.0)
    short = _state("short", 525, -1, 4, 20.0)
    assert momentum_veto(fresh) == 0
    assert confirms(ready, ADOPTED_RULE)
    assert confirms(losing, ADOPTED_RULE) is False
    from consensus import veto_side

    assert veto_side([ready], LONG_RULE, "momentum", 0) == -1
    assert veto_side([losing], LONG_RULE, "momentum", 0) == 0
    assert veto_side([short], LONG_RULE, "momentum", 0) == 0
    assert veto_side([short, ready], ADOPTED_RULE, "momentum", 0) == 2
