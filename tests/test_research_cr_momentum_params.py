import numpy as np
import pandas as pd
import pytest

from research import core, cr_momentum as cm
from research import cr_momentum_params as cp


@pytest.fixture(autouse=True)
def restore_instrument():
    saved = (core.MULTIPLIER, core.TICK, core.UNIT)
    core.configure(1000.0, 0.001)
    yield
    core.MULTIPLIER, core.TICK, core.UNIT = saved


def make_series(ret, roll_at=(), level=12.0, start="2023-01-02"):
    ret = np.asarray(ret, dtype=float)
    days = pd.bdate_range(start, periods=len(ret))
    roll = np.zeros(len(ret), dtype=bool)
    roll[list(roll_at)] = True
    return cm.Series(
        days=days.to_numpy().astype("datetime64[ns]"),
        contract=np.array(["A"] * len(ret)),
        ret=ret,
        roll=roll,
        level=np.full(len(ret), level),
    )


def test_daily_position_matches_rebalance_and_gross():
    series = make_series([0.0, 0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.0, -0.03, 0.01, 0.02, -0.01])
    position = cp.position_schedule(series, lookback=3, step=1)
    rebalanced = cm.daily_rebalanced(series, 3)
    assert np.array_equal(position, rebalanced["position"].to_numpy())
    frame, _ = cp.simulate_hold(series, lookback=3, step=1, liquidate=False)
    assert frame["gross"].sum() == pytest.approx(rebalanced["gross"].sum())


def test_block_net_matches_trade_sum_including_rolls():
    ret = [0.0, 0.01, 0.01, 0.02, -0.01, 0.03, 0.01, 0.01, -0.02, 0.01, 0.02, -0.04, 0.01, 0.0, 0.02]
    series = make_series(ret, roll_at=(5, 8))
    frame, trips = cp.simulate_block(series, lookback=2, step=3)
    trades = cm.trades(series, 2, 3)
    assert frame["net"].sum() == pytest.approx(trades["net"].sum())
    assert frame["gross"].sum() == pytest.approx(trades["gross"].sum())
    assert trips["net"].sum() == pytest.approx(trades["net"].sum())
    assert len(trips) == len(trades)


def test_hold_charges_one_round_trip_when_the_sign_does_not_change():
    ret = [0.0, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.0, 0.0]
    series = make_series(ret)
    hold, hold_trips = cp.simulate_hold(series, lookback=2, step=3)
    block, _ = cp.simulate_block(series, lookback=2, step=3)
    # Два окна одного знака: удержание платит вход и выход, блок — два полных круга.
    assert hold["cost"].sum() == pytest.approx(core.round_trip_cost())
    assert block["cost"].sum() == pytest.approx(2 * core.round_trip_cost())
    assert hold["gross"].sum() == pytest.approx(block["gross"].sum())
    assert hold["net"].sum() == pytest.approx(hold["gross"].sum() - core.round_trip_cost())
    assert hold_trips["net"].sum() == pytest.approx(hold["net"].sum())
    assert len(hold_trips) == 1


def test_roll_with_the_same_sign_costs_two_sides_and_a_flip_on_roll_is_not_four():
    ret = [0.0, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.0, 0.0]
    stayed = cp.simulate_hold(make_series(ret, roll_at=(5,)), lookback=2, step=3)[0]
    # Вход 1 сторона, роллировка внутри позиции 2 стороны, выход 1 сторона.
    assert stayed["cost"].sum() == pytest.approx(4 * cp.side_cost())
    flipped_ret = [0.0, 0.01, 0.02, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.0]
    plain = cp.simulate_hold(make_series(flipped_ret), lookback=2, step=3)[0]
    on_roll = cp.simulate_hold(make_series(flipped_ret, roll_at=(6,)), lookback=2, step=3)[0]
    assert plain["position"][6] == -1
    assert plain["position"][5] == 1
    assert on_roll["cost"].sum() == pytest.approx(plain["cost"].sum())
    assert float(on_roll["cost"].iloc[6]) == pytest.approx(2 * cp.side_cost())


def test_opening_on_a_roll_from_flat_does_not_close_a_position_that_was_not_there():
    ret = [0.0, 0.0, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.0, 0.0]
    frame, _ = cp.simulate_hold(make_series(ret, roll_at=(3,)), lookback=2, step=3)
    assert frame["position"][3] == 1
    assert float(frame["cost"].iloc[3]) == pytest.approx(cp.side_cost())


def test_threshold_drops_a_weak_move_and_zero_threshold_matches_the_sign():
    ret = np.array([0.0, 0.001, 0.001, 0.001, 0.08, 0.08, 0.08, 0.08, 0.0, 0.0])
    series = make_series(ret, level=10.0)
    # Ход за 2 дня на дне 2 — 2 б.п.; на дне 6 — 160 б.п. при цене 10.
    assert cp.move_bps(series, 2)[2] == pytest.approx(2.0)
    assert cp.move_bps(series, 2)[6] == pytest.approx(160.0)
    assert cp.signal_of(series, 2, 0)[2] == np.sign(series.price[2] - series.price[0])
    assert cp.signal_of(series, 2, 50)[2] == 0
    assert cp.signal_of(series, 2, 50)[6] == 1
    position = cp.position_schedule(series, 2, 3, min_move_bp=50)
    assert position[3:6].sum() == 0
    assert position[6:9].sum() == 3


def test_extra_slippage_scales_with_the_number_of_sides():
    series = make_series([0.0, 0.01, 0.02, 0.01, 0.01, -0.03, -0.02, -0.01, 0.0, 0.0, 0.01, 0.0])
    base, _ = cp.simulate_hold(series, lookback=2, step=3)
    worse, _ = cp.simulate_hold(series, lookback=2, step=3, extra_side_cost=4)
    sides = base["cost"].sum() / cp.side_cost()
    assert worse["net"].sum() == pytest.approx(base["net"].sum() - 4 * sides)
    assert worse["gross"].sum() == pytest.approx(base["gross"].sum())


def test_final_open_position_is_liquidated_once():
    ret = [0.0, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    closed, trips = cp.simulate_hold(make_series(ret), lookback=2, step=3, liquidate=True)
    marked, _ = cp.simulate_hold(make_series(ret), lookback=2, step=3, liquidate=False)
    assert closed["position"].iloc[-1] != 0
    assert closed["cost"].sum() == pytest.approx(marked["cost"].sum() + cp.side_cost())
    assert trips["net"].sum() == pytest.approx(closed["net"].sum())


def test_financials_and_walk_forward_on_a_persistent_trend():
    rng = np.random.default_rng(3)
    regimes = np.repeat(rng.choice([-1.0, 1.0], size=40), 20)[:700]
    ret = regimes * 0.02 + rng.normal(0, 0.01, 700)
    series = make_series(np.r_[0.0, ret])
    frame, trips = cp.simulate_hold(series, lookback=10, step=5)
    stats = cp.financials(frame, trips, base=12_000)
    assert stats["net"] > 0
    assert stats["sharpe"] > 0
    assert stats["trades"] == len(trips)
    assert stats["long_net"] + stats["short_net"] == pytest.approx(stats["net"])
    years, total, book = cp.walk_forward(
        series,
        {(lookback, step): cp.simulate_hold(series, lookback=lookback, step=step)[0] for lookback in (5, 10) for step in (5, 10)},
        12_000,
        first_test_year=2024,
    )
    assert not years.empty
    assert years["L"].isin((5, 10)).all()
    assert book["day"].dt.year.min() >= 2024
    assert total["net"] == pytest.approx(book["net"].sum())


def test_report_contains_both_cost_models():
    series = make_series(np.r_[0.0, np.resize([0.01, 0.02, -0.01, 0.03, -0.02, 0.01], 500)], start="2022-04-21")
    study = cp.run_study(series, reps=20, first_test_year=2024)
    text = cp.build_report(series, study)
    assert "Удержание: деньги и риск" in text
    assert "Блок: деньги и риск" in text
    assert "p = " in text
    hold = cp._cell(study.rows, "hold", 20, 5)
    block = cp._cell(study.rows, "block", 20, 5)
    assert hold["gross"] == pytest.approx(block["gross"])
