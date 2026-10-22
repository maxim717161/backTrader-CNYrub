import numpy as np
import pandas as pd
import pytest

from research import core, cr_momentum as cm


@pytest.fixture(autouse=True)
def restore_instrument():
    saved = (core.MULTIPLIER, core.TICK, core.UNIT)
    core.configure(1000.0, 0.001)
    yield
    core.MULTIPLIER, core.TICK, core.UNIT = saved


def make_series(days=300, drift=0.01, noise=0.02, block=20, seed=3, roll_every=60):
    rng = np.random.default_rng(seed)
    regimes = np.repeat(rng.choice([-1, 1], size=days // block + 1), block)[:days]
    ret = regimes * drift + rng.normal(0, noise, days)
    index = pd.bdate_range("2023-01-02", periods=days)
    roll = np.zeros(days, dtype=bool)
    roll[roll_every::roll_every] = True
    return cm.Series(
        days=index.to_numpy().astype("datetime64[ns]"),
        contract=np.array(["A"] * days),
        ret=ret,
        roll=roll,
        level=12.0 + np.cumsum(ret),
    )


def test_signal_is_zero_until_history_and_follows_price_change():
    series = make_series()
    sig = cm.signal(series, 10)
    assert (sig[:10] == 0).all()
    price = series.price
    assert sig[10] == np.sign(price[10] - price[0])


def test_trades_do_not_overlap_and_charge_cost_per_roll():
    series = make_series()
    frame = cm.trades(series, 10, 5)
    assert frame["day"].diff().dropna().dt.days.min() >= 7
    cost = core.round_trip_cost()
    assert ((frame["gross"] - frame["net"]) / cost - 1).round(9).eq(frame["rolls"]).all()
    assert frame["rolls"].sum() > 0


def test_momentum_follows_trend_and_mirror_loses():
    series = make_series()
    follow = cm.trades(series, 10, 5)
    fade = cm.trades(series, 10, 5, sign=-1)
    assert follow["gross"].mean() > 0 > fade["gross"].mean()
    assert follow["gross"].mean() == pytest.approx(-fade["gross"].mean())


def test_extra_slippage_lowers_net_by_two_sides_per_trade():
    series = make_series()
    base = cm.trades(series, 10, 5)
    worse = cm.trades(series, 10, 5, extra_side_cost=3.0)
    assert (base["net"] - worse["net"]).round(9).eq(6.0 * (1 + base["rolls"])).all()


def test_delay_moves_entry_day():
    series = make_series()
    now = cm.trades(series, 10, 5)
    later = cm.trades(series, 10, 5, delay=1)
    assert (later["day"].iloc[0] - now["day"].iloc[0]).days >= 1


def test_daily_rebalance_costs_only_on_position_changes():
    series = make_series(roll_every=1000)
    frame = cm.daily_rebalanced(series, 10)
    changes = np.abs(np.diff(np.r_[0.0, frame["position"].to_numpy()]))
    assert frame["cost"].sum() == pytest.approx(changes.sum() * cm.SIDE_COST)
    assert (frame["gross"] - frame["cost"]).round(9).equals(frame["net"].round(9))


def test_shift_test_detects_persistent_trend_but_not_noise():
    trend = cm.shift_test(make_series(days=600), 10, reps=500)
    assert trend["p"] < 0.05
    assert trend["наблюдаемый"] > trend["нуль, 95-й процентиль"]
    noise = cm.shift_test(make_series(days=600, drift=0.0), 10, reps=500)
    assert noise["p"] > 0.05


def test_notional_uses_median_level():
    series = make_series()
    assert cm.notional(series) == pytest.approx(np.median(series.level) * 1000.0)


def test_grid_and_walk_forward_run_on_synthetic_series():
    series = make_series(days=900, block=20)
    table, white = cm.grid(series)
    assert len(table) == len(cm.LOOKBACKS) * len(cm.HOLDS)
    assert 0 <= white <= 1
    walk, total = cm.walk_forward(series, first_test_year=2024)
    assert not walk.empty
    assert total["сделок"] > 0
    variants = cm.variants_table(series)
    assert variants.loc["правило", "net, б.п."] > variants.loc["зеркало (против сигнала)", "net, б.п."]
