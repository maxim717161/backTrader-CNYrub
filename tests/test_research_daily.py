import numpy as np
import pandas as pd
import pytest

from research import core, daily
from tests.test_research import synthetic


def trending_market(days=120, drift=0.00003, noise=0.0004, seed=7, block=20):
    """Дни со случайным знаком дрейфа, держащимся block дней: моментум по знаку 20-дневной доходности должен зарабатывать."""
    rng = np.random.default_rng(seed)
    regimes = np.repeat(rng.choice([-1, 1], size=days // block + 1), block)[:days]
    returns = [np.full(400, regimes[i] * drift) + rng.normal(0, noise, 400) for i in range(days)]
    return synthetic(returns)


@pytest.fixture(autouse=True)
def restore_instrument():
    saved = (core.MULTIPLIER, core.TICK, core.UNIT)
    yield
    core.MULTIPLIER, core.TICK, core.UNIT = saved


def test_daily_closes_take_last_close_of_each_day():
    market = synthetic([np.full(400, 0.0001), np.full(400, 0.0001)])
    closes = daily.daily_closes(market)
    assert len(closes) == 2
    assert closes["close"].iloc[0] == pytest.approx(market.bars["close"].iloc[399])


def test_momentum_trades_do_not_overlap_and_follow_the_sign():
    market = trending_market()
    closes = daily.daily_closes(market)
    trades = daily.tsmom(closes, 5, 5)
    assert trades["day"].is_monotonic_increasing
    assert trades["day"].diff().dropna().dt.days.min() >= 5
    assert set(trades["side"]) <= {-1, 1}


def test_momentum_earns_on_persistent_trend_and_mirror_loses():
    market = trending_market()
    closes = daily.daily_closes(market)
    follow = daily.tsmom(closes, 5, 5)
    fade = daily.tsmom(closes, 5, 5, sign=-1)
    assert follow["net"].mean() > 0 > fade["net"].mean()
    assert follow["gross"].mean() == pytest.approx(-fade["gross"].mean())


def test_costs_are_subtracted_once_per_trade():
    market = trending_market()
    trades = daily.tsmom(daily.daily_closes(market), 5, 5)
    assert (trades["gross"] - trades["net"]).round(9).eq(core.round_trip_cost()).all()


def test_fixed_side_ignores_signal_and_delay_shifts_entry_day():
    market = trending_market()
    closes = daily.daily_closes(market)
    long_only = daily.tsmom(closes, 5, 5, fixed_side=1)
    assert set(long_only["side"]) == {1}
    now = daily.tsmom(closes, 5, 5)
    later = daily.tsmom(closes, 5, 5, delay=1)
    assert later["day"].iloc[0] > now["day"].iloc[0]


def test_grid_and_robustness_run_and_report_every_cell():
    core.configure(1000.0, 0.001)
    market = trending_market(days=200)
    table, p = daily.grid(market, pd.Timestamp("2024-03-01"))
    assert len(table) == len(daily.LOOKBACKS) * len(daily.HOLDS)
    assert 0 < p <= 1
    robust = daily.robustness(market)
    assert list(robust.index) == list(daily.HOLDS)
    assert "лет в плюсе" in robust.columns
    assert len(daily.by_year(market)) >= 1
    assert len(daily.drift_controls(market)) == 2 * len(daily.HOLDS)


def test_momentum_with_trend_is_significant_by_reality_check():
    core.configure(1000.0, 0.001)
    market = trending_market(days=300, block=40)
    _, p = daily.grid(market, pd.Timestamp("2030-01-01"))
    assert p < 0.05
