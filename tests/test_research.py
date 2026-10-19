import numpy as np
import pandas as pd
import pytest

from research import battery, core, evaluate, families
from research.families import Spec

MINUTES = 400


def synthetic(returns_by_day, start_price=12.0, contract="CRX9", tick_round=False) -> core.Market:
    frames = []
    price = start_price
    day = pd.Timestamp("2024-01-08")
    for returns in returns_by_day:
        day += pd.Timedelta(days=1)
        while day.dayofweek >= 5:
            day += pd.Timedelta(days=1)
        stamp = pd.date_range(day + pd.Timedelta(hours=10), periods=len(returns), freq="min")
        close = price * np.exp(np.cumsum(returns))
        if tick_round:
            close = np.round(close, 3)
        opens = np.r_[close[0], close[:-1]]
        frames.append(
            pd.DataFrame(
                {
                    "datetime": stamp,
                    "open": opens,
                    "high": np.maximum(opens, close) + 0.001,
                    "low": np.minimum(opens, close) - 0.001,
                    "close": close,
                    "volume": 10,
                    "contract": contract,
                    "day": day,
                }
            )
        )
        price = close[-1]
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.set_index("datetime", drop=False)
    return core.Market(bars=bars, grid=core._build_grid(bars))


def ar1(phi, days=60, seed=1, sigma=0.0004):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(days):
        shocks = rng.normal(0, sigma, MINUTES)
        r = np.zeros(MINUTES)
        for i in range(1, MINUTES):
            r[i] = phi * r[i - 1] + shocks[i]
        out.append(r)
    return out


def test_random_walk_has_variance_ratio_near_one():
    market = synthetic(ar1(0.0))
    vr, low, high = core.variance_ratio(market.grid, 5, 6)
    assert 0.9 < vr < 1.1
    assert low < 1 < high
    rho, *_ = core.lag_autocorr(core.clock_returns(market.grid, 15), 1)
    assert abs(rho) < 0.1


def test_trending_series_has_vr_above_one_and_positive_autocorrelation():
    market = synthetic(ar1(0.35))
    vr, low, _ = core.variance_ratio(market.grid, 5, 6)
    assert vr > 1.1 and low > 1
    rho, *_ = core.lag_autocorr(core.clock_returns(market.grid, 1), 1)
    assert rho > 0.2


def test_mean_reverting_series_has_vr_below_one_and_negative_autocorrelation():
    market = synthetic(ar1(-0.35))
    vr, _, high = core.variance_ratio(market.grid, 5, 6)
    assert vr < 0.9 and high < 1
    rho, *_ = core.lag_autocorr(core.clock_returns(market.grid, 1), 1)
    assert rho < -0.2


def test_noise_adjustment_removes_bid_ask_bounce_but_keeps_real_reversion():
    rng = np.random.default_rng(3)
    days = []
    for _ in range(60):
        true = np.cumsum(rng.normal(0, 0.0004, MINUTES))
        observed = true + rng.normal(0, 0.0004, MINUTES)
        days.append(np.diff(observed, prepend=0.0))
    bounce = synthetic(days)
    raw, *_ = core.variance_ratio(bounce.grid, 5, 6)
    adjusted, *_ = core.noise_adjusted_vr(bounce.grid, 5, 6, reps=50)
    assert raw < 0.9
    assert 0.9 < adjusted < 1.1
    assert abs(adjusted - 1) < abs(raw - 1)


def test_sign_persistence_is_half_for_random_walk():
    market = synthetic(ar1(0.0))
    share, se, n = core.sign_persistence(core.clock_returns(market.grid, 5))
    assert abs(share - 0.5) < 3 * se + 0.01 and n > 1000


def test_hurst_orders_the_three_regimes():
    trending = core.hurst(synthetic(ar1(0.35)).grid)
    walk = core.hurst(synthetic(ar1(0.0)).grid)
    reverting = core.hurst(synthetic(ar1(-0.35)).grid)
    assert reverting < walk < trending


def flat_data(days=3, price=12.0):
    market = synthetic([np.zeros(MINUTES) for _ in range(days)], start_price=price)
    return market, core.arrays(market)


def test_round_trip_cost_is_commission_plus_slippage():
    assert core.round_trip_cost() == pytest.approx(2 * core.COMMISSION + 2 * core.SLIPPAGE_TICKS * core.TICK * core.MULTIPLIER)


def test_flat_trade_loses_exactly_the_round_trip_cost():
    _, data = flat_data()
    trades = core.run_entries(data, [core.Entry(index=10, side=1, hold=30)])
    assert len(trades) == 1
    assert trades.iloc[0]["gross"] == pytest.approx(0.0, abs=1e-9)
    assert trades.iloc[0]["net"] == pytest.approx(-core.round_trip_cost())
    assert trades.iloc[0]["reason"] == "time"
    assert trades.iloc[0]["bars"] == 30


def test_stop_is_hit_on_adverse_move_and_priced_at_stop():
    drop = np.zeros(MINUTES)
    drop[50:] = -0.0005
    market = synthetic([drop])
    data = core.arrays(market)
    entry = core.Entry(index=10, side=1, stop=0.01)
    trades = core.run_entries(data, [entry])
    assert trades.iloc[0]["reason"] == "stop"
    assert trades.iloc[0]["gross"] == pytest.approx(-0.01 * core.MULTIPLIER)


def test_target_wins_for_short_when_price_falls():
    fall = np.zeros(MINUTES)
    fall[50:] = -0.0004
    market = synthetic([fall])
    data = core.arrays(market)
    trades = core.run_entries(data, [core.Entry(index=10, side=-1, target=0.02, stop=0.5)])
    assert trades.iloc[0]["reason"] == "target"
    assert trades.iloc[0]["gross"] == pytest.approx(0.02 * core.MULTIPLIER)


def test_trades_do_not_overlap():
    _, data = flat_data()
    entries = [core.Entry(index=i, side=1, hold=50) for i in (10, 20, 30, 70)]
    trades = core.run_entries(data, entries)
    assert len(trades) == 2


def test_day_is_flattened_unless_overnight_allowed():
    _, data = flat_data(days=2)
    late = [core.Entry(index=MINUTES - 1, side=1)]
    assert core.run_entries(data, late, flatten_day=True).empty
    kept = core.run_entries(data, late, flatten_day=False)
    assert len(kept) == 1 and kept.iloc[0]["reason"] == "end"


def test_breakout_sign_flips_the_side():
    market = synthetic(ar1(0.0, days=10))
    data = core.arrays(market)
    follow = core.breakout_entries(data, 30, side_sign=1)
    fade = core.breakout_entries(data, 30, side_sign=-1)
    assert len(follow) == len(fade) > 0
    assert all(a.side == -b.side and a.index == b.index for a, b in zip(follow, fade))


def test_forward_moves_follow_the_signal_direction():
    up = np.full(MINUTES, 0.0003)
    market = synthetic([up])
    moves = core.forward_moves(market, np.array([10]), np.array([1]), (5,))
    expected = (np.exp(0.0003 * 5) - 1) * market.grid["close"].iloc[11] * core.MULTIPLIER
    assert moves["m5"].iloc[0] == pytest.approx(expected, rel=0.02)
    short = core.forward_moves(market, np.array([10]), np.array([-1]), (5,))
    assert short["m5"].iloc[0] == pytest.approx(-moves["m5"].iloc[0])


def test_forward_moves_drop_events_leaving_the_day():
    market = synthetic([np.zeros(MINUTES)])
    moves = core.forward_moves(market, np.array([MINUTES - 2]), np.array([1]), (30,))
    assert np.isnan(moves["m30"].iloc[0])


def test_reality_check_does_not_reward_noise():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, size=(250, 40))
    p, _ = core.reality_check(noise, reps=300)
    assert p > 0.05


def test_reality_check_sees_a_real_edge():
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 1, size=(250, 20))
    noise[:, 3] += 0.5
    p, _ = core.reality_check(noise, reps=300)
    assert p < 0.05


def test_cluster_mean_uses_days_as_units():
    values = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    days = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    _, se_clustered, _, _ = core.cluster_mean(values, days)
    _, se_independent, _, _ = core.cluster_mean(values, np.arange(8))
    assert se_clustered > se_independent


def test_subset_keeps_only_requested_days():
    market = synthetic(ar1(0.0, days=6))
    days = sorted(market.bars["day"].unique())[:2]
    part = battery.subset(market, days)
    assert set(part.bars["day"]) == set(days) and set(part.grid["day"]) == set(days)


def test_gap_family_enters_at_first_bar_in_direction_of_gap_with_sign():
    first = np.zeros(MINUTES)
    second = np.zeros(MINUTES)
    second[0] = 0.01
    market = synthetic([first, second])
    data = core.arrays(market)
    follow = families.build(Spec("gap", 1, dict(min_gap=5, hold=30)), market, data)
    fade = families.build(Spec("gap", -1, dict(min_gap=5, hold=30)), market, data)
    assert [(e.side, e.index) for e in follow] == [(1, MINUTES)]
    assert [(e.side, e.index) for e in fade] == [(-1, MINUTES)]


def test_firsthour_family_follows_first_hour_move():
    up = np.zeros(MINUTES)
    up[:60] = 0.0004
    market = synthetic([up])
    data = core.arrays(market)
    entries = families.build(Spec("firsthour", 1, dict(window=60, min_move=10)), market, data)
    assert len(entries) == 1 and entries[0].side == 1 and entries[0].index == 60


def test_scalp_family_signals_each_minute_after_a_move_and_flips_with_sign():
    jumpy = np.zeros(MINUTES)
    jumpy[100] = 0.001
    market = synthetic([jumpy])
    data = core.arrays(market)
    follow = families.build(Spec("scalp", 1, dict(min_move=5, hold=3)), market, data)
    fade = families.build(Spec("scalp", -1, dict(min_move=5, hold=3)), market, data)
    assert [(e.index, e.side) for e in follow] == [(100, 1)]
    assert [(e.index, e.side) for e in fade] == [(100, -1)]


def test_spec_labels_are_unique_and_show_sign_and_overnight():
    a = Spec("donchian", 1, dict(channel=30), True).label()
    b = Spec("donchian", -1, dict(channel=30), True).label()
    c = Spec("donchian", 1, dict(channel=30), False).label()
    assert len({a, b, c}) == 3 and "[+]" in a and "[-]" in b and "ночь" in c


def test_grid_selection_runs_and_flipped_rule_mirrors_gross_on_synthetic_trend():
    market = synthetic(ar1(0.0, days=80, seed=9))
    data = core.arrays(market)
    plan = [
        Spec("donchian", 1, dict(channel=30, stop=4, min_stop=0.008)),
        Spec("donchian", -1, dict(channel=30, stop=4, min_stop=0.008)),
        Spec("momentum", 1, dict(lookback=30, every=60, hold=60, min_move=5)),
    ]
    result = evaluate.run_grid(plan, market, data)
    assert list(result.table.index) == [spec.label() for spec in plan]
    assert result.daily.shape == (len(result.days), len(plan))
    assert (result.table["всё: net"] < result.table["всё: gross"]).all()
    detail = evaluate.detail(plan[0], market, data)
    assert set(detail["sides"].index) == {"покупки", "продажи"}


def test_table_formats_intervals_and_event_tuples():
    frame = pd.DataFrame({"a": [(0.1, -0.2, 0.3)], "b": [(1.5, 2.0)]}, index=["x"])
    text = battery.table(frame)
    assert "+0.100 [-0.200; +0.300]" in text and "+1.5 (t=+2.0)" in text


def test_stop_is_filled_at_the_open_when_price_gaps_through_it():
    jump = np.zeros(MINUTES)
    jump[50] = -0.05
    market = synthetic([jump])
    data = core.arrays(market)
    data["o"] = data["o"].copy()
    data["o"][50] = data["c"][50]
    entry = core.Entry(index=10, side=1, stop=0.01)
    trades = core.run_entries(data, [entry])
    assert trades.iloc[0]["reason"] == "stop"
    entry_price = data["c"][11]
    assert trades.iloc[0]["gross"] < -0.01 * core.MULTIPLIER
    assert trades.iloc[0]["gross"] == pytest.approx((data["o"][50] - entry_price) * core.MULTIPLIER)


def test_short_stop_is_filled_at_the_open_when_price_gaps_up():
    jump = np.zeros(MINUTES)
    jump[50] = 0.05
    market = synthetic([jump])
    data = core.arrays(market)
    data["o"] = data["o"].copy()
    data["o"][50] = data["c"][50]
    trades = core.run_entries(data, [core.Entry(index=10, side=-1, stop=0.01)])
    assert trades.iloc[0]["gross"] == pytest.approx(-(data["o"][50] - data["c"][11]) * core.MULTIPLIER)
