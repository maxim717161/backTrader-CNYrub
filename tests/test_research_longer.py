from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research import core, fetch, instruments, longer
from tests.test_research import ar1, synthetic


@pytest.fixture(autouse=True)
def restore_instrument():
    saved = (core.MULTIPLIER, core.TICK, core.UNIT)
    yield
    core.MULTIPLIER, core.TICK, core.UNIT = saved


def test_expiry_is_third_thursday():
    assert fetch.third_thursday(2022, 6) == date(2022, 6, 16)
    assert fetch.third_thursday(2026, 3) == date(2026, 3, 19)
    assert fetch.third_thursday(2019, 3).weekday() == 3


def test_windows_are_contiguous_and_start_after_previous_expiry():
    windows = fetch.windows("Si", date(2026, 10, 19))
    assert windows[0].secid == "SiH9" and windows[0].start == date(2018, 12, 21)
    for before, after in zip(windows[:-1], windows[1:]):
        assert after.start == before.end + timedelta(days=1)
    assert windows[-1].end == date(2026, 10, 19) and windows[-1].secid == "SiZ6"
    assert len({w.name for w in windows}) == len(windows)


def test_partial_last_window_is_not_treated_as_complete(tmp_path):
    today = date(2026, 10, 19)
    last = fetch.windows("Eu", today)[-1]
    path = fetch.path_for(last, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert not fetch.complete(last, tmp_path, today)
    older = fetch.windows("Eu", today)[0]
    fetch.path_for(older, tmp_path).write_bytes(b"x")
    assert fetch.complete(older, tmp_path, today)


def test_comparison_unit_keeps_same_share_of_price_for_each_instrument():
    cr = instruments.INSTRUMENTS["CR"]
    si = instruments.INSTRUMENTS["Si"]
    assert instruments.comparison_unit(cr, 12.5) == pytest.approx(0.001)
    unit = instruments.comparison_unit(si, 80000.0)
    assert unit == pytest.approx(6.0)
    assert unit / 80000.0 == pytest.approx(instruments.REL_UNIT, rel=0.1)


def test_round_trip_cost_is_equal_in_rubles_for_all_three_instruments():
    costs = []
    for name in ("CR", "Si", "Eu"):
        ins = instruments.INSTRUMENTS[name]
        core.configure(ins.multiplier, ins.tick)
        costs.append(core.round_trip_cost())
    assert costs == [12.0, 12.0, 12.0]


def test_configure_scales_trades_and_thresholds():
    core.configure(1.0, 1.0, 6.0)
    market = synthetic([np.zeros(400), np.r_[np.zeros(100), np.full(300, 0.001)]], start_price=80000.0)
    data = core.arrays(market)
    trades = core.run_entries(data, [core.Entry(index=10, side=1, hold=30)])
    assert trades.iloc[0]["net"] == pytest.approx(-12.0)
    from research.families import Spec, build

    entries = build(Spec("gap", 1, dict(min_gap=20, hold=30)), market, data)
    assert entries == []
    big_gap = synthetic([np.zeros(400), np.r_[0.002, np.zeros(399)]], start_price=80000.0)
    assert build(Spec("gap", 1, dict(min_gap=20, hold=30)), big_gap, core.arrays(big_gap))


def test_gap_periods_split_at_cr_launch():
    core.configure(1000.0, 0.001)
    market = synthetic(ar1(0.0, days=30, seed=2), tick_round=True)
    table = longer.gap_periods(market, core.arrays(market))
    assert set(table["период"]) == {"до 21.04.2022", "с 21.04.2022", "весь период"}
    assert len(table) == 27


def test_loading_missing_data_explains_how_to_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "OTHER_DIR", tmp_path)
    monkeypatch.setattr(instruments.fetch, "path_for", lambda w, directory=tmp_path: directory / "no.parquet")
    with pytest.raises(FileNotFoundError, match="research.fetch"):
        instruments.load("Si")


def test_describe_reports_cost_in_basis_points():
    core.configure(1.0, 1.0, 6.0)
    market = synthetic([np.zeros(400) for _ in range(5)], start_price=80000.0)
    info = instruments.describe(market, instruments.INSTRUMENTS["Si"])
    assert info["круг, ₽"] == 12.0
    assert info["круг, б.п."] == pytest.approx(12.0 / 80000.0 * 1e4)
