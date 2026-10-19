import numpy as np
import pytest

from research import core, families
from tests.test_research import ar1, synthetic


def test_countertrend_grid_has_unique_labels_and_only_fades_signals():
    from research import countertrend

    plan = countertrend.specs()
    labels = [spec.label() for spec in plan]
    assert len(labels) == len(set(labels)) > 50
    assert {spec.sign for spec in plan} == {-1}
    assert {spec.family for spec in plan} == set(families.FAMILIES)


def test_countertrend_targets_exceed_round_trip_cost():
    from research import countertrend

    assert min(countertrend.min_distances()) * core.MULTIPLIER > core.round_trip_cost()


def test_bounce_table_and_gap_table_run_on_synthetic_data():
    from research import countertrend

    market = synthetic(ar1(-0.3, days=40, seed=4), tick_round=True)
    data = core.arrays(market)
    bounce = countertrend.bounce_table(market, data)
    assert len(bounce) == 6
    gaps, years = countertrend.gap_table(market, data)
    assert list(gaps.index) == [5, 10, 20, 30, 50]


def test_countertrend_conclusions_are_filled_in():
    from research import countertrend

    assert "Вердикт" in countertrend.CONCLUSIONS
