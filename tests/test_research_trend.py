import pytest

from research import core, families
from tests.test_research import ar1, synthetic


def test_trend_grid_has_unique_labels_and_only_follows_signals():
    from research import trend

    plan = trend.specs()
    labels = [spec.label() for spec in plan]
    assert len(labels) == len(set(labels)) > 50
    assert {spec.sign for spec in plan} == {1}
    assert {spec.family for spec in plan} == set(families.FAMILIES)


def test_trend_conclusions_are_filled_in():
    from research import trend

    assert "Вердикт" in trend.CONCLUSIONS
