from datetime import date
from pathlib import Path

import pandas as pd

from cnyrub.bars import cache_bounds

ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data" / "bars"

EXPECTED = [
    "CRM2",
    "CRU2",
    "CRZ2",
    "CRH3",
    "CRM3",
    "CRU3",
    "CRZ3",
    "CRH4",
    "CRM4",
    "CRU4",
    "CRZ4",
    "CRH5",
    "CRM5",
    "CRU5",
    "CRZ5",
    "CRH6",
    "CRM6",
    "CRU6",
    "CRZ6",
]


def test_histories_stay_in_separate_files_without_a_stitched_series():
    assert (ROOT / "data" / "continuous" / "cny_front_1m.parquet").exists() is False
    assert (ROOT / "data" / "manifest.json").exists() is False
    assert sorted(path.stem for path in BARS.glob("*.parquet")) == sorted(EXPECTED)
    assert cache_bounds(BARS / "CRM2.parquet") == (date(2022, 4, 21), date(2022, 6, 16))
    assert cache_bounds(BARS / "CRU2.parquet")[0] == date(2022, 5, 16)
    assert cache_bounds(BARS / "CRZ6.parquet") == (date(2026, 8, 17), date(2026, 9, 21))
    first = pd.read_parquet(BARS / "CRM2.parquet", columns=["datetime"]).iloc[0, 0]
    last = pd.read_parquet(BARS / "CRZ6.parquet", columns=["datetime"]).iloc[-1, 0]
    assert pd.Timestamp(first) == pd.Timestamp("2022-04-21 10:06:00")
    assert pd.Timestamp(last) == pd.Timestamp("2026-09-21 23:49:00")
