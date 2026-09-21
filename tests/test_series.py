import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "data" / "continuous" / "cny_front_1m.parquet"
MANIFEST = ROOT / "data" / "manifest.json"


def test_published_series_matches_the_21_september_2026_snapshot():
    frame = pd.read_parquet(SERIES)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert list(frame.columns) == [
        "datetime",
        "end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "secid",
    ]
    assert len(frame) == 907_606
    assert manifest["rows"] == 907_606
    assert manifest["as_of"] == "2026-09-21"
    assert manifest["first"] == "2022-04-21 10:06:00"
    assert manifest["last"] == "2026-09-21 23:49:00"
    assert frame["datetime"].is_monotonic_increasing
    assert frame["datetime"].dt.tz is None
    assert frame["value"].eq(0).all()
    assert frame["secid"].nunique() == 19
    assert frame["secid"].iloc[0] == "CRM2"
    assert frame["secid"].iloc[-1] == "CRZ6"
    assert int(frame["volume"].sum()) == manifest["volume"]
    assert sum(window["rows"] for window in manifest["windows"]) == 907_606
