from datetime import date

import pandas as pd

from cnyrub.manifest import build_manifest, stitch


def bars(secid: str, stamps: list[str], prices: list[float]) -> pd.DataFrame:
    opened = pd.to_datetime(stamps)
    return pd.DataFrame(
        {
            "datetime": opened,
            "end": opened + pd.Timedelta(seconds=59),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1] * len(prices),
            "value": [0] * len(prices),
            "secid": secid,
        }
    )


def test_stitch_keeps_raw_prices_and_later_contract_on_the_same_minute():
    first = bars("CRM2", ["2022-06-16 18:40:00"], [10.0])
    second = bars("CRU2", ["2022-06-16 18:40:00", "2022-06-17 10:00:00"], [10.7, 10.8])
    combined, dropped = stitch([first, second])
    assert dropped == 1
    assert combined["secid"].tolist() == ["CRU2", "CRU2"]
    assert combined["open"].tolist() == [10.7, 10.8]
    assert list(combined.columns) == [
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


def test_manifest_reports_rows_volume_gap_and_roll():
    frame = bars(
        "CRZ2",
        ["2022-12-30 23:50:00"],
        [10.0],
    )
    nxt = bars("CRH3", ["2023-01-04 10:00:00"], [10.4])
    combined, _dropped = stitch([frame, nxt])
    combined.loc[combined.index[0], "close"] = 10.1
    manifest = build_manifest(
        combined,
        as_of=date(2023, 1, 4),
        expected_secids=["CRZ2", "CRH3"],
    )
    assert manifest["rows"] == 2
    assert manifest["volume"] == 2
    assert manifest["first"] == "2022-12-30 23:50:00"
    assert manifest["last"] == "2023-01-04 10:00:00"
    warnings = "\n".join(manifest["warnings"])  # type: ignore[arg-type]
    assert "value равно нулю" in warnings
    assert "5 суток" in warnings
    assert "Стык CRZ2/CRH3: 10.100 -> 10.400 (+0.300)." in warnings
