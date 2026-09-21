from datetime import date

import pandas as pd

from cnyrub.bars import prepare_bars
from cnyrub.cli import build_parser, main
from cnyrub.contracts import Contract
from cnyrub.download import download_front


def contract(secid: str, shortname: str, frst: str, lst: str) -> Contract:
    first = date.fromisoformat(frst)
    last = date.fromisoformat(lst)
    return Contract(secid, shortname, "CNY", first, last, last, 1000)


def candle(secid: str, stamp: str, price: float) -> pd.DataFrame:
    opened = pd.Timestamp(stamp)
    return pd.DataFrame(
        {
            "begin": [stamp],
            "end": [opened + pd.Timedelta(seconds=59)],
            "open": [price],
            "high": [price],
            "low": [price],
            "close": [price],
            "volume": [4],
            "value": [0],
        }
    )


def test_prepare_bars_uses_exchange_clock_and_drops_the_day_outside_the_window():
    frame = pd.DataFrame(
        {
            "begin": ["2022-06-16 18:40:00", "2022-06-17 10:00:00"],
            "end": ["2022-06-16 18:40:59", "2022-06-17 10:00:59"],
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": [1, 1],
            "value": [0, 0],
        }
    )
    prepared = prepare_bars(frame, "CRM2", date(2022, 4, 21), date(2022, 6, 16))
    assert prepared["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == ["2022-06-16 18:40:00"]
    assert prepared["secid"].tolist() == ["CRM2"]
    assert pd.api.types.is_datetime64_any_dtype(prepared["datetime"])
    assert prepared["datetime"].dt.tz is None


def test_download_caches_closed_contracts_and_refetches_the_open_one(tmp_path):
    contracts = [
        contract("CRM2", "CNY-6.22", "2022-04-21", "2022-06-16"),
        contract("CRU2", "CNY-9.22", "2022-04-21", "2022-09-15"),
        contract("CRZ2", "CNY-12.22", "2022-06-02", "2022-12-15"),
    ]
    today = date(2022, 7, 1)
    calls: list[str] = []

    def fetch(item: Contract, start: date, end: date) -> pd.DataFrame:
        calls.append(item.secid)
        price = 10.0 if item.secid == "CRM2" else 10.7
        return candle(item.secid, f"{start.isoformat()} 10:00:00", price)

    frame, manifest = download_front(contracts, today, tmp_path, fetch, workers=1)
    assert calls == ["CRM2", "CRU2"]
    assert frame["open"].tolist() == [10.0, 10.7]
    assert frame["secid"].tolist() == ["CRM2", "CRU2"]
    assert manifest["rows"] == 2
    assert (tmp_path / "bars" / "CRM2.parquet").exists()
    assert (tmp_path / "bars" / "CRZ2.parquet").exists() is False
    assert (tmp_path / "continuous" / "cny_front_1m.parquet").exists()

    calls.clear()
    download_front(contracts, today, tmp_path, fetch, workers=2)
    assert calls == ["CRU2"]

    calls.clear()
    download_front(contracts, today, tmp_path, fetch, force=True, workers=2)
    assert sorted(calls) == ["CRM2", "CRU2"]


def test_download_parser_flags_and_rejects_zero_workers():
    args = build_parser().parse_args(["download", "--force", "--workers", "4", "--today", "2026-09-21"])
    assert args.force is True
    assert args.workers == 4
    assert args.today == date(2026, 9, 21)
    assert main(["download", "--workers", "0"]) == 2
