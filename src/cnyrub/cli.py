"""Команды `python -m cnyrub contracts` и `python -m cnyrub download`."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from cnyrub.contracts import discover_contracts
from cnyrub.download import download_front, publish_contracts
from cnyrub.iss import IssClient

DATA_DIR = Path("data")


def moscow_today() -> date:
    try:
        return datetime.now(ZoneInfo("Europe/Moscow")).date()
    except Exception:
        return (datetime.now(timezone.utc) + timedelta(hours=3)).date()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cnyrub",
        description="Минутная история фронтального фьючерса CNY/RUB с Московской биржи",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    contracts = sub.add_parser("contracts", help="Найти квартальные выпуски CNY и записать data/contracts.json")
    contracts.add_argument(
        "--today",
        type=_parse_date,
        default=None,
        help="Дата, на которую считается фронт, YYYY-MM-DD",
    )

    download = sub.add_parser("download", help="Скачать фронтальные окна и склеить минутный ряд")
    download.add_argument("--force", action="store_true", help="Скачать все окна заново")
    download.add_argument("--workers", type=int, default=4, help="Число параллельных контрактов")
    download.add_argument(
        "--today",
        type=_parse_date,
        default=None,
        help="Зафиксировать конец текущего окна, YYYY-MM-DD",
    )
    return parser


def _print_contracts(document: dict[str, object]) -> None:
    rows = document["contracts"]
    assert isinstance(rows, list)
    print(f"Дата: {document['as_of']}. Фронт: {document['front'] or '—'}. Выпусков: {len(rows)}")
    print(f"{'SECID':<8} {'КОД':<12} {'СТАРТ':<12} {'ЭКСПИР.':<12} ОКНО")
    for row in rows:
        assert isinstance(row, dict)
        if row["in_series"]:
            window = f"{row['window_start']}..{row['window_end']}"
        else:
            window = "—"
        print(f"{row['secid']:<8} {row['shortname']:<12} {row['frsttrade']:<12} {row['lsttrade']:<12} {window}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    today = args.today or moscow_today()
    if args.command == "download" and args.workers < 1:
        print("Ошибка: --workers должен быть >= 1", file=sys.stderr)
        return 2
    try:
        client = IssClient()
        contracts = discover_contracts(today, client.description)
        document = publish_contracts(contracts, today, DATA_DIR)
        _print_contracts(document)
        if args.command == "contracts":
            print(f"Записано {DATA_DIR / 'contracts.json'}")
            return 0
        _frame, manifest = download_front(
            contracts,
            today,
            DATA_DIR,
            lambda contract, start, end: client.candles(contract.secid, start, end),
            force=args.force,
            workers=args.workers,
        )
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(
        f"Склейка: {manifest['rows']} свечей, объём {manifest['volume']}, "
        f"{manifest['first']} — {manifest['last']}"
    )
    print(f"Предупреждений: {len(manifest['warnings'])}")
    print(f"Записано {DATA_DIR / 'continuous' / 'cny_front_1m.parquet'}")
    return 0
