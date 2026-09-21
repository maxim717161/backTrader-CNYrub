"""Загрузка фронтальных окон и запись склейки."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from cnyrub.bars import bars_path, cache_covers, prepare_bars, read_bars, write_bars
from cnyrub.contracts import Contract, Window, contracts_document, front_windows
from cnyrub.manifest import build_manifest, stitch

FetchCandles = Callable[[Contract, date, date], pd.DataFrame]


def continuous_path(data_dir: Path) -> Path:
    return data_dir / "continuous" / "cny_front_1m.parquet"


def contracts_path(data_dir: Path) -> Path:
    return data_dir / "contracts.json"


def manifest_path(data_dir: Path) -> Path:
    return data_dir / "manifest.json"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def download_front(
    contracts: list[Contract],
    today: date,
    data_dir: Path,
    fetch_candles: FetchCandles,
    *,
    force: bool = False,
    workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Скачать фронтальные окна, склеить ряд и записать manifest.

    Закрытый контракт при повторном запуске берётся из `data/bars/{SECID}.parquet`.
    Текущий контракт скачивается заново: его окно каждый день длиннее.
    """
    if workers < 1:
        raise ValueError("workers должен быть >= 1")
    windows = front_windows(contracts, today)
    loaded: dict[int, pd.DataFrame] = {}

    def load(index: int, window: Window) -> tuple[int, pd.DataFrame]:
        path = bars_path(data_dir, window.secid)
        closed = window.contract.lsttrade < today
        if not force and closed and cache_covers(path, window.start, window.end):
            print(f"{window.secid}: кэш {window.start.isoformat()}..{window.end.isoformat()}", flush=True)
            return index, read_bars(path)
        print(
            f"{window.secid}: загрузка {window.start.isoformat()}..{window.end.isoformat()}",
            flush=True,
        )
        fetched = fetch_candles(window.contract, window.start, window.end)
        frame = prepare_bars(fetched, window.secid, window.start, window.end)
        if frame.empty:
            print(f"{window.secid}: в окне нет свечей", flush=True)
            return index, frame
        write_bars(path, frame, window.start, window.end)
        print(f"{window.secid}: записано {len(frame)} свечей", flush=True)
        return index, frame

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(load, index, window) for index, window in enumerate(windows)]
        errors: list[str] = []
        for future in as_completed(futures):
            try:
                index, frame = future.result()
            except Exception as error:
                errors.append(str(error))
                continue
            loaded[index] = frame
        if errors:
            raise RuntimeError("Не удалось скачать часть контрактов:\n" + "\n".join(errors))

    frames = [loaded[index] for index in range(len(windows))]
    combined, dropped = stitch(frames)
    manifest = build_manifest(
        combined,
        as_of=today,
        expected_secids=[window.secid for window in windows],
        dropped_duplicates=dropped,
    )
    if not combined.empty:
        _atomic_parquet(continuous_path(data_dir), combined)
    write_json(manifest_path(data_dir), manifest)
    return combined, manifest


def publish_contracts(contracts: list[Contract], today: date, data_dir: Path) -> dict[str, object]:
    document = contracts_document(contracts, today)
    write_json(contracts_path(data_dir), document)
    return document
