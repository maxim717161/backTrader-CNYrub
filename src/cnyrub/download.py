"""Загрузка истории контрактов и запись фронтальной склейки."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from cnyrub.bars import bars_path, cache_bounds, prepare_bars, read_bars, write_bars
from cnyrub.contracts import Contract, Window, contracts_document, front_windows, history_windows
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


def _merge_bars(prefix: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if prefix.empty:
        return existing.reset_index(drop=True)
    if existing.empty:
        return prefix.reset_index(drop=True)
    frame = pd.concat([prefix, existing], ignore_index=True)
    frame = frame.sort_values("datetime", kind="mergesort").drop_duplicates("datetime", keep="last")
    return frame.reset_index(drop=True)


def _clip(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Оставить в склейке только фронтальное окно, без добавленного месяца."""
    if frame.empty:
        return frame
    day = frame["datetime"].dt.normalize()
    mask = (day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end))
    return frame.loc[mask].reset_index(drop=True)


def _load_window(
    window: Window,
    data_dir: Path,
    fetch_candles: FetchCandles,
    *,
    force: bool,
) -> pd.DataFrame:
    """Скачать историю контракта. Уже лежащий хвост с тем же концом дописывается спереди."""
    path = bars_path(data_dir, window.secid)
    bounds = None if force else cache_bounds(path)
    if bounds == (window.start, window.end):
        print(f"{window.secid}: кэш {window.start.isoformat()}..{window.end.isoformat()}", flush=True)
        return read_bars(path)

    if bounds is not None and bounds[1] == window.end and bounds[0] > window.start:
        prefix_end = bounds[0] - timedelta(days=1)
        print(
            f"{window.secid}: дополнение {window.start.isoformat()}..{prefix_end.isoformat()} "
            f"к кэшу до {window.end.isoformat()}",
            flush=True,
        )
        fetched = fetch_candles(window.contract, window.start, prefix_end)
        prefix = prepare_bars(fetched, window.secid, window.start, prefix_end)
        frame = _merge_bars(prefix, read_bars(path))
        if frame.empty:
            print(f"{window.secid}: в окне нет свечей", flush=True)
            return frame
        write_bars(path, frame, window.start, window.end)
        print(f"{window.secid}: записано {len(frame)} свечей", flush=True)
        return frame

    print(
        f"{window.secid}: загрузка {window.start.isoformat()}..{window.end.isoformat()}",
        flush=True,
    )
    fetched = fetch_candles(window.contract, window.start, window.end)
    frame = prepare_bars(fetched, window.secid, window.start, window.end)
    if frame.empty:
        print(f"{window.secid}: в окне нет свечей", flush=True)
        return frame
    write_bars(path, frame, window.start, window.end)
    print(f"{window.secid}: записано {len(frame)} свечей", flush=True)
    return frame


def download_front(
    contracts: list[Contract],
    today: date,
    data_dir: Path,
    fetch_candles: FetchCandles,
    *,
    force: bool = False,
    workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Скачать историю с лишним месяцем, а в склейку положить только фронтальные окна.

    Закрытый контракт при повторном запуске берётся из `data/bars/{SECID}.parquet`.
    Если кэш короче спереди, а конец совпадает, дописывается только недостающий месяц.
    Текущий контракт скачивается целиком, когда его конец стал длиннее кэша.
    """
    if workers < 1:
        raise ValueError("workers должен быть >= 1")
    windows = history_windows(contracts, today)
    fronts = {window.secid: window for window in front_windows(contracts, today)}
    loaded: dict[int, pd.DataFrame] = {}

    def load(index: int, window: Window) -> tuple[int, pd.DataFrame]:
        return index, _load_window(window, data_dir, fetch_candles, force=force)

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

    clipped = [
        _clip(loaded[index], fronts[window.secid].start, fronts[window.secid].end)
        for index, window in enumerate(windows)
    ]
    combined, dropped = stitch(clipped)
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
