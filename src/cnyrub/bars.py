"""Минутные свечи одного контракта: колонки, типы и кэш parquet."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BARS_COLUMNS = [
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

_META_START = b"cnyrub.window_start"
_META_END = b"cnyrub.window_end"
_META_COMPLETE = b"cnyrub.complete"
_CACHE_VERSION = b"1"
_META_VERSION = b"cnyrub.cache_version"


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.Series(dtype="datetime64[ns]"),
            "end": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64"),
            "secid": pd.Series(dtype="object"),
        }
    )


def _as_moscow_naive(values: pd.Series) -> pd.Series:
    """Оставить московское время биржи, без сдвига стрелок в UTC."""
    parsed = pd.to_datetime(values, errors="coerce")
    timezone = getattr(parsed.dt, "tz", None)
    if timezone is not None:
        parsed = parsed.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    return parsed


def prepare_bars(frame: pd.DataFrame, secid: str, start: date, end: date) -> pd.DataFrame:
    """Привести страницу ISS или уже собранный кадр к колонкам ряда.

    `begin` биржи становится `datetime`. Свечи вне календарного окна отбрасываются.
    """
    if frame.empty and "datetime" not in frame.columns and "begin" not in frame.columns:
        return empty_bars()

    renamed = frame.rename(columns={"begin": "datetime"})
    missing = [column for column in BARS_COLUMNS if column not in renamed.columns and column != "secid"]
    if missing:
        raise ValueError(f"{secid}: в свечах нет колонок {', '.join(missing)}")

    out = pd.DataFrame(
        {
            "datetime": _as_moscow_naive(renamed["datetime"]),
            "end": _as_moscow_naive(renamed["end"]),
            "open": pd.to_numeric(renamed["open"], errors="coerce"),
            "high": pd.to_numeric(renamed["high"], errors="coerce"),
            "low": pd.to_numeric(renamed["low"], errors="coerce"),
            "close": pd.to_numeric(renamed["close"], errors="coerce"),
            "volume": pd.to_numeric(renamed["volume"], errors="coerce").fillna(0).astype("int64"),
            "value": pd.to_numeric(renamed["value"], errors="coerce").fillna(0.0),
            "secid": secid,
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    day = out["datetime"].dt.normalize()
    mask = (day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end))
    out = out.loc[mask]
    out = out.sort_values("datetime", kind="mergesort").drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def bars_path(data_dir: Path, secid: str) -> Path:
    return data_dir / "bars" / f"{secid}.parquet"


def write_bars(path: Path, frame: pd.DataFrame, start: date, end: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame.loc[:, BARS_COLUMNS], preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[_META_START] = start.isoformat().encode()
    metadata[_META_END] = end.isoformat().encode()
    metadata[_META_COMPLETE] = b"1"
    metadata[_META_VERSION] = _CACHE_VERSION
    table = table.replace_schema_metadata(metadata)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def cache_bounds(path: Path) -> tuple[date, date] | None:
    """Границы уже записанного окна или None, если кэш неполный."""
    if not path.exists():
        return None
    metadata = pq.read_schema(path).metadata or {}
    if metadata.get(_META_COMPLETE) != b"1" or metadata.get(_META_VERSION) != _CACHE_VERSION:
        return None
    raw_start = metadata.get(_META_START)
    raw_end = metadata.get(_META_END)
    if not raw_start or not raw_end:
        return None
    try:
        return date.fromisoformat(raw_start.decode()), date.fromisoformat(raw_end.decode())
    except ValueError:
        return None


def cache_covers(path: Path, start: date, end: date) -> bool:
    return cache_bounds(path) == (start, end)


def read_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in BARS_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"{path}: нет колонки {column}")
    frame = frame.loc[:, BARS_COLUMNS]
    frame["datetime"] = _as_moscow_naive(frame["datetime"])
    frame["end"] = _as_moscow_naive(frame["end"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype("int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["secid"] = frame["secid"].astype(str)
    return frame.reset_index(drop=True)
