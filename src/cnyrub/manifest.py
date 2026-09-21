"""Сводка склейки: число свечей, объём, края ряда и предупреждения."""

from __future__ import annotations

from datetime import date

import pandas as pd

from cnyrub.bars import empty_bars


def _stamp(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def stitch(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Склеить окна по времени без поправки цены на календарный спред.

    При двух свечах на одну минуту остаётся свеча более позднего контракта:
    окна передаются в порядке экспирации, сортировка по времени устойчивая.
    """
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return empty_bars(), 0
    combined = pd.concat(usable, ignore_index=True)
    combined = combined.sort_values("datetime", kind="mergesort")
    duplicates = int(combined["datetime"].duplicated().sum())
    combined = combined.drop_duplicates("datetime", keep="last").reset_index(drop=True)
    return combined, duplicates


def build_manifest(
    frame: pd.DataFrame,
    *,
    as_of: date,
    expected_secids: list[str],
    dropped_duplicates: int = 0,
) -> dict[str, object]:
    warnings: list[str] = []
    if frame.empty:
        warnings.append("В склейке нет свечей.")
        return {
            "as_of": as_of.isoformat(),
            "rows": 0,
            "volume": 0,
            "first": None,
            "last": None,
            "windows": [],
            "warnings": warnings,
        }

    present = set(frame["secid"].astype(str))
    for secid in expected_secids:
        if secid not in present:
            warnings.append(f"{secid}: в окне нет свечей.")

    values = pd.to_numeric(frame["value"], errors="coerce")
    if bool((values.fillna(0) == 0).all()):
        warnings.append(
            "Поле value равно нулю во всех свечах: ISS так отдаёт оборот по этим фьючерсам."
        )

    if dropped_duplicates:
        warnings.append(
            f"Повторяющиеся минуты: {dropped_duplicates}. Оставлена свеча более позднего контракта."
        )

    ordered = frame.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    dates = ordered["datetime"]
    for position in range(1, len(ordered)):
        gap_days = (dates.iloc[position].normalize() - dates.iloc[position - 1].normalize()).days
        if gap_days >= 5:
            warnings.append(
                "Перерыв "
                f"{gap_days} суток между {_stamp(dates.iloc[position - 1])} "
                f"и {_stamp(dates.iloc[position])}."
            )

    secids = ordered["secid"].astype(str)
    for position in range(1, len(ordered)):
        previous = secids.iloc[position - 1]
        current = secids.iloc[position]
        if previous == current:
            continue
        close = float(ordered["close"].iloc[position - 1])
        opened = float(ordered["open"].iloc[position])
        jump = opened - close
        warnings.append(f"Стык {previous}/{current}: {close:.3f} -> {opened:.3f} ({jump:+.3f}).")

    windows = []
    for secid in expected_secids:
        part = ordered.loc[secids == secid]
        if part.empty:
            continue
        windows.append(
            {
                "secid": secid,
                "rows": int(len(part)),
                "volume": int(part["volume"].sum()),
                "first": _stamp(part["datetime"].iloc[0]),
                "last": _stamp(part["datetime"].iloc[-1]),
            }
        )

    return {
        "as_of": as_of.isoformat(),
        "rows": int(len(ordered)),
        "volume": int(ordered["volume"].sum()),
        "first": _stamp(ordered["datetime"].iloc[0]),
        "last": _stamp(ordered["datetime"].iloc[-1]),
        "windows": windows,
        "warnings": warnings,
    }
