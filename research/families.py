"""Семейства правил входа с параметром направления.

sign=+1 — торговать вместе с сигналом (тренд), sign=-1 — против него (контртренд).
Одно и то же правило проверяется в обе стороны, поэтому гипотезы честно сравнимы:
меняется только знак, а не определение сигнала.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research import core

FAMILIES = ("donchian", "zscore", "momentum", "gap", "firsthour", "scalp")


@dataclass(frozen=True)
class Spec:
    family: str
    sign: int
    params: dict = field(default_factory=dict)
    flatten_day: bool = True

    def label(self) -> str:
        parts = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        overnight = "" if self.flatten_day else ",ночь"
        return f"{self.family}[{'+' if self.sign > 0 else '-'}]({parts}{overnight})"


def session(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Начало дня по каждой минутке и минуты от первой сделки дня."""
    if "_session" not in data:
        day, contract, t = data["day"], data["contract"], data["t"]
        first = np.r_[True, (day[1:] != day[:-1]) | (contract[1:] != contract[:-1])]
        start = np.maximum.accumulate(np.where(first, np.arange(len(day)), 0))
        since = (t - t[start]).astype("timedelta64[m]").astype(float)
        data["_session"] = {"first": first, "start": start, "since": since}
    return data["_session"]


def lagged_close(market: core.Market, data: dict[str, np.ndarray], minutes: int) -> np.ndarray:
    """Цена закрытия ровно minutes минут назад по часам (NaN, если там был другой день)."""
    key = f"_past_{minutes}"
    if key not in data:
        bars = market.bars
        stamp = bars.index - pd.Timedelta(minutes=minutes)
        close = market.grid["close"].reindex(stamp).to_numpy()
        same_day = market.grid["day"].reindex(stamp).to_numpy() == bars["day"].to_numpy()
        data[key] = np.where(same_day, close, np.nan)
    return data[key]


def _range(data: dict[str, np.ndarray], channel: int = 60) -> np.ndarray:
    return core.cached_levels(data, channel)["range"]


def _make(index: int, side: int, rng: np.ndarray, stop_mult: float, target_mult: float, hold: int, min_stop: float) -> core.Entry | None:
    width = rng[index]
    if (stop_mult or target_mult) and not width == width:
        return None
    stop = max(stop_mult * width, min_stop) if stop_mult else 0.0
    target = max(target_mult * width, min_stop) if target_mult else 0.0
    return core.Entry(int(index), int(side), stop, target, int(hold))


def build(spec: Spec, market: core.Market, data: dict[str, np.ndarray]) -> list[core.Entry]:
    p = spec.params
    stop = p.get("stop", 0.0)
    target = p.get("target", 0.0)
    hold = p.get("hold", 0)
    min_stop = p.get("min_stop", 0.0)
    if spec.family == "donchian":
        return core.breakout_entries(data, p["channel"], side_sign=spec.sign, stop_mult=stop, target_mult=target, hold=hold, min_stop=min_stop)
    close = data["c"]
    rng = _range(data)
    entries: list[core.Entry] = []

    def add(index: int, direction: int) -> None:
        entry = _make(index, direction * spec.sign, rng, stop, target, hold, min_stop)
        if entry is not None:
            entries.append(entry)

    if spec.family == "zscore":
        by = pd.Series(data["contract"])
        series = pd.Series(close)
        mean = series.groupby(by).transform(lambda s: s.rolling(p["window"]).mean())
        std = series.groupby(by).transform(lambda s: s.rolling(p["window"]).std())
        z = ((series - mean) / std.where(std > 0)).to_numpy()
        up = z > p["k"]
        down = z < -p["k"]
        up_fresh = up & ~np.r_[False, up[:-1]]
        down_fresh = down & ~np.r_[False, down[:-1]]
        for i in np.flatnonzero(up_fresh | down_fresh):
            add(i, 1 if up_fresh[i] else -1)
    elif spec.family == "momentum":
        past = lagged_close(market, data, p["lookback"])
        move = close - past
        stamp = market.bars.index
        aligned = ((stamp.hour * 60 + stamp.minute) % p["every"]) == 0
        ok = aligned & (np.abs(move) >= p["min_move"] * core.UNIT)
        for i in np.flatnonzero(ok):
            add(i, 1 if move[i] > 0 else -1)
    elif spec.family == "gap":
        s = session(data)
        previous_close = np.r_[np.nan, close[:-1]]
        same_contract = np.r_[False, data["contract"][1:] == data["contract"][:-1]]
        gap = data["o"] - previous_close
        ok = s["first"] & same_contract & (np.abs(gap) >= p["min_gap"] * core.UNIT)
        for i in np.flatnonzero(ok):
            add(i, 1 if gap[i] > 0 else -1)
    elif spec.family == "firsthour":
        s = session(data)
        since, start = s["since"], s["start"]
        window = p["window"]
        inside = since <= window
        last_inside = inside & ~np.r_[inside[1:], False]
        last_inside |= inside & np.r_[start[1:] != start[:-1], True]
        move = close - data["o"][start]
        ok = last_inside & (since >= window * 0.5) & (np.abs(move) >= p["min_move"] * core.UNIT)
        for i in np.flatnonzero(ok):
            add(i, 1 if move[i] > 0 else -1)
    elif spec.family == "scalp":
        past = lagged_close(market, data, 1)
        move = close - past
        ok = np.abs(move) >= p["min_move"] * core.UNIT
        for i in np.flatnonzero(ok):
            add(i, 1 if move[i] > 0 else -1)
    else:
        raise ValueError(spec.family)
    return entries
