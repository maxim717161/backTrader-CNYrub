"""Уровни канала на последней минуте. Окно не включает саму эту минуту.

Считается так же, как channel_view в исследовании: максимум, минимум и
медиана предыдущих N баров, медиана объёма той же минуты суток.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime

from cnyrub.engine import CLOCK_DAYS
from cnyrub.live.config import StrategyParams


def bar_levels(bars: list[dict[str, object]], params: StrategyParams, clock_days: int = CLOCK_DAYS) -> dict[str, float]:
    """Уровни для последнего бара. Неполное окно даёт NaN и entry_ready 0."""
    nan = float("nan")
    levels = {
        "prior_high": nan,
        "prior_low": nan,
        "prior_vol": nan,
        "prior_range": nan,
        "exit_high": nan,
        "exit_low": nan,
        "clock_vol": nan,
        "entry_ready": 0.0,
    }
    if not bars:
        return levels
    levels["prior_high"], levels["prior_low"], levels["prior_vol"], levels["prior_range"], ready = _window(
        bars, params.channel
    )
    levels["entry_ready"] = 1.0 if ready else 0.0
    if params.exit_channel > 0:
        high, low, _vol, _range, exit_ready = _window(bars, params.exit_channel)
        if exit_ready:
            levels["exit_high"] = high
            levels["exit_low"] = low
    levels["clock_vol"] = _clock_volume(bars, clock_days)
    return levels


def _window(bars: list[dict[str, object]], length: int) -> tuple[float, float, float, float, bool]:
    nan = float("nan")
    if length <= 0 or len(bars) < length + 1:
        return nan, nan, nan, nan, False
    window = bars[-(length + 1) : -1]
    highs = [float(bar["h"]) for bar in window]
    lows = [float(bar["l"]) for bar in window]
    volumes = [float(bar["v"]) for bar in window]
    ranges = [high - low for high, low in zip(highs, lows)]
    return max(highs), min(lows), statistics.median(volumes), statistics.median(ranges), True


def _clock_volume(bars: list[dict[str, object]], lookback: int) -> float:
    """Медиана объёма той же минуты суток по предыдущим lookback наблюдениям."""
    if lookback <= 0 or len(bars) < 2:
        return float("nan")
    minute = _minute_of_day(str(bars[-1]["t"]))
    seen = [
        float(bar["v"])
        for bar in bars[:-1]
        if _minute_of_day(str(bar["t"])) == minute
    ]
    if len(seen) < lookback:
        return float("nan")
    return float(statistics.median(seen[-lookback:]))


def _minute_of_day(stamp: str) -> int:
    moment = datetime.fromisoformat(stamp)
    return moment.hour * 60 + moment.minute


def is_nan(value: float) -> bool:
    return math.isnan(value)
