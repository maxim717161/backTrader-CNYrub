"""Дневной моментум на живом счёте. С окнами не разговаривает.

Знак хода за 20 дней держится 5 дней. Плечо 3. Следующий день без позиции,
если |ход| больше 2 своих σ. День решения с объёмом или диапазоном больше
двух медиан предыдущих 20 дней не открывает следующие 5 дней. Свой файл
peers/momentum.json пишет. Чужие файлы не читает, семь окон его пока не смотрят.
"""

from __future__ import annotations

from datetime import datetime
from statistics import median, stdev
from zoneinfo import ZoneInfo

LOOKBACK = 20
HOLD = 5
LEVERAGE = 3.0
SHOCK = 2.0
WINDOW = 20
SPIKE = 2.0
# Первая заявка — на следующий день после 20-го закрытия. Медиана объёма и
# диапазона на дне решения уже берёт 20 предыдущих сессий, раньше копить не нужно.
HISTORY_DAYS = LOOKBACK + 1
MULTIPLIER = 1000.0


def sign_for_today(days: list[dict[str, object]]) -> int:
    """Знак на сегодня. days — только завершённые сессии, сегодня в список не входит."""
    today = len(days)
    if today < LOOKBACK + 1:
        return 0
    decision = LOOKBACK + ((today - (LOOKBACK + 1)) // HOLD) * HOLD
    if not decision + 1 <= today <= decision + HOLD:
        return 0
    if _shock(days, today - 1):
        return 0
    if _climax(days, decision) or _wide(days, decision):
        return 0
    move = _close(days[decision]) - _close(days[decision - LOOKBACK])
    if move > 0:
        return 1
    if move < 0:
        return -1
    return 0


def target_lots(sign: int, equity: float, previous_close: float, margin: float, leverage: float = LEVERAGE) -> int:
    """Контракты со знаком: плечо от прошлого закрытия, но не больше, чем покрывает ГО."""
    if sign == 0 or equity <= 0 or previous_close <= 0 or leverage <= 0:
        return 0
    notional = previous_close * MULTIPLIER
    raw = int(leverage * equity // notional)
    if margin > 0:
        raw = min(raw, int(equity // margin))
    if raw < 1:
        return 0
    return sign * raw


def fold_candle(day: dict[str, object] | None, moment: datetime, opened: float, high: float, low: float, close: float, volume: float) -> dict[str, object]:
    """Собрать минутки одной даты в дневную свечу."""
    stamp = moment.date().isoformat()
    if day is None or day.get("d") != stamp:
        return {"d": stamp, "o": opened, "h": high, "l": low, "c": close, "v": volume}
    day["h"] = max(float(day["h"]), high)
    day["l"] = min(float(day["l"]), low)
    day["c"] = close
    day["v"] = float(day["v"]) + volume
    return day


def absorb(days: list[dict[str, object]], today: dict[str, object] | None, candles: list[object]) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Минутки по времени. Смена даты закрывает предыдущий день."""
    completed = list(days)
    current = today
    for candle in candles:
        moment = _moscow(candle.time)
        stamp = moment.date().isoformat()
        if current is not None and current.get("d") != stamp:
            completed.append(current)
            current = None
        current = fold_candle(current, moment, candle.open, candle.high, candle.low, candle.close, candle.volume)
    return completed, current


def _moscow(moment: datetime) -> datetime:
    zone = ZoneInfo("Europe/Moscow")
    if moment.tzinfo is None:
        return moment.replace(tzinfo=zone)
    return moment.astimezone(zone)


def _close(day: dict[str, object]) -> float:
    return float(day["c"])


def _percent(days: list[dict[str, object]], index: int) -> float:
    if index <= 0:
        return 0.0
    previous = _close(days[index - 1])
    if abs(previous) <= 1e-9:
        return 0.0
    return (_close(days[index]) - previous) / previous


def _shock(days: list[dict[str, object]], index: int) -> bool:
    if index < WINDOW - 1:
        return False
    sample = [_percent(days, cursor) for cursor in range(index - WINDOW + 1, index + 1)]
    sigma = stdev(sample)
    if not sigma > 0:
        return False
    return abs(_percent(days, index)) > SHOCK * sigma


def _prior_median(values: list[float]) -> float | None:
    if len(values) < WINDOW:
        return None
    return float(median(values[-WINDOW:]))


def _climax(days: list[dict[str, object]], index: int) -> bool:
    if index < WINDOW:
        return False
    prior = _prior_median([float(days[cursor]["v"]) for cursor in range(index)])
    if prior is None or not prior > 0:
        return False
    return float(days[index]["v"]) > SPIKE * prior


def _wide(days: list[dict[str, object]], index: int) -> bool:
    if index < WINDOW:
        return False
    ranges = [_span(days[cursor]) for cursor in range(index)]
    prior = _prior_median(ranges)
    if prior is None or not prior > 0:
        return False
    return _span(days[index]) > SPIKE * prior


def _span(day: dict[str, object]) -> float:
    close = _close(day)
    if abs(close) <= 1e-9:
        return 0.0
    return (float(day["h"]) - float(day["l"])) / close
