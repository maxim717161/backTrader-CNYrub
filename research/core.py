"""Общая основа исследования: ряд, статистики, прогон сделок с издержками.

Ряд — минутки фронтального контракта (окна из data/contracts.json), по умолчанию
только будни. Все выводы считаются внутри дня: ночной разрыв изучается отдельно.
Все статистики с доверительными интервалами строятся бутстрепом по дням: минутки
внутри дня зависимы, поэтому независимая единица — день.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BARS_DIR = ROOT / "data" / "bars"
MULTIPLIER = 1000.0
TICK = 0.001
COMMISSION = 5.0
SLIPPAGE_TICKS = 1.0
SEED = 20261018


@dataclass(frozen=True)
class Market:
    """bars — настоящие минутки; grid — цена закрытия на каждой минуте дня (без пропусков)."""

    bars: pd.DataFrame
    grid: pd.DataFrame


def round_trip_cost() -> float:
    """Рублей на контракт за круг: комиссия с двух сторон и проскальзывание в тиках."""
    return 2 * COMMISSION + 2 * SLIPPAGE_TICKS * TICK * MULTIPLIER


def load_front(weekdays_only: bool = True, bars_dir: Path = BARS_DIR) -> Market:
    document = json.loads((bars_dir.parent / "contracts.json").read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for row in document["contracts"]:
        start, end = row.get("window_start"), row.get("window_end")
        path = bars_dir / f"{row['secid']}.parquet"
        if not start or not end or not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["datetime", "open", "high", "low", "close", "volume"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        day = frame["datetime"].dt.normalize()
        keep = (day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end))
        frame = frame.loc[keep].copy()
        frame["contract"] = row["secid"]
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    if weekdays_only:
        bars = bars[bars["datetime"].dt.dayofweek < 5].reset_index(drop=True)
    bars["day"] = bars["datetime"].dt.normalize()
    bars = bars.set_index("datetime", drop=False)
    return Market(bars=bars, grid=_build_grid(bars))


def _build_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Каждая минута между первой и последней минуткой дня; пропуск — цена не менялась."""
    pieces = []
    for (contract, day), group in bars.groupby(["contract", "day"], sort=True):
        index = pd.date_range(group.index[0], group.index[-1], freq="min")
        close = group["close"].reindex(index).ffill()
        pieces.append(pd.DataFrame({"close": close, "day": day, "contract": contract}))
    return pd.concat(pieces).sort_index()


def clock_returns(grid: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Непересекающиеся доходности за minutes минут по часам: только внутри дня и подряд."""
    stamp = grid.index
    marks = grid[((stamp.hour * 60 + stamp.minute) % minutes) == 0]
    price = marks["close"].to_numpy()
    when = marks.index
    gap = np.diff(when.values).astype("timedelta64[m]").astype(int)
    same = (marks["day"].to_numpy()[1:] == marks["day"].to_numpy()[:-1]) & (gap == minutes)
    log = np.log(price)
    out = pd.DataFrame(
        {
            "t": when[1:][same],
            "day": marks["day"].to_numpy()[1:][same],
            "r": (log[1:] - log[:-1])[same],
        }
    )
    return out


def cluster_mean(values: np.ndarray, days: np.ndarray) -> tuple[float, float, float, int]:
    """Среднее, его стандартная ошибка с кластеризацией по дням, t-статистика, число событий."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(values))
    frame = pd.DataFrame({"d": days, "x": values - mean})
    sums = frame.groupby("d")["x"].sum().to_numpy()
    se = float(np.sqrt(np.sum(sums**2)) / n)
    t = mean / se if se > 0 else float("nan")
    return mean, se, t, n


def _day_index(days: np.ndarray) -> tuple[np.ndarray, int]:
    codes, unique = pd.factorize(days)
    return codes, len(unique)


def _boot(stat_num: np.ndarray, stat_den: np.ndarray, codes: np.ndarray, ndays: int, reps: int = 400):
    """Бутстреп по дням для отношения сумм: возвращает оценку и 95% интервал."""
    num = np.bincount(codes, weights=stat_num, minlength=ndays)
    den = np.bincount(codes, weights=stat_den, minlength=ndays)
    point = num.sum() / den.sum() if den.sum() else float("nan")
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, ndays, size=(reps, ndays))
    ratios = num[picks].sum(axis=1) / den[picks].sum(axis=1)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(point), float(low), float(high)


def variance_ratio(grid: pd.DataFrame, base: int, q: int) -> tuple[float, float, float]:
    """VR(q) = Var(q*base минут) / (q * Var(base минут)). 1 — случайное блуждание,
    больше 1 — тренд, меньше 1 — возврат. Окна не пересекают границу дня."""
    stamp = grid.index
    marks = grid[((stamp.hour * 60 + stamp.minute) % base) == 0]
    log = np.log(marks["close"].to_numpy())
    day = marks["day"].to_numpy()
    when = marks.index.values
    step = np.diff(when).astype("timedelta64[m]").astype(int)

    def windowed(k: int):
        if len(log) <= k:
            return np.array([]), np.array([], dtype=int)
        ok = (day[k:] == day[:-k]) & ((when[k:] - when[:-k]).astype("timedelta64[m]").astype(int) == k * base)
        return (log[k:] - log[:-k])[ok], pd.factorize(day[k:][ok])[0]

    r1, _ = windowed(1)
    rq, _ = windowed(q)
    d1 = day[1:][((day[1:] == day[:-1]) & (step == base))]
    dq = day[q:][((day[q:] == day[:-q]) & ((when[q:] - when[:-q]).astype("timedelta64[m]").astype(int) == q * base))]
    all_days = pd.unique(np.concatenate([d1, dq]))
    code_of = {d: i for i, d in enumerate(all_days)}
    c1 = np.array([code_of[d] for d in d1])
    cq = np.array([code_of[d] for d in dq])
    n = len(all_days)
    s1 = np.bincount(c1, weights=r1**2, minlength=n)
    n1 = np.bincount(c1, minlength=n).astype(float)
    sq = np.bincount(cq, weights=rq**2, minlength=n)
    nq = np.bincount(cq, minlength=n).astype(float)
    point = (sq.sum() / nq.sum()) / (q * s1.sum() / n1.sum())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, n, size=(400, n))
    boots = (sq[picks].sum(axis=1) / nq[picks].sum(axis=1)) / (q * s1[picks].sum(axis=1) / n1[picks].sum(axis=1))
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(point), float(low), float(high)


def lag_autocorr(returns: pd.DataFrame, lag: int = 1) -> tuple[float, float, float]:
    """Автокорреляция доходностей, разнесённых на lag шагов, внутри дня.

    Эффект «купили по аску, продали по биду» живёт только на соседних шагах: если
    при lag=2 знак тот же, это не шум последней сделки."""
    r = returns["r"].to_numpy()
    day = returns["day"].to_numpy()
    same = day[lag:] == day[:-lag]
    prod = (r[lag:] * r[:-lag])[same]
    sq = (r[:-lag] ** 2)[same]
    codes, ndays = _day_index(day[lag:][same])
    return _boot(prod, sq, codes, ndays)


def lag1_autocorr(returns: pd.DataFrame) -> tuple[float, float, float]:
    return lag_autocorr(returns, 1)


def noise_adjusted_vr(grid: pd.DataFrame, base: int, q: int, reps: int = 400) -> tuple[float, float, float]:
    """VR без шума последней сделки (модель: цена закрытия = истинная цена + независимый шум).

    Шум входит в дисперсию каждого интервала одинаково, а в автоковариацию соседних
    интервалов с минусом: sigma_noise^2 = -cov1. Истинная дисперсия шага base —
    var1 + 2*cov1, дисперсия q шагов без шума — var_q - 2*sigma_noise^2."""
    stamp = grid.index
    marks = grid[((stamp.hour * 60 + stamp.minute) % base) == 0]
    log = np.log(marks["close"].to_numpy())
    day = marks["day"].to_numpy()
    when = marks.index.values

    def pairs(k: int):
        ok = (day[k:] == day[:-k]) & ((when[k:] - when[:-k]).astype("timedelta64[m]").astype(int) == k * base)
        return (log[k:] - log[:-k])[ok], day[k:][ok]

    r1, d1 = pairs(1)
    rq, dq = pairs(q)
    # соседние шаги: доходности подряд внутри дня
    ok = (day[2:] == day[:-2]) & ((when[2:] - when[:-2]).astype("timedelta64[m]").astype(int) == 2 * base)
    a = (log[1:-1] - log[:-2])[ok]
    b = (log[2:] - log[1:-1])[ok]
    dc = day[2:][ok]
    all_days = pd.unique(np.concatenate([d1, dq, dc]))
    code = {d: i for i, d in enumerate(all_days)}
    n = len(all_days)

    def agg(values, days, power=1):
        codes = np.fromiter((code[d] for d in days), dtype=int, count=len(days))
        return np.bincount(codes, weights=values, minlength=n), np.bincount(codes, minlength=n).astype(float)

    s1, n1 = agg(r1**2, d1)
    sq, nq = agg(rq**2, dq)
    sc, nc = agg(a * b, dc)

    def value(i):
        var1 = s1[i].sum() / n1[i].sum()
        varq = sq[i].sum() / nq[i].sum()
        cov1 = sc[i].sum() / nc[i].sum()
        noise = max(-cov1, 0.0)
        true1 = var1 - 2 * noise
        return (varq - 2 * noise) / (q * true1)

    point = value(np.arange(n))
    rng = np.random.default_rng(SEED)
    boots = [value(rng.integers(0, n, size=n)) for _ in range(reps)]
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(point), float(low), float(high)


def sign_persistence(returns: pd.DataFrame) -> tuple[float, float, int]:
    """Доля пар соседних ненулевых движений в одну сторону. 0.5 — нет памяти."""
    r = returns["r"].to_numpy()
    day = returns["day"].to_numpy()
    same = (day[1:] == day[:-1]) & (r[1:] != 0) & (r[:-1] != 0)
    agree = (np.sign(r[1:]) == np.sign(r[:-1]))[same].astype(float)
    mean, se, _, n = cluster_mean(agree, day[1:][same])
    return mean, se, n


def hurst(grid: pd.DataFrame, horizons=(5, 10, 20, 40, 80, 160)) -> float:
    """Показатель Хёрста по наклону log Var(h) от log h: 0.5 блуждание, больше — тренд."""
    variances = []
    for h in horizons:
        r = clock_returns(grid, h)["r"].to_numpy()
        variances.append(np.var(r))
    slope = np.polyfit(np.log(horizons), np.log(variances), 1)[0]
    return float(slope / 2)


def overnight_gaps(market: Market) -> pd.DataFrame:
    """Разрыв закрытие вчера — открытие сегодня и ход первого часа и всего дня после него."""
    rows = []
    for contract, part in market.bars.groupby("contract"):
        days = sorted(part["day"].unique())
        for previous, today in zip(days[:-1], days[1:]):
            a = part[part["day"] == previous]
            b = part[part["day"] == today]
            if len(a) < 30 or len(b) < 90:
                continue
            first = b.iloc[0]
            hour_end = b[b.index <= b.index[0] + pd.Timedelta(minutes=60)].iloc[-1]
            rows.append(
                {
                    "day": today,
                    "gap": float(first["open"] - a.iloc[-1]["close"]),
                    "first_hour": float(hour_end["close"] - first["open"]),
                    "rest": float(b.iloc[-1]["close"] - hour_end["close"]),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Entry:
    index: int
    side: int
    stop: float = 0.0
    target: float = 0.0
    hold: int = 0


def arrays(market: Market) -> dict[str, np.ndarray]:
    bars = market.bars
    return {
        "t": bars.index.values,
        "day": bars["day"].to_numpy(),
        "contract": bars["contract"].to_numpy(),
        "o": bars["open"].to_numpy(float),
        "h": bars["high"].to_numpy(float),
        "l": bars["low"].to_numpy(float),
        "c": bars["close"].to_numpy(float),
        "v": bars["volume"].to_numpy(float),
    }


def _segment_ends(keys: np.ndarray) -> np.ndarray:
    """Для каждой минутки индекс последней минутки того же сегмента (контракт или день)."""
    n = len(keys)
    ends = np.empty(n, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and keys[i] != keys[i + 1]:
            last = i
        ends[i] = last
    return ends


def run_entries(
    data: dict[str, np.ndarray],
    entries: list[Entry],
    *,
    delay: int = 1,
    flatten_day: bool = True,
) -> pd.DataFrame:
    """Прогон входов на одном контракте на счёте, без наложения сделок.

    Вход по закрытию минуты index+delay (так делает и живая функция: сигнал на закрытии,
    заявка исполняется следующей минуткой). Стоп и цель проверяются по high/low следующих
    минуток, внутри минутки стоп приоритетнее цели. hold — предельное число минуток
    (0 — без предела). Конец дня закрывает сделку, если flatten_day, конец контракта —
    всегда. Результат — рубли на контракт после комиссии и проскальзывания."""
    cost = round_trip_cost()
    close, high, low, day = data["c"], data["h"], data["l"], data["day"]
    n = len(close)
    if "_end_day" not in data:
        data["_end_day"] = _segment_ends(np.char.add(data["contract"].astype(str), day.astype("datetime64[D]").astype(str)))
        data["_end_contract"] = _segment_ends(data["contract"].astype(str))
    end_day, end_contract = data["_end_day"], data["_end_contract"]
    out = []
    busy_until = -1
    for entry in sorted(entries, key=lambda e: e.index):
        i = entry.index + delay
        if i >= n - 1 or entry.index <= busy_until:
            continue
        if end_contract[i] != end_contract[entry.index] or (flatten_day and end_day[i] != end_day[entry.index]):
            continue
        price = close[i]
        side = entry.side
        limit = end_day[i] if flatten_day else end_contract[i]
        if entry.hold:
            limit = min(limit, i + entry.hold)
        if limit <= i:
            continue
        sl = slice(i + 1, limit + 1)
        hit = np.zeros(limit - i, dtype=bool)
        stop_hit = target_hit = None
        if entry.stop > 0:
            stop_px = price - side * entry.stop
            stop_hit = (low[sl] <= stop_px) if side > 0 else (high[sl] >= stop_px)
        if entry.target > 0:
            target_px = price + side * entry.target
            target_hit = (high[sl] >= target_px) if side > 0 else (low[sl] <= target_px)
        first_stop = int(np.argmax(stop_hit)) if stop_hit is not None and stop_hit.any() else None
        first_target = int(np.argmax(target_hit)) if target_hit is not None and target_hit.any() else None
        reason = "end"
        k = limit - i - 1
        exit_px = close[limit]
        if first_stop is not None and (first_target is None or first_stop <= first_target):
            k, exit_px, reason = first_stop, price - side * entry.stop, "stop"
        elif first_target is not None:
            k, exit_px, reason = first_target, price + side * entry.target, "target"
        elif entry.hold and limit - i >= entry.hold:
            reason = "time"
        last = i + 1 + k
        out.append(
            {
                "day": day[i],
                "contract": data["contract"][i],
                "side": side,
                "bars": last - i,
                "reason": reason,
                "gross": side * (exit_px - price) * MULTIPLIER,
                "net": side * (exit_px - price) * MULTIPLIER - cost,
            }
        )
        busy_until = last
    return pd.DataFrame(out, columns=["day", "contract", "side", "bars", "reason", "gross", "net"])


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"n": 0, "net": float("nan"), "t": float("nan"), "gross": float("nan"), "hit": float("nan"), "pf": float("nan")}
    mean, _, t, n = cluster_mean(trades["net"].to_numpy(), trades["day"].to_numpy())
    wins = trades.loc[trades["net"] > 0, "net"].sum()
    losses = -trades.loc[trades["net"] <= 0, "net"].sum()
    return {
        "n": n,
        "net": mean,
        "t": t,
        "gross": float(trades["gross"].mean()),
        "hit": float((trades["net"] > 0).mean()),
        "pf": float(wins / losses) if losses > 0 else float("inf"),
    }


def daily_pnl(trades: pd.DataFrame, days: np.ndarray) -> np.ndarray:
    """Сумма результата по каждому торговому дню выборки (нули там, где сделок не было)."""
    if trades.empty:
        return np.zeros(len(days))
    sums = trades.groupby("day")["net"].sum()
    return sums.reindex(days, fill_value=0.0).to_numpy()


def reality_check(daily: np.ndarray, reps: int = 2000) -> tuple[float, np.ndarray]:
    """Проверка перебора (White): p-значение для лучшего t среди всех вариантов.

    daily — матрица дни x варианты. Нуль моделируется случайной сменой знака дневного
    результата, поэтому ни один вариант не имеет настоящего среднего."""
    mean = daily.mean(axis=0)
    std = daily.std(axis=0, ddof=1)
    std[std == 0] = np.nan
    t_obs = mean / (std / np.sqrt(len(daily)))
    best = np.nanmax(t_obs)
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(reps):
        signs = rng.choice([-1.0, 1.0], size=(len(daily), 1))
        flipped = daily * signs
        m = flipped.mean(axis=0)
        s = flipped.std(axis=0, ddof=1)
        s[s == 0] = np.nan
        if np.nanmax(m / (s / np.sqrt(len(daily)))) >= best:
            count += 1
    return (count + 1) / (reps + 1), t_obs


def prior_levels(data: dict[str, np.ndarray], channel: int) -> dict[str, np.ndarray]:
    """Канал по channel предыдущим минуткам внутри контракта: так считает и стратегия."""
    high = pd.Series(data["h"])
    low = pd.Series(data["l"])
    rng = pd.Series(data["h"] - data["l"])
    by = pd.Series(data["contract"])
    prior_high = high.groupby(by).transform(lambda s: s.rolling(channel).max().shift(1))
    prior_low = low.groupby(by).transform(lambda s: s.rolling(channel).min().shift(1))
    prior_range = rng.groupby(by).transform(lambda s: s.rolling(channel).median().shift(1))
    return {
        "high": prior_high.to_numpy(),
        "low": prior_low.to_numpy(),
        "range": prior_range.to_numpy(),
    }


def cached_levels(data: dict[str, np.ndarray], channel: int) -> dict[str, np.ndarray]:
    key = f"_levels_{channel}"
    if key not in data:
        data[key] = prior_levels(data, channel)
    return data[key]


def breakout_entries(data: dict[str, np.ndarray], channel: int, *, side_sign: int = 1, stop_mult: float = 0.0, target_mult: float = 0.0, hold: int = 0, min_stop: float = 0.0) -> list[Entry]:
    """Свежие пробои канала. side_sign=1 — по пробою (тренд), -1 — против (контртренд)."""
    levels = cached_levels(data, channel)
    close = data["c"]
    up = close > levels["high"]
    down = close < levels["low"]
    up_fresh = up & ~np.r_[False, up[:-1]]
    down_fresh = down & ~np.r_[False, down[:-1]]
    entries = []
    for i in np.flatnonzero(up_fresh | down_fresh):
        raw = levels["range"][i]
        stop = max(stop_mult * raw, min_stop) if stop_mult and raw == raw else 0.0
        target = max(target_mult * raw, min_stop) if target_mult and raw == raw else 0.0
        side = (1 if up_fresh[i] else -1) * side_sign
        entries.append(Entry(int(i), side, stop, target, hold))
    return entries


def forward_moves(market: Market, indices: np.ndarray, sides: np.ndarray, horizons: tuple[int, ...], delay: int = 1) -> pd.DataFrame:
    """Ход цены в сторону сигнала, рублей на контракт, до выхода через h минут после входа.

    Вход по закрытию минуты index+delay, выход по закрытию через h минут; события, у
    которых выход уходит за границу дня, отбрасываются."""
    grid = market.grid
    close = grid["close"]
    gday = grid["day"]
    bars = market.bars
    entry_time = bars.index[indices] + pd.Timedelta(minutes=delay)
    rows = {"day": bars["day"].to_numpy()[indices], "side": sides}
    base = close.reindex(entry_time).to_numpy()
    base_day = gday.reindex(entry_time).to_numpy()
    for h in horizons:
        exit_time = entry_time + pd.Timedelta(minutes=h)
        price = close.reindex(exit_time).to_numpy()
        same = gday.reindex(exit_time).to_numpy() == base_day
        move = sides * (price - base) * MULTIPLIER
        move[~same] = np.nan
        rows[f"m{h}"] = move
    return pd.DataFrame(rows)
