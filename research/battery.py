"""Описательная батарея: есть ли в минутках память, и на каком масштабе.

Одинакова для обеих гипотез (тренд и контртренд): сначала меряем, есть ли вообще
структура в ценах, и только потом проверяем, можно ли на ней заработать после издержек.
Каждая таблица — это DataFrame со строками-масштабами и готовыми подписями вида
«оценка [нижняя; верхняя]» (95% интервал бутстрепом по дням, для дневного масштаба —
по контрактам).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research import core

CLOCK_HORIZONS = (1, 5, 15, 30, 60, 120)
BREAKOUT_CHANNELS = (30, 60, 120, 525)
EVENT_HORIZONS = (5, 15, 30, 60, 120, 240)


def fmt(point: float, low: float, high: float, digits: int = 3) -> str:
    return f"{point:+.{digits}f} [{low:+.{digits}f}; {high:+.{digits}f}]"


def verdict(low: float, high: float, neutral: float) -> str:
    """Знак вывода по интервалу: интервал целиком выше или ниже нейтрального значения."""
    if low > neutral:
        return "+"
    if high < neutral:
        return "-"
    return "0"


def subset(market: core.Market, keep_days) -> core.Market:
    days = pd.Index(keep_days)
    bars = market.bars[market.bars["day"].isin(days)]
    grid = market.grid[market.grid["day"].isin(days)]
    return core.Market(bars=bars, grid=grid)


def autocorrelations(market: core.Market, horizons=CLOCK_HORIZONS, lags=(1, 2, 3)) -> pd.DataFrame:
    rows = {}
    for h in horizons:
        returns = core.clock_returns(market.grid, h)
        rows[f"{h} мин"] = {f"лаг {lag}": core.lag_autocorr(returns, lag) for lag in lags}
    return pd.DataFrame(rows).T


def persistence(market: core.Market, horizons=CLOCK_HORIZONS) -> pd.DataFrame:
    rows = []
    for h in horizons:
        returns = core.clock_returns(market.grid, h)
        share, se, n = core.sign_persistence(returns)
        rows.append({"масштаб": f"{h} мин", "доля пар в одну сторону": share, "ст.ошибка": se, "пар": n})
    return pd.DataFrame(rows)


def variance_ratios(market: core.Market, plan=((5, (2, 3, 6, 12, 24)), (15, (2, 4, 8)))) -> pd.DataFrame:
    rows = []
    for base, multiples in plan:
        for q in multiples:
            raw = core.variance_ratio(market.grid, base, q)
            adjusted = core.noise_adjusted_vr(market.grid, base, q, reps=200)
            rows.append(
                {
                    "шаг": f"{base} мин",
                    "горизонт": f"{base * q} мин",
                    "VR сырой": raw,
                    "VR без шума последней сделки": adjusted,
                }
            )
    return pd.DataFrame(rows)


def daily_returns(market: core.Market) -> pd.DataFrame:
    """Закрытие дня к закрытию дня внутри контракта; ключ кластера — контракт."""
    last = market.bars.groupby(["contract", "day"])["close"].last().reset_index()
    frames = []
    for contract, part in last.groupby("contract"):
        part = part.sort_values("day")
        log = np.log(part["close"].to_numpy())
        frames.append(pd.DataFrame({"t": part["day"].to_numpy()[1:], "day": contract, "r": np.diff(log)}))
    return pd.concat(frames, ignore_index=True)


def daily_variance_ratio(daily: pd.DataFrame, q: int) -> tuple[float, float, float]:
    contracts = []
    for _, part in daily.groupby("day"):
        r = part["r"].to_numpy()
        if len(r) <= q:
            continue
        cum = np.cumsum(np.r_[0.0, r])
        rq = cum[q:] - cum[:-q]
        contracts.append((np.sum(r**2), len(r), np.sum(rq**2), len(rq)))
    s1, n1, sq, nq = (np.array(column, dtype=float) for column in zip(*contracts))
    point = (sq.sum() / nq.sum()) / (q * s1.sum() / n1.sum())
    rng = np.random.default_rng(core.SEED)
    picks = rng.integers(0, len(s1), size=(400, len(s1)))
    boots = (sq[picks].sum(axis=1) / nq[picks].sum(axis=1)) / (q * s1[picks].sum(axis=1) / n1[picks].sum(axis=1))
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(point), float(low), float(high)


def daily_scale(market: core.Market) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_returns(market)
    acf = pd.DataFrame({"": {f"лаг {lag}": core.lag_autocorr(daily, lag) for lag in range(1, 6)}}).T
    vr = pd.DataFrame({f"{q} дн": {"VR": daily_variance_ratio(daily, q)} for q in (2, 5, 10, 20)})
    return acf, vr


def gap_memory(market: core.Market) -> pd.DataFrame:
    """Продолжается ли ночной разрыв: наклон хода дня на размер разрыва (1 — полное продолжение)."""
    gaps = core.overnight_gaps(market)
    codes, ndays = core._day_index(gaps["day"].to_numpy())
    rows = []
    for name, column in (("первый час", "first_hour"), ("остаток дня", "rest")):
        slope = core._boot(gaps["gap"].to_numpy() * gaps[column].to_numpy(), gaps["gap"].to_numpy() ** 2, codes, ndays)
        big = gaps[gaps["gap"].abs() >= 0.01]
        agree, se, _, n = core.cluster_mean(
            (np.sign(big["gap"]) == np.sign(big[column])).astype(float).to_numpy(), big["day"].to_numpy()
        )
        rows.append({"после разрыва": name, "наклон": slope, "доля в сторону разрыва (|gap|≥10 тиков)": agree, "ст.ошибка": se, "дней": n})
    return pd.DataFrame(rows)


def event_study(market: core.Market, channels=BREAKOUT_CHANNELS, horizons=EVENT_HORIZONS) -> pd.DataFrame:
    """Средний ход в сторону пробоя канала за h минут после входа, рублей на контракт, t по кластерам-дням."""
    data = core.arrays(market)
    rows = {}
    for channel in channels:
        entries = core.breakout_entries(data, channel)
        index = np.array([e.index for e in entries])
        sides = np.array([e.side for e in entries])
        moves = core.forward_moves(market, index, sides, horizons)
        row = {"событий": len(entries)}
        for h in horizons:
            values = moves[f"m{h}"].to_numpy()
            ok = ~np.isnan(values)
            mean, _, t, _ = core.cluster_mean(values[ok], moves["day"].to_numpy()[ok])
            row[f"{h} мин"] = (mean, t)
        rows[f"канал {channel}"] = row
    return pd.DataFrame(rows).T


def by_year(market: core.Market) -> pd.DataFrame:
    """Устойчивость по годам: автокорреляция 15-минуток и 30-минутный VR без шума."""
    rows = []
    days = pd.DatetimeIndex(market.grid["day"].unique())
    for year in sorted(set(days.year)):
        part = subset(market, days[days.year == year])
        returns = core.clock_returns(part.grid, 15)
        rho = core.lag_autocorr(returns, 1)
        vr = core.noise_adjusted_vr(part.grid, 5, 6, reps=200)
        rows.append(
            {
                "год": year,
                "дней": part.bars["day"].nunique(),
                "автокорр. 15 мин, лаг 1": rho,
                "VR 30 мин без шума": vr,
                "Хёрст": core.hurst(part.grid),
            }
        )
    return pd.DataFrame(rows)


def by_activity(market: core.Market) -> pd.DataFrame:
    """Те же меры отдельно для спокойных и оживлённых дней (по обороту дня): шум тонкого рынка не должен рождать вывод."""
    volume = market.bars.groupby("day")["volume"].sum()
    cut = volume.median()
    rows = []
    for name, keep in (("оборот ниже медианы", volume[volume <= cut].index), ("оборот выше медианы", volume[volume > cut].index)):
        part = subset(market, keep)
        rows.append(
            {
                "дни": name,
                "автокорр. 15 мин, лаг 1": core.lag_autocorr(core.clock_returns(part.grid, 15), 1),
                "VR 30 мин без шума": core.noise_adjusted_vr(part.grid, 5, 6, reps=200),
                "Хёрст": core.hurst(part.grid),
            }
        )
    return pd.DataFrame(rows)


def run_all(market: core.Market) -> dict[str, object]:
    acf_daily, vr_daily = daily_scale(market)
    return {
        "autocorr": autocorrelations(market),
        "persistence": persistence(market),
        "vr": variance_ratios(market),
        "hurst": core.hurst(market.grid),
        "daily_acf": acf_daily,
        "daily_vr": vr_daily,
        "gaps": gap_memory(market),
        "events": event_study(market),
        "years": by_year(market),
        "activity": by_activity(market),
        "cost": core.round_trip_cost(),
    }


def table(frame: pd.DataFrame, index_name: str = "", digits: int = 3) -> str:
    """Markdown-таблица; кортежи (оценка, нижняя, верхняя) и (среднее, t) печатаются понятно."""

    def cell(value) -> str:
        if isinstance(value, tuple) and len(value) == 3:
            return fmt(*value, digits=digits)
        if isinstance(value, tuple) and len(value) == 2:
            return f"{value[0]:+.1f} (t={value[1]:+.1f})"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    header = [index_name or " "] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for label, row in frame.iterrows():
        lines.append("| " + " | ".join([str(label)] + [cell(v) for v in row]) + " |")
    return "\n".join(lines)
