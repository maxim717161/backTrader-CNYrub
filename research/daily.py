"""Моментум на дневном масштабе: тянется ли движение последних недель на следующие недели.

На длинной истории Si и Eu внутридневные правила с удержанием через ночь оказались прибыльны
одинаково в обе стороны (зеркало не хуже правила) — признак того, что прибыль даёт не сигнал входа,
а медленный тренд, который ловит широкий стоп. Здесь это проверяется прямо: знак доходности за L
дней определяет позицию на H дней, без стопов и внутридневных деталей. Сделки не пересекаются
(шаг H дней внутри контракта), вход и выход по цене закрытия дня, издержки — тот же круг в рублях.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from research import core

LOOKBACKS = (5, 10, 20, 40)
CRISES = (("2020-02-15", "2020-04-30"), ("2022-02-15", "2022-04-30"))
HOLDS = (5, 10, 20)


def daily_closes(market: core.Market) -> pd.DataFrame:
    last = market.bars.groupby(["contract", "day"])["close"].last().reset_index()
    return last.sort_values(["contract", "day"]).reset_index(drop=True)


def tsmom(closes: pd.DataFrame, lookback: int, hold: int, sign: int = 1, fixed_side: int = 0, delay: int = 0) -> pd.DataFrame:
    """Сделки: на закрытии дня i позиция sign·знак(цена[i] − цена[i−L]) на H дней, не пересекаясь.

    fixed_side=±1 вместо сигнала — всегда в одну сторону (контроль дрейфа). delay — на сколько
    дней позже закрытия сигнала входим (0 — по закрытию того же дня)."""
    rows = []
    cost = core.round_trip_cost()
    for contract, part in closes.groupby("contract", sort=False):
        price = part["close"].to_numpy()
        day = part["day"].to_numpy()
        i = lookback
        while i + delay + hold < len(price):
            if fixed_side:
                side = fixed_side
            else:
                move = price[i] - price[i - lookback]
                side = sign * int(np.sign(move))
            if side:
                gross = side * (price[i + delay + hold] - price[i + delay]) * core.MULTIPLIER
                rows.append({"day": day[i + delay], "contract": contract, "side": side, "gross": gross, "net": gross - cost})
            i += hold
    return pd.DataFrame(rows, columns=["day", "contract", "side", "gross", "net"])


def month_cluster_stats(trades: pd.DataFrame, column: str = "net") -> tuple[float, float, int]:
    """Средняя сделка, t с кластеризацией по месяцам (соседние сделки делят движение), число сделок."""
    if trades.empty:
        return float("nan"), float("nan"), 0
    months = pd.to_datetime(trades["day"]).dt.to_period("M").astype(str).to_numpy()
    mean, _, t, n = core.cluster_mean(trades[column].to_numpy(), months)
    return mean, t, n


def grid(market: core.Market, split: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Сетка L×H: net на сделку в б.п. номинала и t отдельно до/после split, плюс p проверки перебора."""
    closes = daily_closes(market)
    notional = _notional(market)
    days = np.array(sorted(closes["day"].unique()))
    rows, columns = [], []
    for lookback, hold in itertools.product(LOOKBACKS, HOLDS):
        trades = tsmom(closes, lookback, hold)
        trades = trades.assign(day=pd.to_datetime(trades["day"]))
        columns.append(core.daily_pnl(trades, days))
        row = {"L, дней": lookback, "H, дней": hold}
        for name, part in (("до", trades[trades["day"] < split]), ("после", trades[trades["day"] >= split]), ("всё", trades)):
            mean, t, n = month_cluster_stats(part)
            row[f"{name}: сделок"] = n
            row[f"{name}: net, б.п."] = mean / notional * 1e4 if n else float("nan")
            row[f"{name}: t"] = t
        rows.append(row)
    p, _ = core.reality_check(np.column_stack(columns), reps=1000)
    return pd.DataFrame(rows), float(p)


def drift_controls(market: core.Market) -> pd.DataFrame:
    """Всегда в одну сторону на H дней: дрейф самого инструмента, который моментум не должен путать с сигналом."""
    closes = daily_closes(market)
    notional = _notional(market)
    rows = []
    for hold in HOLDS:
        for name, side in (("всегда покупка", 1), ("всегда продажа", -1)):
            trades = tsmom(closes, 20, hold, fixed_side=side)
            mean, t, n = month_cluster_stats(trades)
            rows.append({"позиция": name, "H, дней": hold, "сделок": n, "net, б.п.": mean / notional * 1e4, "t": t})
    return pd.DataFrame(rows)


def _notional(market: core.Market) -> float:
    return float(market.bars["close"].median()) * core.MULTIPLIER


def robustness(market: core.Market, lookback: int = 20) -> pd.DataFrame:
    """Устойчивость моментума по L=20: без кризисных эпизодов, со входом на день позже, по годам."""
    closes = daily_closes(market)
    notional = _notional(market)
    rows = []
    for hold in HOLDS:
        base = tsmom(closes, lookback, hold)
        base = base.assign(day=pd.to_datetime(base["day"]))
        crisis = np.zeros(len(base), dtype=bool)
        for start, end in CRISES:
            crisis |= ((base["day"] >= start) & (base["day"] <= end)).to_numpy()
        late = tsmom(closes, lookback, hold, delay=1)
        flipped = tsmom(closes, lookback, hold, sign=-1)
        years = base.groupby(base["day"].dt.year)["net"].mean()
        row = {"H, дней": hold}
        for name, part in (("все сделки", base), ("без кризисов 2020 и 2022", base[~crisis]), ("вход на день позже", late), ("зеркало (против сигнала)", flipped)):
            mean, t, n = month_cluster_stats(part)
            row[name] = f"{mean / notional * 1e4:+.1f} б.п. (t={t:+.1f}, n={n})" if n else "—"
        row["лет в плюсе"] = f"{int((years > 0).sum())} из {len(years)}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("H, дней")


def by_year(market: core.Market, lookback: int = 20, hold: int = 5) -> pd.DataFrame:
    closes = daily_closes(market)
    notional = _notional(market)
    trades = tsmom(closes, lookback, hold)
    trades = trades.assign(day=pd.to_datetime(trades["day"]))
    rows = []
    for year, part in trades.groupby(trades["day"].dt.year):
        mean, t, n = month_cluster_stats(part)
        rows.append({"год": year, "сделок": n, "net, б.п.": mean / notional * 1e4, "t": t})
    return pd.DataFrame(rows).set_index("год")
