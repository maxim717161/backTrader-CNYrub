"""Прогон сетки правил с защитой от самообмана.

Для каждого варианта считаются сделки (с комиссией и проскальзыванием), затем:
1. период делится на обучающий и проверочный по дате SPLIT — выбор делается только по первому;
2. «проверка перебора» (White) говорит, насколько лучший вариант из сетки отличается от
   лучшего, который дал бы случайный шум при таком же числе попыток;
3. лучший по обучению вариант отдельно показывается на проверочном периоде и по годам.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import core
from research.families import Spec, build

SPLIT = pd.Timestamp("2024-07-01")


@dataclass
class GridResult:
    table: pd.DataFrame
    daily: np.ndarray
    days: np.ndarray
    trades: dict[str, pd.DataFrame]
    labels: list[str]


def run_grid(specs: list[Spec], market: core.Market, data: dict[str, np.ndarray] | None = None, verbose: bool = False) -> GridResult:
    data = data if data is not None else core.arrays(market)
    days = np.array(sorted(market.bars["day"].unique()))
    rows = []
    columns = []
    trades_by_label: dict[str, pd.DataFrame] = {}
    for position, spec in enumerate(specs):
        entries = build(spec, market, data)
        trades = core.run_entries(data, entries, flatten_day=spec.flatten_day)
        label = spec.label()
        trades_by_label[label] = trades
        columns.append(core.daily_pnl(trades, days))
        row = {"вариант": label}
        for name, part in (
            ("всё", trades),
            ("обучение", trades[trades["day"] < SPLIT]),
            ("проверка", trades[trades["day"] >= SPLIT]),
        ):
            stats = core.trade_stats(part)
            row[f"{name}: сделок"] = stats["n"]
            row[f"{name}: net"] = stats["net"]
            row[f"{name}: t"] = stats["t"]
            if name == "всё":
                row["всё: gross"] = stats["gross"]
                row["всё: доля плюсовых"] = stats["hit"]
        rows.append(row)
        if verbose:
            print(position + 1, len(specs), label, round(row["всё: net"], 1) if row["всё: net"] == row["всё: net"] else "—", flush=True)
    table = pd.DataFrame(rows).set_index("вариант")
    return GridResult(table=table, daily=np.column_stack(columns), days=days, trades=trades_by_label, labels=list(table.index))


def selection(result: GridResult) -> dict[str, object]:
    """Выбор по обучению, проверка на отложенном периоде и поправка на перебор."""
    table = result.table
    in_sample = result.days < np.datetime64(SPLIT)
    live = result.table["обучение: сделок"].to_numpy() >= 30
    daily_in = result.daily[in_sample][:, live]
    p_is, t_is = core.reality_check(daily_in)
    p_all, t_all = core.reality_check(result.daily[:, live])
    candidates = table[live]
    best = candidates["обучение: t"].idxmax()
    best_all = candidates["всё: t"].idxmax()
    trades = result.trades[best]
    years = []
    for year, part in trades.groupby(pd.to_datetime(trades["day"]).dt.year):
        stats = core.trade_stats(part)
        years.append({"год": year, "сделок": stats["n"], "net": stats["net"], "t": stats["t"]})
    return {
        "variants": int(live.sum()),
        "p_in_sample": p_is,
        "p_all": p_all,
        "best_in_sample": best,
        "best_all": best_all,
        "best_years": pd.DataFrame(years).set_index("год") if years else pd.DataFrame(),
        "oos_positive": int((candidates["проверка: net"] > 0).sum()),
        "oos_significant": int((candidates["проверка: t"] > 2).sum()),
        "is_positive": int((candidates["обучение: net"] > 0).sum()),
        "is_significant": int((candidates["обучение: t"] > 2).sum()),
        "gross_positive": int((candidates["всё: gross"] > 0).sum()),
        "net_positive": int((candidates["всё: net"] > 0).sum()),
    }


def grid_table(result: GridResult, order_by: str = "обучение: t", top: int | None = None) -> pd.DataFrame:
    frame = result.table.sort_values(order_by, ascending=False)
    return frame.head(top) if top else frame


def detail(spec: Spec, market: core.Market, data: dict[str, np.ndarray]) -> dict[str, object]:
    """Разрез лучшего варианта: покупки и продажи отдельно и «зеркало» (то же правило с обратным знаком)."""
    from dataclasses import replace

    trades = core.run_entries(data, build(spec, market, data), flatten_day=spec.flatten_day)
    sides = []
    for side, name in ((1, "покупки"), (-1, "продажи")):
        stats = core.trade_stats(trades[trades["side"] == side])
        sides.append({"сторона": name, "сделок": stats["n"], "net": stats["net"], "t": stats["t"], "gross": stats["gross"]})
    mirror_spec = replace(spec, sign=-spec.sign)
    mirror = core.run_entries(data, build(mirror_spec, market, data), flatten_day=spec.flatten_day)
    mirror_stats = core.trade_stats(mirror)
    reasons = trades["reason"].value_counts().rename_axis("выход").reset_index(name="сделок").set_index("выход")
    return {
        "sides": pd.DataFrame(sides).set_index("сторона"),
        "mirror_label": mirror_spec.label(),
        "mirror": mirror_stats,
        "reasons": reasons,
        "bars_median": float(trades["bars"].median()) if len(trades) else float("nan"),
    }
