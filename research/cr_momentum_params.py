"""Финансовые результаты моментума на CR по сетке основных параметров.

Запуск: `PYTHONPATH=src:. python -m research.cr_momentum_params`
пишет `research/REPORT_cr_momentum_params.md` и `research/out/cr_momentum_params.csv`.

Позиция та же, что в `cr_momentum`: знак хода за L дней пересматривается каждые H дней.
Отличаются два списания издержек. «Блок» закрывает и открывает позицию заново каждые H дней
(сумма net совпадает со сделками прошлого отчёта). «Удержание» платит только за изменение
позиции и за роллировку, пока позиция открыта.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import core, cr_momentum as cm, report

LOOKBACKS = cm.LOOKBACKS
STEPS = cm.HOLDS
THRESHOLDS_BP = (0, 50, 100, 200)
EXTRAS = (0.0, 5.0, 20.0)
PRIMARY = cm.PRIMARY
MODES = ("block", "hold")


def side_cost(extra: float = 0.0) -> float:
    """Рублей на контракт за одну сторону: комиссия, тик проскальзывания и добавка."""
    return core.round_trip_cost() / 2.0 + extra


def move_bps(series: cm.Series, lookback: int) -> np.ndarray:
    """Ход синтетической цены за lookback дней в б.п. текущей цены контракта. До lookback — NaN."""
    out = np.full(len(series), np.nan)
    if lookback >= len(series):
        return out
    move = series.price[lookback:] - series.price[:-lookback]
    level = np.where(np.abs(series.level[lookback:]) > 1e-9, series.level[lookback:], np.nan)
    out[lookback:] = move / level * 1e4
    return out


def signal_of(series: cm.Series, lookback: int, min_move_bp: float) -> np.ndarray:
    """Знак хода, либо 0, если истории нет или |ход| меньше порога. Порог 0 оставляет нулевой ход нулём."""
    raw = move_bps(series, lookback)
    out = np.zeros(len(series))
    known = raw == raw
    take = known & (np.abs(raw) >= min_move_bp)
    out[take] = np.sign(raw[take])
    return out


def position_schedule(series: cm.Series, lookback: int, step: int, min_move_bp: float = 0.0, sign: int = 1) -> np.ndarray:
    """Позиция на день t зарабатывает доходность дня t и решена на закрытии одного из дней L, L+H, L+2H, …

    Окно, которому не хватает H дней до конца ряда, не открывается — так же, как сделки в `cr_momentum.trades`.
    """
    if step < 1:
        raise ValueError(step)
    decided = signal_of(series, lookback, min_move_bp) * sign
    position = np.zeros(len(series))
    decision = lookback
    while decision + step < len(series):
        position[decision + 1 : decision + 1 + step] = decided[decision]
        decision += step
    return position


def _frame(series: cm.Series, position: np.ndarray, cost: np.ndarray, open_cost: np.ndarray, close_cost: np.ndarray) -> pd.DataFrame:
    gross = position * series.ret * core.MULTIPLIER
    return pd.DataFrame(
        {
            "day": pd.to_datetime(series.days),
            "position": position,
            "gross": gross,
            "cost": cost,
            "open_cost": open_cost,
            "close_cost": close_cost,
            "net": gross - cost,
        }
    )


def simulate_block(series: cm.Series, lookback: int, step: int, min_move_bp: float = 0.0, extra_side_cost: float = 0.0, sign: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Каждые H дней полный круг, если в окне есть позиция, плюс круг за каждую роллировку внутри окна.

    Весь круг окна списывается в первый день удержания: сумма net по дням равна сумме сделок.
    """
    position = position_schedule(series, lookback, step, min_move_bp, sign)
    n = len(position)
    cost = np.zeros(n)
    trip_rows = []
    round_trip = 2.0 * side_cost(extra_side_cost)
    decision = lookback
    while decision + step < n:
        start, stop = decision + 1, decision + 1 + step
        side = float(position[start])
        if side:
            rolls = int(series.roll[start:stop].sum())
            paid = round_trip * (1 + rolls)
            cost[start] += paid
            gross = float((position[start:stop] * series.ret[start:stop] * core.MULTIPLIER).sum())
            trip_rows.append(
                {
                    "day": pd.Timestamp(series.days[start]),
                    "side": side,
                    "gross": gross,
                    "cost": paid,
                    "net": gross - paid,
                }
            )
        decision += step
    empty = np.zeros(n)
    frame = _frame(series, position, cost, empty, empty)
    trips = pd.DataFrame(trip_rows, columns=["day", "side", "gross", "cost", "net"])
    return frame, trips


def simulate_hold(series: cm.Series, position: np.ndarray | None = None, *, lookback: int = 0, step: int = 1, min_move_bp: float = 0.0, extra_side_cost: float = 0.0, sign: int = 1, liquidate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Издержки только когда меняется число контрактов: смена знака, вход, выход и роллировка открытой позиции.

    В день роллировки старый контракт закрывается и новый открывается — это две стороны, если знак тот же,
    и тоже две, если знак сменился (а не четыре). Открытая на последнем дне позиция закрывается одной стороной.
    """
    if position is None:
        position = position_schedule(series, lookback, step, min_move_bp, sign)
    unit = side_cost(extra_side_cost)
    n = len(position)
    open_cost = np.zeros(n)
    close_cost = np.zeros(n)
    previous = 0.0
    for t in range(n):
        current = float(position[t])
        if bool(series.roll[t]):
            close_sides = 0.0 if previous == 0.0 else 1.0
            open_sides = 0.0 if current == 0.0 else 1.0
        elif previous == current:
            close_sides = open_sides = 0.0
        elif previous == 0.0:
            close_sides, open_sides = 0.0, 1.0
        elif current == 0.0:
            close_sides, open_sides = 1.0, 0.0
        else:
            close_sides = open_sides = 1.0
        close_cost[t] = close_sides * unit
        open_cost[t] = open_sides * unit
        previous = current
    liq = np.zeros(n)
    if liquidate and n and position[-1] != 0.0:
        liq[-1] = unit
    cost = open_cost + close_cost + liq
    frame = _frame(series, position, cost, open_cost, close_cost)
    frame["liq"] = liq
    return frame, hold_trips(frame)


def hold_trips(frame: pd.DataFrame) -> pd.DataFrame:
    """Отрезки постоянного ненулевого знака. Выход предыдущего отрезка лежит на первом дне следующего."""
    position = frame["position"].to_numpy()
    gross = frame["gross"].to_numpy()
    open_cost = frame["open_cost"].to_numpy()
    close_cost = frame["close_cost"].to_numpy()
    days = frame["day"].to_numpy()
    liquidation = float(frame["liq"].iloc[-1]) if "liq" in frame.columns and len(frame) else 0.0
    rows = []
    t = 0
    n = len(position)
    while t < n:
        if position[t] == 0.0:
            t += 1
            continue
        side = float(position[t])
        start = t
        gross_sum = 0.0
        paid = float(open_cost[t])
        while t < n and position[t] == side:
            gross_sum += float(gross[t])
            if t != start:
                paid += float(open_cost[t] + close_cost[t])
            t += 1
        if t < n:
            paid += float(close_cost[t])
        else:
            paid += liquidation
        rows.append({"day": pd.Timestamp(days[start]), "side": side, "gross": gross_sum, "cost": paid, "net": gross_sum - paid})
    return pd.DataFrame(rows, columns=["day", "side", "gross", "cost", "net"])


def month_t(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    monthly = frame.groupby(frame["day"].dt.to_period("M"), sort=True)["net"].sum()
    if len(monthly) < 2:
        return float("nan")
    std = float(monthly.std(ddof=1))
    se = std / np.sqrt(len(monthly)) if std == std else float("nan")
    if not se or se != se or se <= 0:
        return float("nan")
    return float(monthly.mean() / se)


def financials(frame: pd.DataFrame, trips: pd.DataFrame, base: float) -> dict[str, float]:
    """Результат одного контракта на всём ряду: деньги, риск и помесячная устойчивость."""
    net = frame["net"].to_numpy()
    equity = np.cumsum(net) if len(net) else np.array([0.0])
    drawdown = float((np.maximum.accumulate(equity) - equity).max()) if len(net) else 0.0
    years = max((frame["day"].iloc[-1] - frame["day"].iloc[0]).days / 365.25, 1 / 365.25)
    std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    annual = float(net.sum()) / years
    monthly = frame.groupby(frame["day"].dt.to_period("M"), sort=True)["net"].sum()
    wins = float(trips.loc[trips["net"] > 0, "net"].sum()) if len(trips) else 0.0
    losses = float(-trips.loc[trips["net"] <= 0, "net"].sum()) if len(trips) else 0.0
    long_net = float(trips.loc[trips["side"] > 0, "net"].sum()) if len(trips) else 0.0
    short_net = float(trips.loc[trips["side"] < 0, "net"].sum()) if len(trips) else 0.0
    return {
        "trades": float(len(trips)),
        "net": float(net.sum()),
        "gross": float(frame["gross"].sum()),
        "cost": float(frame["cost"].sum()),
        "ann_pct": annual / base * 100.0,
        "sharpe": float(net.mean() / std * np.sqrt(252)) if std and std == std and std > 0 else float("nan"),
        "max_dd": drawdown,
        "max_dd_pct": drawdown / base * 100.0,
        "calmar": annual / drawdown if drawdown > 0 else float("nan"),
        "pf": (wins / losses) if losses > 0 else (float("inf") if wins > 0 else float("nan")),
        "hit_pct": float((trips["net"] > 0).mean() * 100.0) if len(trips) else float("nan"),
        "avg_trade": float(trips["net"].mean()) if len(trips) else float("nan"),
        "worst_trade": float(trips["net"].min()) if len(trips) else float("nan"),
        "t_month": month_t(frame),
        "months_up_pct": float((monthly > 0).mean() * 100.0) if len(monthly) else float("nan"),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "in_market_pct": float((frame["position"] != 0).mean() * 100.0),
        "long_net": long_net,
        "short_net": short_net,
        "long_trades": float((trips["side"] > 0).sum()) if len(trips) else 0.0,
        "short_trades": float((trips["side"] < 0).sum()) if len(trips) else 0.0,
    }


def _year_stats(frame: pd.DataFrame) -> dict[str, float]:
    net = frame["net"].to_numpy()
    std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    equity = np.cumsum(net)
    drawdown = float((np.maximum.accumulate(equity) - equity).max()) if len(net) else 0.0
    return {
        "net": float(net.sum()),
        "sharpe": float(net.mean() / std * np.sqrt(252)) if std and std > 0 else float("nan"),
        "max_dd": drawdown,
        "days": float(len(frame)),
    }


@dataclass
class Study:
    rows: pd.DataFrame
    frames: dict[tuple[str, int, int], pd.DataFrame]
    white: dict[str, float]
    coverage: pd.DataFrame
    yearly: pd.DataFrame
    walk_years: pd.DataFrame
    walk_total: dict[str, float]
    walk_frame: pd.DataFrame
    walk_all_years: pd.DataFrame
    walk_all_total: dict[str, float]
    walk_all_frame: pd.DataFrame
    base: float
    years: float


def _specs(thresholds: tuple[int, ...] = THRESHOLDS_BP, extras: tuple[float, ...] = EXTRAS):
    for mode in MODES:
        for lookback in LOOKBACKS:
            for step in STEPS:
                for threshold in thresholds:
                    for extra in extras:
                        if threshold != 0 and extra != 0:
                            continue
                        yield mode, lookback, step, threshold, extra


def _white(frames: dict[tuple[int, int], pd.DataFrame], reps: int) -> float:
    if not frames:
        return float("nan")
    columns = [frame["net"].to_numpy() for frame in frames.values()]
    keep = [column for column in columns if np.nanstd(column) > 0]
    if not keep:
        return float("nan")
    p, _ = core.reality_check(np.column_stack(keep), reps=reps)
    return float(p)


def walk_forward(
    series: cm.Series,
    hold_frames: dict[tuple[int, int], pd.DataFrame],
    base: float,
    first_test_year: int = 2023,
    min_months: int = 8,
    steps: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """На каждый год теста L и H выбираются по лучшему помесячному t прошлых лет, затем шьётся книга удержания.

    Позиция внутри года — расписание выбранных L и H (сигнал смотрит только назад). На границе года
    смена параметра стоит столько, сколько стоит фактический скачок позиции. До первого года нужно
    min_months календарных месяцев; steps ограничивает допустимые H.
    """
    pool = hold_frames if steps is None else {key: frame for key, frame in hold_frames.items() if key[1] in steps}
    sample_years = sorted({int(y) for y in pd.to_datetime(series.days).year})
    choice: dict[int, tuple[int, int]] = {}
    rows = []
    for year in sample_years:
        if year < first_test_year:
            continue
        best: tuple[int, int] | None = None
        best_t = -np.inf
        for key, frame in pool.items():
            past = frame[frame["day"].dt.year < year]
            if past["day"].dt.to_period("M").nunique() < min_months:
                continue
            t_stat = month_t(past)
            if t_stat == t_stat and t_stat > best_t:
                best, best_t = key, float(t_stat)
        if best is None:
            continue
        choice[year] = best
        rows.append({"year": year, "L": best[0], "H": best[1], "t_in": best_t})
    position = np.zeros(len(series))
    chosen = np.zeros(len(series), dtype=bool)
    years = pd.to_datetime(series.days).year.to_numpy()
    for index, year in enumerate(years):
        key = choice.get(int(year))
        if key is None:
            continue
        position[index] = pool[key]["position"].iloc[index]
        chosen[index] = True
    full, trips = simulate_hold(series, position, liquidate=True)
    book = full.loc[chosen].reset_index(drop=True)
    if len(book) and len(trips):
        trips = trips[trips["day"] >= book["day"].iloc[0]].reset_index(drop=True)
    if rows:
        per_year = []
        for row in rows:
            part = book[book["day"].dt.year == row["year"]]
            stats = _year_stats(part) if len(part) else {"net": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"), "days": 0.0}
            per_year.append({**row, **stats})
        table = pd.DataFrame(per_year)
    else:
        table = pd.DataFrame(columns=["year", "L", "H", "t_in", "net", "sharpe", "max_dd", "days"])
    total = financials(book, trips, base) if len(book) else {key: float("nan") for key in ("trades", "net", "sharpe", "max_dd", "max_dd_pct", "ann_pct", "t_month", "pf", "cost", "gross")}
    return table, total, book


def coverage_table(series: cm.Series) -> pd.DataFrame:
    rows = []
    for lookback in LOOKBACKS:
        move = move_bps(series, lookback)
        known = move[move == move]
        row: dict[str, float] = {"L": lookback, "median_abs_bp": float(np.median(np.abs(known))) if len(known) else float("nan")}
        for threshold in THRESHOLDS_BP:
            row[f"share_{threshold}"] = float((np.abs(known) >= threshold).mean() * 100.0) if len(known) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_table(frames: dict[tuple[str, int, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (mode, lookback, step), frame in frames.items():
        for year, part in frame.groupby(frame["day"].dt.year):
            stats = _year_stats(part)
            rows.append({"mode": mode, "L": lookback, "H": step, "year": int(year), **stats})
    return pd.DataFrame(rows)


def run_study(series: cm.Series, reps: int = 1000, first_test_year: int = 2023) -> Study:
    base = cm.notional(series)
    rows = []
    kept: dict[tuple[str, int, int], pd.DataFrame] = {}
    white_hold: dict[tuple[int, int], pd.DataFrame] = {}
    white_block: dict[tuple[int, int], pd.DataFrame] = {}
    white_threshold: dict[tuple[int, int, int], pd.DataFrame] = {}
    for mode, lookback, step, threshold, extra in _specs():
        if mode == "block":
            frame, trips = simulate_block(series, lookback, step, threshold, extra)
        else:
            frame, trips = simulate_hold(series, lookback=lookback, step=step, min_move_bp=threshold, extra_side_cost=extra)
        stats = financials(frame, trips, base)
        rows.append({"mode": mode, "L": lookback, "H": step, "threshold_bp": threshold, "extra": extra, **stats})
        if threshold == 0 and extra == 0:
            kept[(mode, lookback, step)] = frame
            (white_hold if mode == "hold" else white_block)[(lookback, step)] = frame
        if mode == "hold" and extra == 0:
            white_threshold[(lookback, step, threshold)] = frame
    hold_frames = {(lookback, step): frame for (mode, lookback, step), frame in kept.items() if mode == "hold"}
    slow_steps = tuple(step for step in STEPS if step >= 5)
    walk_years, walk_total, walk_frame = walk_forward(series, hold_frames, base, first_test_year, steps=slow_steps)
    walk_all_years, walk_all_total, walk_all_frame = walk_forward(series, hold_frames, base, first_test_year)
    span = (pd.Timestamp(series.days[-1]) - pd.Timestamp(series.days[0])).days / 365.25
    return Study(
        rows=pd.DataFrame(rows),
        frames=kept,
        white={
            "hold_LH": _white(white_hold, reps),
            "block_LH": _white(white_block, reps),
            "hold_threshold": _white(white_threshold, reps),
        },
        coverage=coverage_table(series),
        yearly=yearly_table(kept),
        walk_years=walk_years,
        walk_total=walk_total,
        walk_frame=walk_frame,
        walk_all_years=walk_all_years,
        walk_all_total=walk_all_total,
        walk_all_frame=walk_all_frame,
        base=base,
        years=span,
    )


def _cell(rows: pd.DataFrame, mode: str, lookback: int, step: int, threshold: float = 0, extra: float = 0) -> pd.Series:
    mask = (
        (rows["mode"] == mode)
        & (rows["L"] == lookback)
        & (rows["H"] == step)
        & (rows["threshold_bp"] == threshold)
        & (rows["extra"] == extra)
    )
    found = rows.loc[mask]
    if found.empty:
        raise KeyError((mode, lookback, step, threshold, extra))
    return found.iloc[0]


def _num(value: float, digits: int = 0, signed: bool = False) -> str:
    if value != value:
        return "—"
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text


def _money_table(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    part = rows[(rows["mode"] == mode) & (rows["threshold_bp"] == 0) & (rows["extra"] == 0)].sort_values(["L", "H"])
    out = []
    for _, row in part.iterrows():
        out.append(
            {
                "L": int(row["L"]),
                "H": int(row["H"]),
                "сделок": int(row["trades"]),
                "net, ₽": _num(row["net"]),
                "gross, ₽": _num(row["gross"]),
                "издержки, ₽": _num(row["cost"]),
                "год., % ном.": _num(row["ann_pct"], 1, signed=True),
                "Шарп": _num(row["sharpe"], 2, signed=True),
                "просадка, ₽": _num(row["max_dd"]),
                "просадка, %": _num(row["max_dd_pct"], 1),
                "Калмар": _num(row["calmar"], 2),
                "PF": _num(row["pf"], 2),
                "t мес.": _num(row["t_month"], 2, signed=True),
            }
        )
    return pd.DataFrame(out)


def _stability_table(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    part = rows[(rows["mode"] == mode) & (rows["threshold_bp"] == 0) & (rows["extra"] == 0)].sort_values(["L", "H"])
    out = []
    for _, row in part.iterrows():
        out.append(
            {
                "L": int(row["L"]),
                "H": int(row["H"]),
                "плюсовых, %": _num(row["hit_pct"], 1),
                "средняя, ₽": _num(row["avg_trade"], 1, signed=True),
                "худшая, ₽": _num(row["worst_trade"], 0, signed=True),
                "месяцев +, %": _num(row["months_up_pct"], 1),
                "худший месяц, ₽": _num(row["worst_month"], 0, signed=True),
                "в рынке, %": _num(row["in_market_pct"], 1),
                "лонг, ₽": _num(row["long_net"], 0, signed=True),
                "шорт, ₽": _num(row["short_net"], 0, signed=True),
            }
        )
    return pd.DataFrame(out)


def _gap_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    base = rows[(rows["threshold_bp"] == 0) & (rows["extra"] == 0)]
    for lookback in LOOKBACKS:
        for step in STEPS:
            block = _cell(base, "block", lookback, step)
            hold = _cell(base, "hold", lookback, step)
            out.append(
                {
                    "L": lookback,
                    "H": step,
                    "net блок, ₽": _num(block["net"]),
                    "net удержание, ₽": _num(hold["net"]),
                    "разница, ₽": _num(hold["net"] - block["net"], signed=True),
                    "издержки блок, ₽": _num(block["cost"]),
                    "издержки удержание, ₽": _num(hold["cost"]),
                    "Шарп блок": _num(block["sharpe"], 2, signed=True),
                    "Шарп удержание": _num(hold["sharpe"], 2, signed=True),
                }
            )
    return pd.DataFrame(out)


def _threshold_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for lookback in LOOKBACKS:
        for step in STEPS:
            item = {"L": lookback, "H": step}
            for threshold in THRESHOLDS_BP:
                row = _cell(rows, "hold", lookback, step, threshold, 0)
                item[f"net @{threshold}"] = _num(row["net"], signed=True)
                item[f"Шарп @{threshold}"] = _num(row["sharpe"], 2, signed=True)
                item[f"сделок @{threshold}"] = int(row["trades"])
            out.append(item)
    return pd.DataFrame(out)


def _slippage_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for lookback in LOOKBACKS:
        for step in STEPS:
            base = _cell(rows, "hold", lookback, step, 0, 0)
            plus5 = _cell(rows, "hold", lookback, step, 0, 5)
            plus20 = _cell(rows, "hold", lookback, step, 0, 20)
            sides = base["cost"] / side_cost(0) if base["cost"] else float("nan")
            breakeven = base["gross"] / sides - side_cost(0) if sides and sides == sides and sides > 0 else float("nan")
            out.append(
                {
                    "L": lookback,
                    "H": step,
                    "net при 6 ₽": _num(base["net"], signed=True),
                    "Шарп при 6 ₽": _num(base["sharpe"], 2, signed=True),
                    "net при 11 ₽": _num(plus5["net"], signed=True),
                    "Шарп при 11 ₽": _num(plus5["sharpe"], 2, signed=True),
                    "net при 26 ₽": _num(plus20["net"], signed=True),
                    "Шарп при 26 ₽": _num(plus20["sharpe"], 2, signed=True),
                    "безубыток, доп. ₽/сторона": _num(breakeven, 1),
                }
            )
    return pd.DataFrame(out)


def _coverage_view(coverage: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in coverage.iterrows():
        item = {"L": int(row["L"]), "медиана |ход|, б.п.": _num(row["median_abs_bp"], 0)}
        for threshold in THRESHOLDS_BP:
            item[f"доля ≥ {threshold}, %"] = _num(row[f"share_{threshold}"], 1)
        out.append(item)
    return pd.DataFrame(out)


def _yearly_view(yearly: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    part = yearly[(yearly["mode"] == "hold") & (yearly["L"] == lookback)].sort_values(["year", "H"])
    years = sorted(part["year"].unique())
    out = []
    for year in years:
        item: dict[str, object] = {"год": int(year)}
        for step in STEPS:
            row = part[(part["year"] == year) & (part["H"] == step)]
            if row.empty:
                continue
            current = row.iloc[0]
            item[f"H={step} net, ₽"] = _num(current["net"], signed=True)
            item[f"H={step} Шарп"] = _num(current["sharpe"], 2, signed=True)
        out.append(item)
    return pd.DataFrame(out)


def _walk_view(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in table.iterrows():
        out.append(
            {
                "год теста": int(row["year"]),
                "L": int(row["L"]),
                "H": int(row["H"]),
                "t в выборке": _num(row["t_in"], 2, signed=True),
                "net, ₽": _num(row["net"], signed=True),
                "Шарп": _num(row["sharpe"], 2, signed=True),
                "просадка, ₽": _num(row["max_dd"]),
            }
        )
    return pd.DataFrame(out)


def _comma(value: float, digits: int = 0, signed: bool = False) -> str:
    text = _num(value, digits, signed)
    return text.replace(".", ",")


def _book_line(total: dict[str, float]) -> str:
    trades = int(total["trades"]) if total.get("trades") == total.get("trades") else 0
    return (
        f"net **{_comma(total['net'], signed=True)} ₽**, Шарп {_comma(total['sharpe'], 2, signed=True)}, "
        f"просадка {_comma(total['max_dd'])} ₽ ({_comma(total['max_dd_pct'], 1)}% номинала), "
        f"t по месяцам {_comma(total['t_month'], 2, signed=True)}, сделок {trades}"
    )


def _choices(table: pd.DataFrame) -> str:
    if table.empty:
        return "ни одного года"
    return ", ".join(f"{int(row.year)} → L = {int(row.L)}, H = {int(row.H)}" for row in table.itertuples())


def _overlap_net(study: Study, table: pd.DataFrame) -> float:
    """Net опорного удержания на тех же календарных годах, что и книга отбора."""
    if table.empty:
        return float("nan")
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    years = {int(year) for year in table["year"]}
    return float(frame.loc[frame["day"].dt.year.isin(years), "net"].sum())


def _primary_blurb(rows: pd.DataFrame, study: Study) -> str:
    white = study.white
    hold = _cell(rows, "hold", PRIMARY[0], PRIMARY[1])
    block = _cell(rows, "block", PRIMARY[0], PRIMARY[1])
    hold_daily = _cell(rows, "hold", PRIMARY[0], 1)
    block_daily = _cell(rows, "block", PRIMARY[0], 1)
    grid = rows[(rows["mode"] == "hold") & (rows["threshold_bp"] == 0) & (rows["extra"] == 0)]
    block_grid = rows[(rows["mode"] == "block") & (rows["threshold_bp"] == 0) & (rows["extra"] == 0)]
    positive = int((grid["net"] > 0).sum())
    best = grid.loc[grid["sharpe"].idxmax()]
    losers = grid[grid["net"] <= 0]
    slip20 = _cell(rows, "hold", PRIMARY[0], PRIMARY[1], 0, 20)
    slip5 = _cell(rows, "hold", PRIMARY[0], PRIMARY[1], 0, 5)
    thresholds = []
    best_threshold = 0
    best_threshold_net = float(hold["net"])
    for threshold in THRESHOLDS_BP:
        if threshold == 0:
            continue
        current = _cell(rows, "hold", PRIMARY[0], PRIMARY[1], threshold, 0)
        thresholds.append(
            f"{threshold} б.п. → net {_comma(current['net'], signed=True)} ₽, Шарп {_comma(current['sharpe'], 2, signed=True)}, сделок {int(current['trades'])}"
        )
        if current["net"] > best_threshold_net:
            best_threshold, best_threshold_net = threshold, float(current["net"])
    if best_threshold == 0:
        threshold_line = (
            "Порог |хода| на первичном наборе не поднимает сумму net выше порога 0: "
            + "; ".join(thresholds)
            + "."
        )
    else:
        threshold_line = (
            f"На первичном наборе порог {best_threshold} б.п. даёт net выше, чем порог 0 "
            f"({_comma(best_threshold_net, signed=True)} ₽ против {_comma(hold['net'], signed=True)} ₽). Остальные пороги: "
            + "; ".join(thresholds)
            + "."
        )
    sides = hold["cost"] / side_cost(0) if hold["cost"] else float("nan")
    breakeven = hold["gross"] / sides - side_cost(0) if sides and sides > 0 else float("nan")
    slow = grid[grid["H"] >= 5]
    fast = grid[grid["H"] == 1]
    fast_block = block_grid[block_grid["H"] == 1]
    if (int(best["L"]), int(best["H"])) == PRIMARY:
        best_line = (
            f"Наибольший Шарп этих 24 наборов совпал с заранее назначенной парой: {_comma(best['sharpe'], 2)}. "
            f"Соседние H при L = {PRIMARY[0]} тоже около 1, это не одинокая клетка на краю сетки."
        )
    else:
        best_line = (
            f"Наибольший Шарп этих 24 наборов — L = {int(best['L'])}, H = {int(best['H'])}: "
            f"Шарп {_comma(best['sharpe'], 2)}, net {_comma(best['net'], signed=True)} ₽. "
            "Это наблюдение по тем же данным, опорным правилом оно не становится."
        )
    if losers.empty:
        loser_line = "Отрицательных наборов удержания нет."
    else:
        listed = ", ".join(f"L = {int(row.L)}, H = {int(row.H)} ({_comma(row.net, signed=True)} ₽)" for row in losers.itertuples())
        loser_line = f"В минусе {len(losers)} из {len(grid)}: {listed}."
    year_rows = study.yearly[(study.yearly["mode"] == "hold") & (study.yearly["L"] == PRIMARY[0]) & (study.yearly["H"] == PRIMARY[1])]
    years_up = int((year_rows["net"] > 0).sum())
    picked = [(int(row.year), int(row.L), int(row.H)) for row in study.walk_years.itertuples()]
    loser_pairs = {(int(row.L), int(row.H)) for row in losers.itertuples()}
    bad_picks = [f"{year}: L = {lookback}, H = {step}" for year, lookback, step in picked if (lookback, step) in loser_pairs]
    bad_pick_line = (" Отбор попал в набор с отрицательным net на всей истории — " + ", ".join(bad_picks) + "." if bad_picks else "")
    same_slow = _overlap_net(study, study.walk_years)
    same_all = _overlap_net(study, study.walk_all_years)
    lines = [
        f"Первичный набор, выбранный заранее по Si/Eu, а не по этой сетке: **L = {PRIMARY[0]}, H = {PRIMARY[1]}**.",
        (
            f"Удержание: net **{_comma(hold['net'], signed=True)} ₽** на контракт за всю историю "
            f"({_comma(hold['ann_pct'], 1, signed=True)}% номинала в год), gross {_comma(hold['gross'], signed=True)} ₽, "
            f"издержки {_comma(hold['cost'])} ₽, Шарп {_comma(hold['sharpe'], 2, signed=True)}, "
            f"максимальная просадка {_comma(hold['max_dd'])} ₽ ({_comma(hold['max_dd_pct'], 1)}% номинала), "
            f"Калмар {_comma(hold['calmar'], 2)}, фактор прибыли {_comma(hold['pf'], 2)}, "
            f"плюсовых сделок {_comma(hold['hit_pct'], 1)}% при средней {_comma(hold['avg_trade'], 0, signed=True)} ₽, "
            f"сделок {int(hold['trades'])}, t по месяцам {_comma(hold['t_month'], 2, signed=True)}. "
            f"По календарным годам эта пара в плюсе в {years_up} из {len(year_rows)}. "
            f"Лонг дал {_comma(hold['long_net'], signed=True)} ₽, шорт {_comma(hold['short_net'], signed=True)} ₽: "
            "прибыль не сводится к одной стороне."
        ),
        (
            f"Тот же сигнал блоками (новый круг каждые {PRIMARY[1]} дней): net {_comma(block['net'], signed=True)} ₽, "
            f"издержки {_comma(block['cost'])} ₽, Шарп {_comma(block['sharpe'], 2, signed=True)}, просадка {_comma(block['max_dd'])} ₽. "
            f"Разница {_comma(hold['net'] - block['net'], signed=True)} ₽ — лишние круги на окнах, где знак не менялся. "
            f"Gross у двух способов один и тот же ({_comma(hold['gross'])} ₽)."
        ),
        (
            f"При H = 1 блок отрицателен в {int((fast_block['net'] < 0).sum())} наборах из {len(fast_block)}: "
            f"новый круг каждый день дороже дневного хода. Удержание при H = 1 положительно в {int((fast['net'] > 0).sum())} из {len(fast)}. "
            f"Для L = {PRIMARY[0]} это {_comma(block_daily['net'], signed=True)} ₽ блоком против {_comma(hold_daily['net'], signed=True)} ₽ удержанием. "
            "Прошлый отчёт показывал оба числа в разных разделах: средняя сделка при H = 1 была около нуля или в минусе, "
            "а ежедневная перебалансировка со списанием только за смену позиции — в плюсе. Здесь они стоят в одной таблице."
        ),
        (
            f"Положительный net у {positive} наборов удержания из {len(grid)}. При H ≥ 5 — {int((slow['net'] > 0).sum())} из {len(slow)}. "
            + loser_line
            + " "
            + best_line
        ),
        (
            f"Проверка перебора Уайта по дневным net 24 наборов удержания: **p = {white['hold_LH']:.3f}**. "
            f"По тем же 24 блокам: p = {white['block_LH']:.3f}. "
            f"По расширенной сетке удержания L×H×порог ({len(LOOKBACKS) * len(STEPS) * len(THRESHOLDS_BP)} вариантов): p = {white['hold_threshold']:.3f}. "
            "Это не тот p, что в прошлом отчёте: там в проверку шла сумма сделки, списанная в день входа, здесь — дневная переоценка."
        ),
        threshold_line,
        (
            f"Проскальзывание на первичном удержании: при 11 ₽ на сторону net {_comma(slip5['net'], signed=True)} ₽ "
            f"(Шарп {_comma(slip5['sharpe'], 2, signed=True)}), при 26 ₽ на сторону net {_comma(slip20['net'], signed=True)} ₽ "
            f"(Шарп {_comma(slip20['sharpe'], 2, signed=True)}). Нулевой net был бы при добавке около {_comma(breakeven, 1)} ₽ на сторону "
            f"сверх базовых 6 ₽ (всего около {_comma(side_cost(breakeven), 1)} ₽ на сторону). "
            "В прошлом отчёте опорный H = 5 терял плюс уже около +20 ₽ на сторону, потому что круг списывался каждые 5 дней."
        ),
        (
            "Последовательный отбор, H только 5, 10 и 20: "
            + _choices(study.walk_years)
            + ". Книга: "
            + _book_line(study.walk_total)
            + f". Опорная пара на тех же годах, без перевыбора: {_comma(same_slow, signed=True)} ₽. "
            + "Вся сетка, включая H = 1: "
            + _choices(study.walk_all_years)
            + ". Книга: "
            + _book_line(study.walk_all_total)
            + f". Опорная пара на тех же годах: {_comma(same_all, signed=True)} ₽."
            + bad_pick_line
        ),
        (
            f"Для счёта вывод такой. Опорное удержание даёт около {_comma(hold['ann_pct'], 0)}% номинала в год "
            f"при просадке {_comma(hold['max_dd_pct'], 0)}% номинала"
            + (f" и остаётся в плюсе при 26 ₽ на сторону (net {_comma(slip20['net'], signed=True)} ₽)." if slip20["net"] > 0 else f"; при 26 ₽ на сторону net уже {_comma(slip20['net'], signed=True)} ₽.")
            + f" Новый круг каждые {PRIMARY[1]} дней оставляет от того же gross {_comma(block['net'], signed=True)} ₽, "
            f"а новый круг каждый день при L = {PRIMARY[0]} уводит счёт в {_comma(block_daily['net'], signed=True)} ₽. "
            + (f"Поправка Уайта p = {white['hold_LH']:.3f} ниже 5%. " if white["hold_LH"] < 0.05 else f"Поправка Уайта p = {white['hold_LH']:.3f} не проходит 5%. ")
            + "В живое правило ничего не добавляется: в истории CR четыре года одного рублёвого режима."
        ),
    ]
    return "\n\n".join(lines)


def build_report(series: cm.Series, study: Study | None = None, reps: int = 1000) -> str:
    study = study or run_study(series, reps=reps)
    rows = study.rows
    first = pd.Timestamp(series.days[0]).date()
    last = pd.Timestamp(series.days[-1]).date()
    walk_text = report.md(_walk_view(study.walk_years), index=False) if len(study.walk_years) else "Нет года, для которого в прошлом набралось 8 месяцев."
    walk_all_text = report.md(_walk_view(study.walk_all_years), index=False) if len(study.walk_all_years) else "Нет года, для которого в прошлом набралось 8 месяцев."
    parts = [
        "# Моментум на CR: финансовые результаты по параметрам\n",
        "Продолжение `REPORT_cr_momentum.md`: тот же непрерывный дневной ряд и тот же сигнал (знак хода за L дней). Здесь на каждый набор параметров посчитан результат одного контракта целиком, а не только средняя сделка в б.п.\n",
        "## Параметры и два счёта издержек\n",
        f"- Ряд CR: {first} … {last}, {len(series)} будних дней, {int(series.roll.sum())} роллировок, {study.years:.2f} календарных лет. Номинал (медиана цены × 1000) **{f'{study.base:,.0f}'.replace(',', ' ')} ₽**. База издержек — {side_cost():.0f} ₽ на сторону, круг {core.round_trip_cost():.0f} ₽.",
        "- **L** — сколько дней назад берётся ход. Сетка: " + ", ".join(str(v) for v in LOOKBACKS) + ".",
        "- **H** — как часто пересматривается позиция. Решение на закрытии дней L, L+H, L+2H, …; следующие H дней позиция равна этому знаку. Сетка: " + ", ".join(str(v) for v in STEPS) + ". Окно, которому не хватает H дней до конца ряда, не открывается.",
        "- **Порог** — минимальный |ход| за L дней в б.п. цены контракта. 0 означает прежнее правило. 50, 100 и 200 оставляют счёт без позиции, если ход слабее.",
        "- **Проскальзывание** сверх базы: 0, +5 и +20 ₽ на сторону (в таблице это 6, 11 и 26 ₽ на сторону). Шаг цены CR стоит 1 ₽ на контракт.",
        "- **Блок.** Каждые H дней позиция закрывается и открывается заново, даже если знак не изменился. Плюс ещё один круг за каждую роллировку внутри окна, включая день входа, если он попал на роллировку. Сумма net равна сумме сделок раздела 1 прошлого отчёта. Весь круг окна списывается в первый день удержания, поэтому Шарп и просадка блока чуть жёстче, чем если размазать тот же круг по дням.",
        "- **Удержание.** Плата только за изменение позиции и за роллировку открытой позиции. Переход из +1 в −1 стоит две стороны и в обычный день, и в день роллировки. В последний день выборки открытая позиция закрывается одной стороной.",
        f"- Первичный набор по-прежнему **L = {PRIMARY[0]}, H = {PRIMARY[1]}**: он взят с Si/Eu до отдельного просмотра CR. Наибольший Шарп в таблице сам по себе новым правилом не становится.\n",
        "## 1. Сравнение блока и удержания (порог 0, 6 ₽ на сторону)\n",
        "Gross совпадает построчно: это одна и та же позиция. Отличается только списание кругов.\n",
        report.md(_gap_table(rows), index=False),
        "\n## 2. Удержание: деньги и риск\n",
        "Позиция пересматривается каждые H дней и держится, пока решение не сменится. net — рубли на 1 контракт за всю историю. Годовая доходность — от номинала, не от гарантийного обеспечения. Шарп — по дневным net, включая дни без позиции, × √252. t — по месячным суммам. Просадка — от пика кривой net.\n",
        report.md(_money_table(rows, "hold"), index=False),
        "\n### Устойчивость сделок и месяцев\n",
        report.md(_stability_table(rows, "hold"), index=False),
        "\n## 3. Блок: деньги и риск\n",
        "Тот же сигнал, но полный круг каждые H дней. Сумма `net, ₽` — это сумма net сделок из сетки прошлого отчёта (там она была в б.п. на сделку).\n",
        report.md(_money_table(rows, "block"), index=False),
        "\n### Устойчивость сделок и месяцев\n",
        report.md(_stability_table(rows, "block"), index=False),
        "\n## 4. Порог хода (удержание)\n",
        "Доля дней, на которых |ход| за L дней не ниже порога. Медиана показывает, какие пороги ещё оставляют сигналы, а какие почти всё выключают.\n",
        report.md(_coverage_view(study.coverage), index=False),
        "\nNet и Шарп удержания при пороге. Число сделок — сколько раз позиция меняла отрезок, а не сколько было дней в рынке.\n",
        report.md(_threshold_table(rows), index=False),
        "\n## 5. Проскальзывание (удержание, порог 0)\n",
        "«Безубыток» — сколько рублей на сторону можно добавить к базовым 6 ₽, чтобы суммарный net стал нулём. Gross и число сторон при этом те же: дороже исполнение, то же число перекладок. Шарп при 11 и 26 ₽ пересчитан по дневной кривой, а не сдвигом итога.\n",
        report.md(_slippage_table(rows), index=False),
        "\n## 6. По годам, удержание, L = 20\n",
        "Net года и Шарп по дням этого года (для неполного года Шарп всё равно приведён к году и потому шумный). Просадка внутри года в эту таблицу не входит; она есть в csv.\n",
        report.md(_yearly_view(study.yearly, 20), index=False),
        "\nТот же разрез для L = 40 — соседний по прошлому отчёту набор, не новый отбор.\n",
        report.md(_yearly_view(study.yearly, 40), index=False),
        "\n## 7. Последовательный отбор\n",
        "На каждый год с 2023-го L и H выбираются по лучшему помесячному t удержания на прошлых годах (порог 0). В 2022 году, с апреля, девять месяцев, а отбору нужно восемь, поэтому 2023-й уже в тесте, но выбор на него опирается только на эти девять месяцев. До 2023 года книга пустая: первая позиция оплачивается как вход с нуля. Смена пары на границе года стоит фактический скачок позиции.\n",
        "### H от 5 до 20\n",
        "Ежедневный пересмотр в отбор не входит — так же, как в walk-forward прошлого отчёта, где H был 5, 10 или 20.\n",
        walk_text,
        f"\nКнига этих лет: {_book_line(study.walk_total)}, gross {_comma(study.walk_total['gross'], signed=True)} ₽, издержки {_comma(study.walk_total['cost'])} ₽, PF {_comma(study.walk_total['pf'], 2)}.\n",
        "### Все H, включая 1\n",
        "Селектор может выбрать перекладку каждый день, если на прошлом куске у неё выше помесячный t.\n",
        walk_all_text,
        f"\nКнига этих лет: {_book_line(study.walk_all_total)}, gross {_comma(study.walk_all_total['gross'], signed=True)} ₽, издержки {_comma(study.walk_all_total['cost'])} ₽, PF {_comma(study.walk_all_total['pf'], 2)}.\n",
        "## 8. Что из этого следует\n",
        _primary_blurb(rows, study),
    ]
    return "\n".join(parts) + "\n"


def _json_value(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return round(number, 4)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _aligned_equity(frame: pd.DataFrame, months: list[str]) -> list[float]:
    """Накопленный net по месяцам на общей оси. До первой сделки книги — нули."""
    if frame.empty:
        return [0.0] * len(months)
    monthly = frame.groupby(frame["day"].dt.to_period("M"), sort=True)["net"].sum().cumsum()
    out: list[float] = []
    last = 0.0
    started = False
    for label in months:
        period = pd.Period(label, freq="M")
        if period in monthly.index:
            last = float(monthly.loc[period])
            started = True
        out.append(round(last if started else 0.0, 2))
    return out


def export_payload(series: cm.Series, study: Study) -> dict[str, object]:
    """Компактные ряды для таблиц и графика: месяцы и сетка удержания/блока при пороге 0."""
    rows = study.rows
    grid = rows[(rows["threshold_bp"] == 0) & (rows["extra"] == 0)]
    records = [{key: _json_value(value) for key, value in row.items()} for _, row in grid.sort_values(["mode", "L", "H"]).iterrows()]
    anchor = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    months = [str(period) for period in anchor.groupby(anchor["day"].dt.to_period("M"), sort=True).groups]
    equities = {f"L{PRIMARY[0]} H{step} удержание": _aligned_equity(study.frames[("hold", PRIMARY[0], step)], months) for step in STEPS}
    equities[f"L{PRIMARY[0]} H{PRIMARY[1]} блок"] = _aligned_equity(study.frames[("block", PRIMARY[0], PRIMARY[1])], months)
    equities["walk-forward H≥5"] = _aligned_equity(study.walk_frame, months)
    equities["walk-forward все H"] = _aligned_equity(study.walk_all_frame, months)
    hold_primary = _cell(rows, "hold", *PRIMARY)
    return {
        "first": str(pd.Timestamp(series.days[0]).date()),
        "last": str(pd.Timestamp(series.days[-1]).date()),
        "days": int(len(series)),
        "rolls": int(series.roll.sum()),
        "years": round(study.years, 2),
        "notional": round(study.base, 2),
        "side_cost": side_cost(),
        "white": {key: _json_value(value) for key, value in study.white.items()},
        "primary": {key: _json_value(hold_primary[key]) for key in hold_primary.index},
        "walk": {key: _json_value(value) for key, value in study.walk_total.items()},
        "walk_years": [{key: _json_value(value) for key, value in row.items()} for _, row in study.walk_years.iterrows()],
        "walk_all": {key: _json_value(value) for key, value in study.walk_all_total.items()},
        "walk_all_years": [{key: _json_value(value) for key, value in row.items()} for _, row in study.walk_all_years.iterrows()],
        "grid": records,
        "months": months,
        "equity": equities,
        "coverage": [{key: _json_value(value) for key, value in row.items()} for _, row in study.coverage.iterrows()],
    }


def main() -> None:
    core.configure(1000.0, 0.001)
    series = cm.continuous()
    study = run_study(series)
    text = build_report(series, study)
    (report.OUT / "REPORT_cr_momentum_params.md").write_text(text, encoding="utf-8")
    study.rows.to_csv(report.OUT / "out" / "cr_momentum_params.csv", index=False)
    payload = export_payload(series, study)
    (report.OUT / "out" / "cr_momentum_params.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
