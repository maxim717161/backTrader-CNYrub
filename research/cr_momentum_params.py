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


# Доля риска та же, что у короткого окна в backtest.py: запланированный убыток не больше 10% счёта.
# У этого моментума нет стопа, поэтому убыток — просадка одного контракта.
ACCOUNT_CASH = 100_000.0
ACCOUNT_RISK = 0.10
# ISS FORTS на 2026-10-21: CRZ6, INITIALMARGIN 1200.40 при PREVSETTLEPRICE 12.532.
EXCHANGE_MARGIN_DAY = "2026-10-21"
EXCHANGE_MARGIN_RUB = 1200.40
EXCHANGE_SETTLE = 12.532
MARGIN_RATE = EXCHANGE_MARGIN_RUB / (EXCHANGE_SETTLE * 1000.0)
ACCOUNT_GRID = (1, 2, 4, 5, 8, 10, 20, 40)
# Плечо — номинал / счёт. Формула контрактов та же, что в минутном движке: floor(плечо × счёт / номинал).
LEVERAGES = (1, 2, 3, 4, 5, 6, 7, 8)
# Плечо и фильтр, которые оставляем. Остальные фильтры — сравнение, не новый отбор.
CHOSEN_LEVERAGE = 3
FILTER_SHOCK = 2.0
PRICE_PAUSE = f"пауза после {FILTER_SHOCK:.0f}σ"
# Счёт до широкого дня. На нём смотрелись прямота и остальные правила раздела 15.
PAUSE_NO_CLIMAX = "пауза и без кульминации"
CHOSEN_FILTER = "пауза, без кульминации и без широкого дня"
FILTER_VOL_MULTIPLE = 1.5
# Громкость — дневной объём активного контракта. Ориентир объявлен заранее: медиана предыдущих 20 дней, всплеск — 2 медианы.
VOLUME_WINDOW = 20
VOLUME_SPIKE = 2.0
# Прямота — |чистый ход за L дней| / сумма модулей дневных ходов. Коридор взят из 30-минутного окна.
# 0,25 — округление 1/sqrt(20), масштаб случайного блуждания за L дней, не подбор по счёту CR.
STRAIGHT_LOW = 0.15
STRAIGHT_HIGH = 0.5
STRAIGHT_FLOOR = 0.25
# Доля дней окна с тем же знаком, что у сигнала. Для L = 20 это 12 дней. Не подбор по счёту.
DAY_SHARE = 0.60
# Широкий день — диапазон high–low больше стольких медиан предыдущих 20 дней. Та же отсечка, что у объёма.
RANGE_SPIKE = 2.0


def contracts_for_drawdown(max_dd: float, cash: float = ACCOUNT_CASH, risk: float = ACCOUNT_RISK) -> int:
    """Сколько контрактов, чтобы просадка max_dd на контракт забрала не больше risk от счёта."""
    if max_dd != max_dd or max_dd <= 0 or cash <= 0 or risk <= 0:
        return 0
    return int((risk * cash) // max_dd)


def account_curve(
    frame: pd.DataFrame,
    level: np.ndarray,
    contracts: int,
    cash: float = ACCOUNT_CASH,
    margin_rate: float = MARGIN_RATE,
) -> pd.DataFrame:
    """Счёт с постоянным числом контрактов. PnL и ГО одного контракта умножаются на это число.

    ГО дня — та же доля номинала, что у биржи на CRZ6 в EXCHANGE_MARGIN_DAY. В день без позиции залог не блокируется.
    Свободные деньги — equity минус ГО. Ноль и ниже — маржин-колл: биржа закрыла бы позицию, кривая дальше этого не учитывает.
    """
    if contracts < 0:
        raise ValueError(contracts)
    if len(level) != len(frame):
        raise ValueError("длина цены и дневной кривой не совпадает")
    lots = int(contracts)
    net = frame["net"].to_numpy(dtype=float) * lots
    equity = cash + (np.cumsum(net) if len(net) else np.array([cash], dtype=float))
    peak = np.maximum.accumulate(equity)
    held = frame["position"].to_numpy() != 0.0
    margin = np.where(held, margin_rate * np.asarray(level, dtype=float) * core.MULTIPLIER * lots, 0.0)
    return pd.DataFrame(
        {
            "day": frame["day"].to_numpy(),
            "contracts": lots,
            "net": net,
            "equity": equity,
            "drawdown": peak - equity,
            "margin": margin,
            "free": equity - margin,
        }
    )


def account_summary(curve: pd.DataFrame, cash: float = ACCOUNT_CASH) -> dict[str, float]:
    equity = curve["equity"].to_numpy(dtype=float)
    net = curve["net"].to_numpy(dtype=float)
    drawdown = curve["drawdown"].to_numpy(dtype=float)
    worst = int(drawdown.argmax()) if len(drawdown) else 0
    dd = float(drawdown[worst]) if len(drawdown) else 0.0
    peak = float(equity[worst] + dd) if len(equity) else cash
    std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    return {
        "contracts": float(curve["contracts"].iloc[0]) if len(curve) else 0.0,
        "end": float(equity[-1]) if len(equity) else cash,
        "net": float(net.sum()) if len(net) else 0.0,
        "max_dd": dd,
        "max_dd_pct_start": dd / cash * 100.0 if cash else float("nan"),
        "max_dd_pct_peak": dd / peak * 100.0 if peak else float("nan"),
        "min_equity": float(equity.min()) if len(equity) else cash,
        "min_free": float(curve["free"].min()) if len(curve) else cash,
        "margin_call": float((curve["free"] < 0).any()) if len(curve) else 0.0,
        "ruined": float((equity <= 0).any()) if len(equity) else 0.0,
        "sharpe": float(net.mean() / std * np.sqrt(252)) if std and std == std and std > 0 else float("nan"),
    }


def max_contracts_by_margin(
    frame: pd.DataFrame,
    level: np.ndarray,
    cash: float = ACCOUNT_CASH,
    margin_rate: float = MARGIN_RATE,
) -> int:
    """Самое большое целое число контрактов, при котором свободные деньги каждый день строго выше нуля.

    Свободные деньги линейны по числу контрактов, поэтому предел считается по дням, а не перебором.
    """
    if frame.empty or cash <= 0:
        return 0
    equity_one = np.cumsum(frame["net"].to_numpy(dtype=float))
    margin_one = margin_rate * np.asarray(level, dtype=float) * core.MULTIPLIER
    held = frame["position"].to_numpy() != 0.0
    coefficient = equity_one - np.where(held, margin_one, 0.0)
    negative = coefficient < -1e-9
    if not negative.any():
        held_margin = margin_one[held]
        if held_margin.size == 0 or float(held_margin.min()) <= 0:
            return 0
        return int(cash // float(held_margin.min()))
    return max(int(np.floor(np.min(cash / -coefficient[negative]) - 1e-9)), 0)


def drawdown_by_year(frame: pd.DataFrame, cash: float = ACCOUNT_CASH, risk: float = ACCOUNT_RISK) -> pd.DataFrame:
    """На конец каждого года — просадка одного контракта, уже видная к этой дате, и число контрактов по доле риска."""
    rows = []
    net = frame["net"].to_numpy(dtype=float)
    equity = np.cumsum(net)
    drawdown = np.maximum.accumulate(equity) - equity
    seen = np.maximum.accumulate(drawdown)
    years = frame["day"].dt.year.to_numpy()
    for year in sorted(set(years.tolist())):
        last = int(np.where(years == year)[0][-1])
        unit = float(seen[last])
        rows.append({"year": int(year), "max_dd": unit, "contracts": contracts_for_drawdown(unit, cash, risk)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AccountBook:
    chosen: int
    per_contract_dd: float
    by_year: pd.DataFrame
    grid: pd.DataFrame
    chosen_curve: pd.DataFrame
    max_by_margin: int
    dd_from: str
    dd_to: str
    low_day: str
    worst_day: str
    worst_day_pnl: float
    go_min: float
    go_max: float


def describe_account(
    frame: pd.DataFrame,
    level: np.ndarray,
    cash: float = ACCOUNT_CASH,
    risk: float = ACCOUNT_RISK,
    margin_rate: float = MARGIN_RATE,
    grid: tuple[int, ...] = ACCOUNT_GRID,
) -> AccountBook:
    net = frame["net"].to_numpy(dtype=float)
    equity = np.cumsum(net) if len(net) else np.array([0.0])
    drawdown = np.maximum.accumulate(equity) - equity
    trough = int(drawdown.argmax()) if len(drawdown) else 0
    unit_dd = float(drawdown[trough]) if len(drawdown) else 0.0
    peak_at = int(np.where(equity[: trough + 1] == equity[: trough + 1].max())[0][-1]) if len(equity) else 0
    low = int(equity.argmin()) if len(equity) else 0
    worst = int(net.argmin()) if len(net) else 0
    margin_one = margin_rate * np.asarray(level, dtype=float) * core.MULTIPLIER
    chosen = contracts_for_drawdown(unit_dd, cash, risk)
    counts = list(dict.fromkeys([*grid, chosen, max_contracts_by_margin(frame, level, cash, margin_rate)]))
    rows = []
    curves: dict[int, pd.DataFrame] = {}
    for count in counts:
        if count < 0:
            continue
        curve = account_curve(frame, level, count, cash, margin_rate)
        curves[count] = curve
        rows.append(account_summary(curve, cash))
    table = pd.DataFrame(rows)
    return AccountBook(
        chosen=chosen,
        per_contract_dd=unit_dd,
        by_year=drawdown_by_year(frame, cash, risk),
        grid=table,
        chosen_curve=curves.get(chosen, account_curve(frame, level, chosen, cash, margin_rate)),
        max_by_margin=max_contracts_by_margin(frame, level, cash, margin_rate),
        dd_from=str(pd.Timestamp(frame["day"].iloc[peak_at]).date()) if len(frame) else "",
        dd_to=str(pd.Timestamp(frame["day"].iloc[trough]).date()) if len(frame) else "",
        low_day=str(pd.Timestamp(frame["day"].iloc[low]).date()) if len(frame) else "",
        worst_day=str(pd.Timestamp(frame["day"].iloc[worst]).date()) if len(frame) else "",
        worst_day_pnl=float(net[worst]) if len(net) else 0.0,
        go_min=float(margin_one.min()) if len(margin_one) else 0.0,
        go_max=float(margin_one.max()) if len(margin_one) else 0.0,
    )


def _target_lots(sign: float, equity: float, notional: float, leverage: float, margin_rate: float) -> int:
    """Контракты со знаком. Ноль, если нет сигнала, номинала или места под ГО на входе."""
    if sign == 0.0 or equity <= 0 or notional <= 1e-9 or leverage <= 0:
        return 0
    raw = int(leverage * equity // notional)
    room = int(equity // (margin_rate * notional)) if margin_rate > 0 else raw
    count = min(raw, max(room, 0))
    if count <= 0:
        return 0
    return int(np.sign(sign) * count)


def _resize_sides(previous: int, target: int, roll: bool) -> int:
    """Стороны за смену числа контрактов. Роллировка закрывает всё старое и открывает всё новое, даже при смене знака."""
    if roll:
        return abs(previous) + abs(target)
    if previous == target:
        return 0
    if previous == 0 or target == 0 or (previous > 0) == (target > 0):
        return abs(abs(target) - abs(previous))
    return abs(previous) + abs(target)


def leverage_curve(
    frame: pd.DataFrame,
    level: np.ndarray,
    ret: np.ndarray,
    roll: np.ndarray,
    leverage: float,
    cash: float = ACCOUNT_CASH,
    margin_rate: float = MARGIN_RATE,
    unit: float | None = None,
) -> pd.DataFrame:
    """Счёт с постоянным плечом. Каждый день контрактов floor(плечо × счёт / номинал на закрытии предыдущего дня).

    Дневной ход зарабатывает уже новое число. Издержки — за изменение числа и за роллировку. Если к закрытию
    счёт ниже ГО по цене закрытия, позиция закрывается в этот же день и на следующий набирается заново.
    На последнем дне открытый остаток закрывается одной стороной. Счёт, упавший до нуля, дальше не торгует.
    """
    if not len(frame) == len(level) == len(ret) == len(roll):
        raise ValueError("длина ряда, цены и позиции не совпадает")
    if leverage <= 0:
        raise ValueError(leverage)
    cost_unit = side_cost() if unit is None else unit
    sign = frame["position"].to_numpy(dtype=float)
    level = np.asarray(level, dtype=float)
    ret = np.asarray(ret, dtype=float)
    roll = np.asarray(roll, dtype=bool)
    n = len(frame)
    equity = np.empty(n)
    net = np.empty(n)
    gross_out = np.empty(n)
    cost_out = np.empty(n)
    lots = np.empty(n)
    free = np.empty(n)
    called = np.zeros(n, dtype=bool)
    balance = cash
    previous = 0
    dead = False
    for t in range(n):
        if dead or balance <= 0:
            dead = True
            balance = 0.0
            equity[t] = net[t] = gross_out[t] = cost_out[t] = lots[t] = free[t] = 0.0
            previous = 0
            continue
        price_in = level[t] - ret[t]
        notional = price_in * core.MULTIPLIER
        target = _target_lots(sign[t], balance, notional, leverage, margin_rate)
        sides = _resize_sides(previous, target, bool(roll[t]))
        cost = sides * cost_unit
        gross = target * ret[t] * core.MULTIPLIER
        balance_after = balance + gross - cost
        required = abs(target) * margin_rate * level[t] * core.MULTIPLIER
        margin_call = target != 0 and balance_after < required
        carry = target
        if margin_call:
            balance_after -= abs(target) * cost_unit
            cost += abs(target) * cost_unit
            carry = 0
            required = 0.0
            called[t] = True
        if t == n - 1 and carry and balance_after > 0:
            balance_after -= abs(carry) * cost_unit
            cost += abs(carry) * cost_unit
            carry = 0
            required = 0.0
        if balance_after <= 0:
            dead = True
            balance_after = 0.0
            carry = 0
            required = 0.0
        net[t] = balance_after - balance
        gross_out[t] = gross
        cost_out[t] = cost
        lots[t] = target
        free[t] = balance_after - required
        balance = balance_after
        previous = carry
        equity[t] = balance
    peak = np.maximum.accumulate(equity) if n else equity
    return pd.DataFrame(
        {
            "day": frame["day"].to_numpy(),
            "leverage": float(leverage),
            "contracts": lots,
            "gross": gross_out,
            "cost": cost_out,
            "net": net,
            "equity": equity,
            "drawdown": peak - equity,
            "free": free,
            "called": called,
        }
    )


def leverage_summary(curve: pd.DataFrame, cash: float = ACCOUNT_CASH) -> dict[str, float]:
    equity = curve["equity"].to_numpy(dtype=float)
    net = curve["net"].to_numpy(dtype=float)
    drawdown = curve["drawdown"].to_numpy(dtype=float)
    worst = int(drawdown.argmax()) if len(drawdown) else 0
    dd = float(drawdown[worst]) if len(drawdown) else 0.0
    peak = float(equity[worst] + dd) if len(equity) else cash
    previous = np.r_[cash, equity[:-1]] if len(equity) else np.array([cash])
    returns = np.divide(net, previous, out=np.zeros_like(net), where=previous > 0)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    held = curve["contracts"].to_numpy() != 0
    return {
        "leverage": float(curve["leverage"].iloc[0]) if len(curve) else 0.0,
        "end": float(equity[-1]) if len(equity) else cash,
        "net": float(net.sum()) if len(net) else 0.0,
        "gross": float(curve["gross"].sum()) if len(curve) else 0.0,
        "cost": float(curve["cost"].sum()) if len(curve) else 0.0,
        "max_dd": dd,
        "max_dd_pct_peak": dd / peak * 100.0 if peak else float("nan"),
        "min_equity": float(equity.min()) if len(equity) else cash,
        "min_free": float(curve["free"].min()) if len(curve) else cash,
        "margin_calls": float(curve["called"].sum()) if len(curve) else 0.0,
        "ruined": float((equity <= 0).any()) if len(equity) else 0.0,
        "sharpe": float(returns.mean() / std * np.sqrt(252)) if std and std == std and std > 0 else float("nan"),
        "avg_lots": float(np.abs(curve.loc[held, "contracts"]).mean()) if held.any() else 0.0,
        "max_lots": float(np.abs(curve["contracts"]).max()) if len(curve) else 0.0,
        "start_lots": float(abs(curve["contracts"].to_numpy()[np.flatnonzero(curve["contracts"].to_numpy())[0]])) if held.any() else 0.0,
        "worst_day": float(net.min()) if len(net) else 0.0,
    }


def leverage_table(
    frame: pd.DataFrame,
    level: np.ndarray,
    ret: np.ndarray,
    roll: np.ndarray,
    leverages: tuple[int, ...] = LEVERAGES,
    cash: float = ACCOUNT_CASH,
    margin_rate: float = MARGIN_RATE,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    curves = {int(lev): leverage_curve(frame, level, ret, roll, lev, cash, margin_rate) for lev in leverages}
    table = pd.DataFrame([leverage_summary(curve, cash) for curve in curves.values()])
    return table, curves


def daily_percent(series: cm.Series) -> np.ndarray:
    """Доходность дня к закрытию предыдущего дня. В первый день ряда — ноль."""
    previous = series.level - series.ret
    return np.divide(series.ret, previous, out=np.zeros(len(series)), where=np.abs(previous) > 1e-9)


def trailing_vol(percent: np.ndarray, window: int = 20) -> np.ndarray:
    return pd.Series(percent).rolling(window).std(ddof=1).to_numpy()


def expand_decisions(decided: np.ndarray, lookback: int = 20, step: int = 5) -> np.ndarray:
    """Решение на закрытии дня d держится следующие step дней. Короткое окно в конце ряда не открывается."""
    position = np.zeros(len(decided))
    decision = lookback
    while decision + step < len(decided):
        position[decision + 1 : decision + 1 + step] = decided[decision]
        decision += step
    return position


def pause_after_shock(position: np.ndarray, percent: np.ndarray, vol: np.ndarray, multiple: float = FILTER_SHOCK) -> np.ndarray:
    """Следующий день без позиции, если сегодняшний ход больше multiple своих σ. Сегодняшний день фильтр не трогает."""
    raw = (vol > 0) & (np.abs(percent) > multiple * vol)
    pause = np.zeros(len(position), dtype=bool)
    pause[1:] = raw[:-1]
    out = np.array(position, dtype=float, copy=True)
    out[pause] = 0.0
    return out


def filter_positions(series: cm.Series, lookback: int = 20, step: int = 5) -> list[tuple[str, np.ndarray]]:
    """Позиции с заранее фиксированными фильтрами. Порог и окна взяты из уже посчитанной сетки, не из счёта с плечом."""
    base = signal_of(series, lookback, 0)
    slow60 = signal_of(series, 60, 0)
    slow120 = signal_of(series, 120, 0)
    fast5 = signal_of(series, 5, 0)
    percent = daily_percent(series)
    vol = trailing_vol(percent)
    past_median = pd.Series(vol).expanding(min_periods=60).median().shift(1).to_numpy()
    high_vol = (vol > FILTER_VOL_MULTIPLE * past_median) & (past_median == past_median)
    move = move_bps(series, lookback)
    noise = vol * np.sqrt(lookback) * 1e4
    strong = (noise > 0) & (move == move) & (np.abs(move) >= noise)
    decided = {
        "без фильтра": base,
        "порог 50 б.п.": signal_of(series, lookback, 50),
        "порог 100 б.п.": signal_of(series, lookback, 100),
        "порог 200 б.п.": signal_of(series, lookback, 200),
        "знак как у 60 дней": np.where((slow60 != 0) & (base == slow60), base, 0.0),
        "знак как у 120 дней": np.where((slow120 != 0) & (base == slow120), base, 0.0),
        "без спора с 5 днями": np.where((fast5 != 0) & (base != 0) & (fast5 != base), 0.0, base),
        "без высокой волатильности": np.where(high_vol, 0.0, base),
        "ход сильнее своей σ": np.where(strong, base, 0.0),
        "только лонг": np.where(base > 0, base, 0.0),
        "только шорт": np.where(base < 0, base, 0.0),
    }
    positions = [(name, expand_decisions(values, lookback, step)) for name, values in decided.items()]
    bare = expand_decisions(base, lookback, step)
    positions.append((PRICE_PAUSE, pause_after_shock(bare, percent, vol, FILTER_SHOCK)))
    return positions


def _path_rates(summary: dict[str, float], cash: float, years: float) -> dict[str, float]:
    end = summary["end"]
    growth = end / cash if cash else float("nan")
    cagr = growth ** (1 / years) - 1 if growth > 0 and years > 0 else float("nan")
    dd = summary["max_dd_pct_peak"]
    calmar = cagr / (dd / 100.0) if dd and dd == dd and dd > 0 else float("nan")
    return {"cagr": cagr * 100.0, "calmar": calmar}


def filter_book(
    series: cm.Series,
    leverage: float = CHOSEN_LEVERAGE,
    cash: float = ACCOUNT_CASH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Счёт с фиксированным плечом для каждого фильтра. Порядок строк — порядок объявления, не порядок удачности."""
    years = max((pd.Timestamp(series.days[-1]) - pd.Timestamp(series.days[0])).days / 365.25, 1 / 365.25)
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for name, position in filter_positions(series):
        frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
        curve = leverage_curve(frame, series.level, series.ret, series.roll, leverage, cash)
        summary = leverage_summary(curve, cash)
        summary.update(_path_rates(summary, cash, years))
        summary["name"] = name
        summary["in_market"] = float((position != 0).mean() * 100.0)
        rows.append(summary)
        curves[name] = curve
    return pd.DataFrame(rows), curves


def contract_daily_volume(series: cm.Series, bars_dir=None) -> np.ndarray:
    """Сумма минутного volume активного контракта за день. На роллировке — объём нового контракта."""
    out = np.zeros(len(series))
    bars_dir = core.BARS_DIR if bars_dir is None else bars_dir
    meta = bars_dir.parent / "contracts.json"
    if not meta.exists():
        return out
    document = json.loads(meta.read_text(encoding="utf-8"))
    index = {
        (pd.Timestamp(day).normalize(), str(series.contract[i])): i
        for i, day in enumerate(series.days)
    }
    for row in document["contracts"]:
        start, end = row.get("window_start"), row.get("window_end")
        path = bars_dir / f"{row['secid']}.parquet"
        if not start or not end or not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["datetime", "volume"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame[frame["datetime"].dt.dayofweek < 5]
        daily = frame.groupby(frame["datetime"].dt.normalize())["volume"].sum()
        window = daily[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))]
        for day, value in window.items():
            slot = index.get((pd.Timestamp(day).normalize(), row["secid"]))
            if slot is not None:
                out[slot] = float(value)
    return out


def prior_volume_median(volume: np.ndarray, window: int = VOLUME_WINDOW) -> np.ndarray:
    """Медиана объёма предыдущих window дней. Сегодняшний день в медиану не входит."""
    return pd.Series(volume).rolling(window, min_periods=window).median().shift(1).to_numpy()


def pause_after_volume(
    position: np.ndarray,
    volume: np.ndarray,
    multiple: float = VOLUME_SPIKE,
    window: int = VOLUME_WINDOW,
) -> np.ndarray:
    """Следующий день без позиции, если сегодняшний объём больше multiple медиан прошлого. Сегодняшний день не трогает."""
    prior = prior_volume_median(volume, window)
    raw = (prior > 0) & (volume > multiple * prior)
    pause = np.zeros(len(position), dtype=bool)
    pause[1:] = raw[:-1]
    out = np.array(position, dtype=float, copy=True)
    out[pause] = 0.0
    return out


def _volume_decision_masks(volume: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Маски решения на закрытии дня: тихий день, кульминация, объём окна не выше предыдущего окна."""
    prior = prior_volume_median(volume)
    known = prior == prior
    quiet = known & ~(volume > prior)
    climax = known & (prior > 0) & (volume > VOLUME_SPIKE * prior)
    recent = pd.Series(volume).rolling(lookback, min_periods=lookback).sum()
    rising = (recent > recent.shift(lookback)).to_numpy()
    fading = (rising == rising) & ~rising
    return quiet, climax, fading


def volume_filter_positions(
    series: cm.Series,
    volume: np.ndarray,
    lookback: int = 20,
    step: int = 5,
) -> list[tuple[str, np.ndarray]]:
    """Фильтры громкости по одному. Сначала выбранная ценовая пауза, затем те же правила без неё."""
    base = signal_of(series, lookback, 0)
    percent = daily_percent(series)
    price_vol = trailing_vol(percent)
    quiet, climax, fading = _volume_decision_masks(volume, lookback)
    decided = {
        "громкое решение": np.where(quiet, 0.0, base),
        "без кульминации": np.where(climax, 0.0, base),
        "растущий объём": np.where(fading, 0.0, base),
    }
    bare = expand_decisions(base, lookback, step)
    chosen = pause_after_shock(bare, percent, price_vol)
    positions = [(PRICE_PAUSE, chosen)]
    for name, values in decided.items():
        held = pause_after_shock(expand_decisions(values, lookback, step), percent, price_vol)
        positions.append((f"пауза и {name}", held))
    positions.append(("пауза и пауза объёма", pause_after_volume(chosen, volume)))
    positions.append(("без фильтра", bare))
    for name, values in decided.items():
        positions.append((name, expand_decisions(values, lookback, step)))
    positions.append(("пауза после 2× объёма", pause_after_volume(bare, volume)))
    return positions


def volume_book(
    series: cm.Series,
    volume: np.ndarray,
    leverage: float = CHOSEN_LEVERAGE,
    cash: float = ACCOUNT_CASH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Счёт с плечом для фильтров громкости. Порядок строк — порядок объявления."""
    years = max((pd.Timestamp(series.days[-1]) - pd.Timestamp(series.days[0])).days / 365.25, 1 / 365.25)
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for name, position in volume_filter_positions(series, volume):
        frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
        curve = leverage_curve(frame, series.level, series.ret, series.roll, leverage, cash)
        summary = leverage_summary(curve, cash)
        summary.update(_path_rates(summary, cash, years))
        summary["name"] = name
        summary["in_market"] = float((position != 0).mean() * 100.0)
        rows.append(summary)
        curves[name] = curve
    return pd.DataFrame(rows), curves


def path_efficiency(series: cm.Series, lookback: int) -> np.ndarray:
    """|Чистый ход за lookback дней| / сумма модулей дневных ходов. До lookback — NaN."""
    out = np.full(len(series), np.nan)
    if lookback <= 0 or lookback >= len(series):
        return out
    drift = np.abs(series.price[lookback:] - series.price[:-lookback])
    cumulative = np.cumsum(np.abs(series.ret))
    path = cumulative[lookback:] - cumulative[:-lookback]
    out[lookback:] = np.divide(drift, path, out=np.zeros(len(drift)), where=path > 1e-12)
    return out


def _straight_rules(efficiency: np.ndarray) -> dict[str, np.ndarray]:
    """Какие дни решения оставить. NaN в прямоте ни одно правило не выключает."""
    known = efficiency == efficiency
    return {
        "коридор 0,15–0,5": known & (efficiency >= STRAIGHT_LOW) & (efficiency <= STRAIGHT_HIGH),
        "прямее 0,5": known & (efficiency > STRAIGHT_HIGH),
        "прямее 0,25": known & (efficiency >= STRAIGHT_FLOOR),
        "пила ниже 0,25": known & (efficiency < STRAIGHT_FLOOR),
    }


def straight_filter_positions(
    series: cm.Series,
    volume: np.ndarray,
    lookback: int = 20,
    step: int = 5,
) -> list[tuple[str, np.ndarray]]:
    """Прямота хода за L дней. Сначала поверх выбранного фильтра, затем на позиции без него."""
    base = signal_of(series, lookback, 0)
    percent = daily_percent(series)
    price_vol = trailing_vol(percent)
    _, climax, _ = _volume_decision_masks(volume, lookback)
    no_climax = np.where(climax, 0.0, base)
    efficiency = path_efficiency(series, lookback)
    known = efficiency == efficiency
    rules = _straight_rules(efficiency)

    def gated(mask: np.ndarray, decided: np.ndarray) -> np.ndarray:
        return np.where(known & ~mask, 0.0, decided)

    chosen = pause_after_shock(expand_decisions(no_climax, lookback, step), percent, price_vol)
    positions = [(PAUSE_NO_CLIMAX, chosen)]
    for name, mask in rules.items():
        held = pause_after_shock(expand_decisions(gated(mask, no_climax), lookback, step), percent, price_vol)
        positions.append((f"выбранный и {name}", held))
    positions.append(("без фильтра", expand_decisions(base, lookback, step)))
    for name, mask in rules.items():
        positions.append((name, expand_decisions(gated(mask, base), lookback, step)))
    return positions


def straight_book(
    series: cm.Series,
    volume: np.ndarray,
    leverage: float = CHOSEN_LEVERAGE,
    cash: float = ACCOUNT_CASH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Счёт с плечом для фильтров прямоты. Порядок строк — порядок объявления."""
    years = max((pd.Timestamp(series.days[-1]) - pd.Timestamp(series.days[0])).days / 365.25, 1 / 365.25)
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for name, position in straight_filter_positions(series, volume):
        frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
        curve = leverage_curve(frame, series.level, series.ret, series.roll, leverage, cash)
        summary = leverage_summary(curve, cash)
        summary.update(_path_rates(summary, cash, years))
        summary["name"] = name
        summary["in_market"] = float((position != 0).mean() * 100.0)
        rows.append(summary)
        curves[name] = curve
    return pd.DataFrame(rows), curves


def contract_daily_range(series: cm.Series, bars_dir=None) -> np.ndarray:
    """(high − low) / close активного контракта за день. На роллировке — диапазон нового контракта."""
    out = np.zeros(len(series))
    bars_dir = core.BARS_DIR if bars_dir is None else bars_dir
    meta = bars_dir.parent / "contracts.json"
    if not meta.exists():
        return out
    document = json.loads(meta.read_text(encoding="utf-8"))
    index = {
        (pd.Timestamp(day).normalize(), str(series.contract[i])): i
        for i, day in enumerate(series.days)
    }
    for row in document["contracts"]:
        start, end = row.get("window_start"), row.get("window_end")
        path = bars_dir / f"{row['secid']}.parquet"
        if not start or not end or not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["datetime", "high", "low", "close"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame[frame["datetime"].dt.dayofweek < 5]
        last = frame.groupby(frame["datetime"].dt.normalize()).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        span = (last["high"] - last["low"]) / last["close"].where(last["close"].abs() > 1e-9)
        window = span[(span.index >= pd.Timestamp(start)) & (span.index <= pd.Timestamp(end))]
        for day, value in window.items():
            slot = index.get((pd.Timestamp(day).normalize(), row["secid"]))
            if slot is not None and value == value:
                out[slot] = float(value)
    return out


def flat_on_roll(position: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """В день смены контракта позиция ноль. Календарь ролла известен до этого дня."""
    out = np.array(position, dtype=float, copy=True)
    out[np.asarray(roll, dtype=bool)] = 0.0
    return out


def same_sign_counts(series: cm.Series, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Сколько дней окна закрылись вверх и вниз. Окно то же, что у чистого хода за lookback дней."""
    sign = np.sign(series.ret)
    up = np.cumsum(sign > 0)
    down = np.cumsum(sign < 0)
    up_count = np.full(len(series), np.nan)
    down_count = np.full(len(series), np.nan)
    if lookback <= 0 or lookback >= len(series):
        return up_count, down_count
    up_count[lookback:] = up[lookback:] - up[:-lookback]
    down_count[lookback:] = down[lookback:] - down[:-lookback]
    return up_count, down_count


def wide_days(day_range: np.ndarray) -> np.ndarray:
    """День решения шире RANGE_SPIKE медиан диапазона за предыдущие VOLUME_WINDOW дней."""
    prior = prior_volume_median(day_range)
    return (prior > 0) & (day_range > RANGE_SPIKE * prior)


def chosen_account_position(
    series: cm.Series,
    volume: np.ndarray,
    day_range: np.ndarray,
    lookback: int = 20,
    step: int = 5,
) -> np.ndarray:
    """Выбранный счёт: без кульминации объёма, без широкого дня, затем пауза после 2σ."""
    _, no_climax, _, _, percent = _chosen_position(series, volume, lookback, step)
    decided = np.where(wide_days(day_range), 0.0, no_climax)
    return pause_after_shock(expand_decisions(decided, lookback, step), percent, trailing_vol(percent))


def _chosen_position(
    series: cm.Series,
    volume: np.ndarray,
    lookback: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Базовый сигнал, сигнал без кульминации, голая позиция, пауза без кульминации и дневная доходность."""
    base = signal_of(series, lookback, 0)
    percent = daily_percent(series)
    _, climax, _ = _volume_decision_masks(volume, lookback)
    no_climax = np.where(climax, 0.0, base)
    bare = expand_decisions(base, lookback, step)
    chosen = pause_after_shock(expand_decisions(no_climax, lookback, step), percent, trailing_vol(percent))
    return base, no_climax, bare, chosen, percent


def extra_filter_positions(
    series: cm.Series,
    volume: np.ndarray,
    day_range: np.ndarray,
    lookback: int = 20,
    step: int = 5,
) -> list[tuple[str, np.ndarray]]:
    """Откат, продолжение, пропуск ролла, 60% дней окна и широкий день. По одному, поверх выбранного фильтра и без него."""
    base, no_climax, bare, chosen, percent = _chosen_position(series, volume, lookback, step)
    price_vol = trailing_vol(percent)
    day = np.sign(series.ret)
    pullback = (base != 0) & (day != 0) & (day != base)
    continuation = (base != 0) & (day != 0) & (day == base)
    up_count, down_count = same_sign_counts(series, lookback)
    need = int(np.ceil(DAY_SHARE * lookback))
    majority = ((base > 0) & (up_count >= need)) | ((base < 0) & (down_count >= need))
    majority_known = up_count == up_count
    wide = wide_days(day_range)

    def hold(decided: np.ndarray, on_chosen: bool) -> np.ndarray:
        held = expand_decisions(decided, lookback, step)
        if on_chosen:
            return pause_after_shock(held, percent, price_vol)
        return held

    on_chosen = {
        "откат": np.where(pullback, no_climax, 0.0),
        "продолжение": np.where(continuation, no_climax, 0.0),
        "60% дней": np.where(majority_known & (no_climax != 0) & ~majority, 0.0, no_climax),
        "без широкого дня": np.where(wide, 0.0, no_climax),
    }
    alone = {
        "откат": np.where(pullback, base, 0.0),
        "продолжение": np.where(continuation, base, 0.0),
        "60% дней": np.where(majority_known & (base != 0) & ~majority, 0.0, base),
        "без широкого дня": np.where(wide, 0.0, base),
    }
    positions = [(PAUSE_NO_CLIMAX, chosen)]
    for name, decided in on_chosen.items():
        positions.append((f"выбранный и {name}", hold(decided, True)))
    positions.append(("выбранный и без ролла", flat_on_roll(chosen, series.roll)))
    positions.append(("без фильтра", bare))
    for name, decided in alone.items():
        positions.append((name, hold(decided, False)))
    positions.append(("без ролла", flat_on_roll(bare, series.roll)))
    return positions


def extra_book(
    series: cm.Series,
    volume: np.ndarray,
    day_range: np.ndarray,
    leverage: float = CHOSEN_LEVERAGE,
    cash: float = ACCOUNT_CASH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Счёт с плечом для ещё не смотренных фильтров. Порядок строк — порядок объявления."""
    years = max((pd.Timestamp(series.days[-1]) - pd.Timestamp(series.days[0])).days / 365.25, 1 / 365.25)
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for name, position in extra_filter_positions(series, volume, day_range):
        frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
        curve = leverage_curve(frame, series.level, series.ret, series.roll, leverage, cash)
        summary = leverage_summary(curve, cash)
        summary.update(_path_rates(summary, cash, years))
        summary["name"] = name
        summary["in_market"] = float((position != 0).mean() * 100.0)
        rows.append(summary)
        curves[name] = curve
    return pd.DataFrame(rows), curves


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


def _account_grid_view(grid: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in grid.sort_values("contracts").iterrows():
        out.append(
            {
                "контрактов": int(row["contracts"]),
                "финиш, ₽": _num(row["end"]),
                "net, ₽": _num(row["net"], signed=True),
                "просадка, ₽": _num(row["max_dd"]),
                "просадка, % старта": _num(row["max_dd_pct_start"], 1),
                "минимум счёта, ₽": _num(row["min_equity"]),
                "мин. свободно, ₽": _num(row["min_free"]),
                "маржин-колл": "да" if row["margin_call"] else "нет",
            }
        )
    return pd.DataFrame(out)


def _account_year_rule(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in table.iterrows():
        out.append(
            {
                "на конец года": int(row["year"]),
                "видимая просадка 1 контракта, ₽": _num(row["max_dd"]),
                "контрактов по 10%": int(row["contracts"]),
            }
        )
    return pd.DataFrame(out)


def _account_pnl_by_year(curve: pd.DataFrame) -> pd.DataFrame:
    out = []
    for year, part in curve.groupby(curve["day"].dt.year, sort=True):
        out.append({"год": int(year), "net, ₽": _num(float(part["net"].sum()), signed=True)})
    return pd.DataFrame(out)


def _five_contract_line(book: AccountBook) -> str:
    match = book.grid[book.grid["contracts"] == 5]
    if match.empty or book.chosen == 5:
        return ""
    five = match.iloc[0]
    return (
        f" Пять контрактов на той же истории: финиш {_rub(five['end'])} ₽, "
        f"просадка {_rub(five['max_dd'])} ₽ ({_comma(five['max_dd_pct_start'], 1)}% старта)."
    )


def _account_section(series: cm.Series, study: Study) -> list[str]:
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    book = describe_account(frame, series.level)
    chosen = book.grid[book.grid["contracts"] == book.chosen]
    row = chosen.iloc[0] if len(chosen) else None
    slots = int(ACCOUNT_CASH // EXCHANGE_MARGIN_RUB)
    lines = [
        "## 9. Счёт 100 000 ₽\n",
        (
            f"Опорное удержание L = {PRIMARY[0]}, H = {PRIMARY[1]}, число контрактов одно и то же всю историю. "
            "Шарп от этого числа не меняется: и прибыль, и просадка растут пропорционально. "
            f"Стартовые деньги — {f'{int(ACCOUNT_CASH):,}'.replace(',', ' ')} ₽."
        ),
        (
            f"ГО. На {EXCHANGE_MARGIN_DAY} биржа по CRZ6 берёт {_rub(EXCHANGE_MARGIN_RUB, 2)} ₽ "
            f"при расчётной цене {str(EXCHANGE_SETTLE).replace('.', ',')} — это {_comma(MARGIN_RATE * 100, 1)}% номинала. "
            f"На стартовые 100 000 ₽ по этому ГО влезает {slots} контрактов. "
            f"Если ту же долю приложить к ценам истории, ГО одного контракта было от {_rub(book.go_min)} до {_rub(book.go_max)} ₽. "
            f"Свободные деньги на кривой остаются выше нуля до {book.max_by_margin} контрактов включительно. "
            "Это потолок залога, не размер для торговли."
        ),
        (
            "Размер. Доля риска та же, что у короткого минутного окна: запланированный убыток не больше 10% счёта, то есть 10 000 ₽. "
            + (
                f"У моментума нет стопа, поэтому мера убытка — просадка одного контракта {_rub(book.per_contract_dd)} ₽ с {book.dd_from} по {book.dd_to}. "
                f"10 000 / {_rub(book.per_contract_dd)} — это {_comma(ACCOUNT_RISK * ACCOUNT_CASH / book.per_contract_dd, 2)}, "
                f"целое число вниз — **{book.chosen}**. "
                "До этой просадки видимый убыток был меньше, и то же правило разрешало больше контрактов."
                if book.per_contract_dd > 0
                else "Просадки одного контракта на этом ряду нет, поэтому правило 10% не назначает размер."
            )
        ),
    ]
    if len(book.by_year):
        lines.extend(["", report.md(_account_year_rule(book.by_year), index=False)])
    if row is not None:
        lines.extend(
            [
                "",
                (
                    f"Прогон {book.chosen} контрактов: финиш {_rub(row['end'])} ₽, net {_rub(row['net'], signed=True)} ₽, "
                    f"просадка {_rub(row['max_dd'])} ₽ ({_comma(row['max_dd_pct_start'], 1)}% старта, "
                    f"{_comma(row['max_dd_pct_peak'], 1)}% пика, с которого она началась). "
                    f"Самый низкий счёт — {_rub(row['min_equity'])} ₽ на {book.low_day}. "
                    f"Худший день одного контракта {_rub(book.worst_day_pnl, signed=True)} ₽ ({book.worst_day}), "
                    f"на этом счёте {_rub(book.worst_day_pnl * book.chosen, signed=True)} ₽. "
                    f"Минимум свободных после ГО — {_rub(row['min_free'])} ₽."
                    + _five_contract_line(book)
                ),
                "",
                "Число контрактов взято по просадке той же истории. Прогон показывает, каким был бы счёт, а не то, что правило было известно в 2022 году.",
                "",
                report.md(_account_grid_view(book.grid), index=False),
            ]
        )
        if book.chosen and len(book.chosen_curve):
            lines.extend(["", "Net выбранного числа контрактов по календарным годам. Шарп года совпадает с одним контрактом.\n", report.md(_account_pnl_by_year(book.chosen_curve), index=False)])
    lines.append("В живую функцию правило не переносится: это размер исследовательского счёта, не заявка.")
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _leverage_view(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in table.sort_values("leverage").iterrows():
        out.append(
            {
                "плечо": int(row["leverage"]),
                "финиш, ₽": _rub(row["end"]),
                "net, ₽": _rub(row["net"], signed=True),
                "просадка, ₽": _rub(row["max_dd"]),
                "просадка, % пика": _comma(row["max_dd_pct_peak"], 1),
                "минимум, ₽": _rub(row["min_equity"]),
                "мин. свободно, ₽": _rub(row["min_free"]),
                "коллов": int(row["margin_calls"]),
                "контрактов, среднее": _comma(row["avg_lots"], 1),
                "худший день, ₽": _rub(row["worst_day"], signed=True),
                "Шарп": _comma(row["sharpe"], 2, signed=True),
            }
        )
    return pd.DataFrame(out)


def _leverage_years(curves: dict[int, pd.DataFrame], cash: float = ACCOUNT_CASH) -> pd.DataFrame:
    nets: dict[int, list[float]] = {}
    years: list[int] = []
    for leverage, curve in curves.items():
        previous = cash
        column: list[float] = []
        labels: list[int] = []
        for year, part in curve.groupby(curve["day"].dt.year, sort=True):
            end = float(part["equity"].iloc[-1])
            labels.append(int(year))
            column.append(end - previous)
            previous = end
        years = labels
        nets[int(leverage)] = column
    rows = []
    for index, year in enumerate(years):
        item: dict[str, object] = {"год": year}
        for leverage in sorted(nets):
            item[f"×{leverage}, ₽"] = _rub(nets[leverage][index], signed=True)
        rows.append(item)
    return pd.DataFrame(rows)


def _leverage_row(table: pd.DataFrame, leverage: int) -> pd.Series | None:
    match = table[table["leverage"] == leverage]
    if match.empty:
        return None
    return match.iloc[0]


def _leverage_section(series: cm.Series, study: Study) -> list[str]:
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    table, curves = leverage_table(frame, series.level, series.ret, series.roll)
    one = _leverage_row(table, 1)
    four = _leverage_row(table, 4)
    five = _leverage_row(table, 5)
    lines = [
        "## 10. Фиксированное плечо от 1 до 8\n",
        (
            "Плечо — номинал позиции, делённый на текущий счёт. Каждый день контрактов "
            "floor(плечо × счёт / номинал), номинал берётся по закрытию предыдущего дня. "
            "Это та же формула, что у минутного движка в момент входа, только здесь размер обновляется ежедневно, "
            "поэтому плечо не уезжает вместе с прибылью. Издержки — 6 ₽ на сторону за каждый добавленный или снятый контракт "
            "и полная роллировка всего объёма. ГО — те же 9,6% номинала. Если к закрытию счёт ниже ГО, позиция закрывается "
            "в этот день и на следующий набирается заново, пока на счёте есть деньги."
        ),
    ]
    if one is not None:
        lines.append(
            f"Плечо 1 на первом входе — это {_rub(one['start_lots'], 0)} контрактов, не один. "
            f"Четыре контракта из раздела 9 держат номинал около половины стартового счёта и не увеличивают его, когда счёт растёт. "
            f"Здесь плечо 1 доводит счёт до {_rub(one['end'])} ₽, просадка {_rub(one['max_dd'])} ₽ "
            f"({_comma(one['max_dd_pct_peak'], 1)}% пика), маржин-коллов {int(one['margin_calls'])}. "
            f"В среднем в позиции {_comma(one['avg_lots'], 1)} контракта."
        )
    if four is not None and five is not None:
        richest = table.loc[table["end"].idxmax()]
        lines.append(
            f"Плечо 4, как у минутных окон 30 и 45: финиш {_rub(four['end'])} ₽, "
            f"просадка {_rub(four['max_dd'])} ₽ ({_comma(four['max_dd_pct_peak'], 1)}% пика), "
            f"худший день {_rub(four['worst_day'], signed=True)} ₽, свободно в минимуме {_rub(four['min_free'])} ₽, "
            f"коллов {int(four['margin_calls'])}. "
            f"Самый высокий финиш на этой истории — плечо {int(richest['leverage'])}: {_rub(richest['end'])} ₽, "
            f"просадка {_comma(richest['max_dd_pct_peak'], 1)}% пика, свободно в минимуме {_rub(richest['min_free'])} ₽."
        )
    called = table[table["margin_calls"] > 0].sort_values("leverage")
    if len(called):
        bits = ", ".join(f"×{int(row.leverage)} — {int(row.margin_calls)}" for row in called.itertuples())
        first = int(called["leverage"].iloc[0])
        lines.append(
            f"Закрытия по ГО начинаются с плеча {first}: {bits} дней. "
            "Дальше финиш уже ниже, чем у плеча без этих закрытий: залог обрезает позицию в просадке, и счёт не успевает восстановить тот же номинал."
        )
    sharpe_low = float(table["sharpe"].min()) if len(table) else float("nan")
    sharpe_high = float(table["sharpe"].max()) if len(table) else float("nan")
    lines.append(
        f"Шарп дневной доходности счёта остаётся между {_comma(sharpe_low, 2, signed=True)} и {_comma(sharpe_high, 2, signed=True)}. "
        "Плечо не делает сигнал лучше: оно растягивает рублёвый результат и просадку, пока ГО не начинает закрывать позицию."
    )
    lines.extend(["", report.md(_leverage_view(table), index=False), "", "Net по календарным годам, рубли на счёт.\n", report.md(_leverage_years(curves), index=False)])
    lines.append("\nВ живую функцию плечо не переносится.")
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _year_changes(curve: pd.DataFrame, cash: float = ACCOUNT_CASH) -> dict[int, float]:
    previous = cash
    out: dict[int, float] = {}
    for year, part in curve.groupby(curve["day"].dt.year, sort=True):
        end = float(part["equity"].iloc[-1])
        out[int(year)] = end - previous
        previous = end
    return out


def _filter_view(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in table.iterrows():
        out.append(
            {
                "фильтр": row["name"],
                "финиш, ₽": _rub(row["end"]),
                "год, %": _comma(row["cagr"], 1),
                "просадка, % пика": _comma(row["max_dd_pct_peak"], 1),
                "Калмар": _comma(row["calmar"], 2),
                "Шарп": _comma(row["sharpe"], 2, signed=True),
                "в рынке, %": _comma(row["in_market"], 0),
                "минимум, ₽": _rub(row["min_equity"]),
            }
        )
    return pd.DataFrame(out)


def _filter_years(curves: dict[str, pd.DataFrame], names: tuple[str, ...]) -> pd.DataFrame:
    changes = {name: _year_changes(curves[name]) for name in names if name in curves}
    years = sorted({year for column in changes.values() for year in column})
    rows = []
    for year in years:
        item: dict[str, object] = {"год": year}
        for name in names:
            if name in changes:
                item[name] = _rub(changes[name].get(year, float("nan")), signed=True)
        rows.append(item)
    return pd.DataFrame(rows)


def _filter_named(table: pd.DataFrame, name: str) -> pd.Series | None:
    match = table[table["name"] == name]
    if match.empty:
        return None
    return match.iloc[0]


def _signed_mean(signed: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(signed[mask].mean())


def _finish_at_shock(series: cm.Series, multiple: float, leverage: float = CHOSEN_LEVERAGE) -> float:
    bare = dict(filter_positions(series))["без фильтра"]
    percent = daily_percent(series)
    position = pause_after_shock(bare, percent, trailing_vol(percent), multiple)
    frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
    curve = leverage_curve(frame, series.level, series.ret, series.roll, leverage)
    return float(curve["equity"].iloc[-1]) if len(curve) else float("nan")


def _filter_section(series: cm.Series, study: Study) -> list[str]:
    table, curves = filter_book(series)
    base = _filter_named(table, "без фильтра")
    mild = _filter_named(table, "порог 50 б.п.")
    tight = _filter_named(table, "порог 100 б.п.")
    wide = _filter_named(table, "порог 200 б.п.")
    pause = _filter_named(table, PRICE_PAUSE)
    focus = ("без фильтра", "порог 50 б.п.", "порог 100 б.п.", PRICE_PAUSE)
    lines = [
        f"## 11. Фильтры при плече {CHOSEN_LEVERAGE}\n",
        (
            f"Счёт тот же, {f'{int(ACCOUNT_CASH):,}'.replace(',', ' ')} ₽, плечо {CHOSEN_LEVERAGE} фиксировано. "
            f"Сигнал по-прежнему L = {PRIMARY[0]}, H = {PRIMARY[1]}. "
            "Фильтр либо гасит решение на закрытии дня решения и держит это H дней, "
            "либо, в случае паузы, снимает позицию только на следующий день после большого хода. "
            "Пороги 50, 100 и 200 б.п. уже были в сетке одного контракта. Остальные пороги тоже заданы заранее: "
            "согласие со знаком 60 и 120 дней, спор со знаком 5 дней, волатильность выше 1,5 своей расширяющейся медианы, "
            "ход за 20 дней больше своей σ, пауза после хода больше 2σ, только лонг и только шорт."
        ),
    ]
    if base is not None and mild is not None and tight is not None and wide is not None and pause is not None:
        base_2024 = _year_changes(curves["без фильтра"]).get(2024, float("nan"))
        tight_2024 = _year_changes(curves["порог 100 б.п."]).get(2024, float("nan"))
        shock3 = _finish_at_shock(series, 3.0)
        lines.append(
            f"Без фильтра финиш {_rub(base['end'])} ₽, просадка {_comma(base['max_dd_pct_peak'], 1)}% пика, "
            f"Шарп {_comma(base['sharpe'], 2, signed=True)}, Калмар {_comma(base['calmar'], 2)}."
        )
        positions = dict(filter_positions(series))
        percent = daily_percent(series)
        signed = positions["без фильтра"] * percent
        weak = (positions["без фильтра"] != 0) & (positions["порог 100 б.п."] == 0)
        other = (positions["без фильтра"] != 0) & ~weak
        vol = trailing_vol(percent)
        raw = (vol > 0) & (np.abs(percent) > FILTER_SHOCK * vol)
        after = np.zeros(len(series), dtype=bool)
        after[1:] = raw[:-1]
        paused = (positions["без фильтра"] != 0) & after
        kept = (positions["без фильтра"] != 0) & ~after
        lines.append(
            f"Порог 50 б.п. оставляет финиш {_rub(mild['end'])} ₽ и снижает просадку до {_comma(mild['max_dd_pct_peak'], 1)}% пика, "
            f"Калмар {_comma(mild['calmar'], 2)}, Шарп {_comma(mild['sharpe'], 2, signed=True)}. "
            f"Порог 100 б.п. режет просадку до {_comma(tight['max_dd_pct_peak'], 1)}% и поднимает Калмар до {_comma(tight['calmar'], 2)}, "
            f"финиш при этом {_rub(tight['end'])} ₽. В 2024 году этот порог даёт {_rub(tight_2024, signed=True)} ₽ "
            f"против {_rub(base_2024, signed=True)} ₽ без фильтра. "
            f"Дни, которые порог 100 б.п. выключает, в среднем дают {_comma(_signed_mean(signed, weak) * 100, 2, signed=True)}% со знаком позиции, "
            f"остальные дни в позиции — {_comma(_signed_mean(signed, other) * 100, 2, signed=True)}%. "
            f"Порог 200 б.п. уже снижает и финиш ({_rub(wide['end'])} ₽), и Шарп ({_comma(wide['sharpe'], 2, signed=True)})."
        )
        lines.append(
            f"Пауза на следующий день после хода больше 2σ поднимает финиш до {_rub(pause['end'])} ₽ "
            f"и Шарп до {_comma(pause['sharpe'], 2, signed=True)}. Просадка {_comma(pause['max_dd_pct_peak'], 1)}% пика. "
            f"В день после такого хода знак позиции в среднем даёт {_comma(_signed_mean(signed, paused) * 100, 2, signed=True)}%, "
            f"в остальные дни в позиции — {_comma(_signed_mean(signed, kept) * 100, 2, signed=True)}%. "
            f"Та же пауза с порогом 3σ заканчивается на {_rub(shock3)} ₽, то есть ниже базы: годится не любая отсечка."
        )
        others = table[~table["name"].isin(("без фильтра", "порог 50 б.п.", "порог 100 б.п.", "порог 200 б.п.", PRICE_PAUSE))]
        weaker = others[others["end"] < base["end"]]
        if len(weaker):
            lines.append(
                "Согласие с длинным знаком, запрет спора с 5 днями, отключение высокой волатильности, "
                "фильтр «ход сильнее своей σ», только лонг и только шорт финиш не поднимают. "
                "Лонг и шорт по отдельности оба слабее базы: на этом плече работают обе стороны. "
                "Фильтры в новое правило не складывались."
            )
    lines.extend(
        [
            "",
            report.md(_filter_view(table), index=False),
            "",
            "Net отобранных для сравнения вариантов по календарным годам.\n",
            report.md(_filter_years(curves, focus), index=False),
            "",
            f"Из этой таблицы оставляем фильтр «{PRICE_PAUSE}». Вместе с отключением кульминации объёма и широкого дня он записан в следующем разделе.",
        ]
    )
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _chosen_section(series: cm.Series, study: Study) -> list[str]:
    """Счёт, который оставляем: пауза после 2σ, без кульминации объёма и без широкого дня."""
    del study
    cash = f"{int(ACCOUNT_CASH):,}".replace(",", " ")
    lines = [
        "## 12. Выбранный фильтр\n",
        (
            f"Оставляем фильтр «{CHOSEN_FILTER}». Счёт {cash} ₽, плечо {CHOSEN_LEVERAGE}, "
            f"L = {PRIMARY[0]}, H = {PRIMARY[1]}, удержание. "
            f"Если модуль дневной доходности больше {FILTER_SHOCK:.0f} своих σ за 20 дней, "
            "на следующий день позиция равна нулю. День самого хода эта пауза не закрывает. "
            f"Если в день решения объём больше {VOLUME_SPIKE:.0f} медиан предыдущих {VOLUME_WINDOW} дней, "
            "следующие H дней позиция не открывается. "
            f"Если (high − low) / close дня решения больше {RANGE_SPIKE:.0f} медиан предыдущих {VOLUME_WINDOW} дней, "
            "следующие H дней позиция тоже не открывается. "
            "Порог 50 б.п., остальные фильтры громкости, прямота, откат, продолжение, доля дней и пропуск ролла в правило не входят."
        ),
    ]
    volume = contract_daily_volume(series)
    day_range = contract_daily_range(series)
    if not np.any(volume > 0):
        lines.append("Для этого фильтра нужны дневной объём и диапазон активного контракта, на этом ряду их нет.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    position = chosen_account_position(series, volume, day_range)
    frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
    years = max((pd.Timestamp(series.days[-1]) - pd.Timestamp(series.days[0])).days / 365.25, 1 / 365.25)
    curve = leverage_curve(frame, series.level, series.ret, series.roll, CHOSEN_LEVERAGE, ACCOUNT_CASH)
    row = leverage_summary(curve, ACCOUNT_CASH)
    row.update(_path_rates(row, ACCOUNT_CASH, years))
    row["in_market"] = float((position != 0).mean() * 100.0)
    _, volume_curves = volume_book(series, volume)
    prior = dict(volume_filter_positions(series, volume))[PAUSE_NO_CLIMAX]
    removed = (prior != 0) & (position == 0)
    kept = (prior != 0) & (position != 0)
    signed = prior * daily_percent(series)
    n_off = int(removed.sum())
    if n_off % 10 == 1 and n_off % 100 != 11:
        off_word = "день"
    elif 2 <= n_off % 10 <= 4 and not 12 <= n_off % 100 <= 14:
        off_word = "дня"
    else:
        off_word = "дней"
    lines.append(
        f"Финиш {_rub(row['end'])} ₽, net {_rub(row['net'], signed=True)} ₽, "
        f"издержки {_rub(row['cost'])} ₽, {_comma(row['cagr'], 1)}% в год, "
        f"просадка {_comma(row['max_dd_pct_peak'], 1)}% пика, "
        f"Калмар {_comma(row['calmar'], 2)}, Шарп {_comma(row['sharpe'], 2, signed=True)}. "
        f"В рынке {_comma(row['in_market'], 0)}% дней. "
        f"Широкий день снимает ещё {n_off} {off_word} сверх «{PAUSE_NO_CLIMAX}»: "
        f"эти дни в среднем дают {_comma(_signed_mean(signed, removed) * 100, 2, signed=True)}% со знаком позиции, "
        f"остальные дни той позиции — {_comma(_signed_mean(signed, kept) * 100, 2, signed=True)}%. "
        f"Минимум счёта {_rub(row['min_equity'])} ₽, свободно после ГО не ниже {_rub(row['min_free'])} ₽, "
        f"маржин-коллов {int(row['margin_calls'])}. "
        f"Худший день {_rub(row['worst_day'], signed=True)} ₽. "
        f"В позиции в среднем {_comma(row['avg_lots'], 1)} контракта."
    )
    volume_curves[CHOSEN_FILTER] = curve
    prior_2023 = _year_changes(volume_curves[PAUSE_NO_CLIMAX]).get(2023, float("nan"))
    wide_2023 = _year_changes(curve).get(2023, float("nan"))
    if prior_2023 == prior_2023 and wide_2023 == wide_2023 and wide_2023 < prior_2023:
        lines.append(
            f"В 2023 году широкий день даёт {_rub(wide_2023, signed=True)} ₽ "
            f"против {_rub(prior_2023, signed=True)} ₽ у «{PAUSE_NO_CLIMAX}»."
        )
    lines.extend(
        [
            "",
            "Net выбранного фильтра, счёта без широкого дня, ценовой паузы и счёта без фильтра по календарным годам.\n",
            report.md(
                _filter_years(volume_curves, ("без фильтра", PRICE_PAUSE, PAUSE_NO_CLIMAX, CHOSEN_FILTER)),
                index=False,
            ),
            "",
            "В живую функцию фильтр не переносится: пауза, кульминация и широкий день смотрелись на той же истории CR.",
        ]
    )
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _lots(value: float) -> str:
    """Целое с пробелом между тысячами. Для объёма в контрактах, не для рублей."""
    if value != value:
        return "—"
    return f"{int(round(abs(value))):,}".replace(",", " ")


def _volume_year_medians(series: cm.Series, volume: np.ndarray) -> str:
    days = pd.to_datetime(series.days)
    bits = []
    for year, part in pd.Series(volume).groupby(days.year):
        bits.append(f"{int(year)} — {_lots(float(part.median()))}")
    return ", ".join(bits)


def _volume_helps(row: pd.Series, base: pd.Series) -> bool:
    """Выше финиш без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при финише не ниже 98% базы."""
    higher = float(row["end"]) > float(base["end"]) and float(row["max_dd_pct_peak"]) <= float(base["max_dd_pct_peak"]) + 0.05
    calmer = float(row["end"]) >= float(base["end"]) * 0.98 and float(row["max_dd_pct_peak"]) <= float(base["max_dd_pct_peak"]) - 2.0
    return bool(higher or calmer)


def _volume_section(series: cm.Series, study: Study) -> list[str]:
    """Громкость на выбранном счёте: дневной объём, фильтры по одному, отдельно от ценовой паузы и поверх неё."""
    del study
    volume = contract_daily_volume(series)
    lines = [
        "## 13. Фильтры по громкости\n",
        (
            "Громкость — дневной объём активного контракта, сумма минутного volume. "
            "На роллировке берётся объём нового контракта. "
            f"Порог объявлен заранее: медиана предыдущих {VOLUME_WINDOW} дней, сегодняшний день в неё не входит. "
            f"Всплеск — объём больше {VOLUME_SPIKE:.0f} таких медиан, та же отсечка, что у ценовой паузы. "
            "Решение на закрытии дня держится H дней. Пауза по объёму, как и ценовая, снимает только следующий день."
        ),
    ]
    if not np.any(volume > 0):
        lines.append("Для этого ряда дневной объём активного контракта не собран.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    table, curves = volume_book(series, volume)
    base = _filter_named(table, PRICE_PAUSE)
    overlays = ("пауза и громкое решение", "пауза и без кульминации", "пауза и растущий объём", "пауза и пауза объёма")
    alone = ("громкое решение", "без кульминации", "растущий объём", "пауза после 2× объёма")
    days = pd.to_datetime(series.days)
    yearly = pd.Series(volume).groupby(days.year).median()
    growth = ""
    if len(yearly) >= 2 and float(yearly.iloc[0]) > 0 and float(yearly.iloc[-1]) > 5 * float(yearly.iloc[0]):
        growth = " Объём от начала ряда к концу вырос в несколько раз, поэтому медиана всей истории почти всегда называет поздние дни громкими."
    lines.append(
        f"Медиана дня по годам, контрактов: {_volume_year_medians(series, volume)}.{growth} "
        f"Счёт тот же: {_lots(ACCOUNT_CASH)} ₽, плечо {CHOSEN_LEVERAGE}, L = {PRIMARY[0]}, H = {PRIMARY[1]}. "
        "Каждый фильтр смотрится отдельно. Поверх выбранной ценовой паузы фильтры громкости не складываются друг с другом."
    )
    if base is not None:
        helped = []
        for name in overlays:
            row = _filter_named(table, name)
            if row is not None and _volume_helps(row, base):
                helped.append(name)
        if helped:
            lines.append(
                "На выбранной паузе после 2σ счёт улучшают: "
                + ", ".join(f"«{name}»" for name in helped)
                + ". Улучшение здесь — финиш выше без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при финише не ниже 98% этой паузы."
            )
        else:
            lines.append(
                "На выбранной паузе после 2σ ни один из четырёх фильтров громкости счёт не улучшает: "
                "финиш не растёт без более глубокой просадки, и просадка не становится меньше хотя бы на 2 пункта при близком финише."
            )
        positions = dict(volume_filter_positions(series, volume))
        percent = daily_percent(series)
        signed = positions[PRICE_PAUSE] * percent
        bare_signed = positions["без фильтра"] * percent
        volume_pause = positions["пауза и пауза объёма"]
        dropped = (positions[PRICE_PAUSE] != 0) & (volume_pause == 0)
        kept = (positions[PRICE_PAUSE] != 0) & (volume_pause != 0)
        loud = positions["пауза и громкое решение"]
        quiet_days = (positions[PRICE_PAUSE] != 0) & (loud == 0)
        loud_days = (positions[PRICE_PAUSE] != 0) & (loud != 0)
        climax_off = (positions["без фильтра"] != 0) & (positions["без кульминации"] == 0)
        climax_on = (positions["без фильтра"] != 0) & (positions["без кульминации"] != 0)
        both_off = (positions[PRICE_PAUSE] != 0) & (positions[PAUSE_NO_CLIMAX] == 0)
        lines.append(
            f"Без ценовой паузы дни, которые «без кульминации» выключает, в среднем дают {_comma(_signed_mean(bare_signed, climax_off) * 100, 2, signed=True)}% со знаком позиции, "
            f"остальные дни в позиции — {_comma(_signed_mean(bare_signed, climax_on) * 100, 2, signed=True)}%. "
            f"Дни, которые пауза объёма снимает уже с выбранной ценовой паузы, в среднем дают {_comma(_signed_mean(signed, dropped) * 100, 2, signed=True)}%, "
            f"остальные дни этой позиции — {_comma(_signed_mean(signed, kept) * 100, 2, signed=True)}%. "
            f"Дни, которые «без кульминации» снимает с той же паузы, в среднем дают {_comma(_signed_mean(signed, both_off) * 100, 2, signed=True)}%. "
            "Поверх ценовой паузы средний день, который убирает громкость, почти нулевой: добавка к финишу здесь идёт от другого пути плеча, а не от большого минуса этих дней. "
            f"Тихие решения дают {_comma(_signed_mean(signed, quiet_days) * 100, 2, signed=True)}%, громкие — {_comma(_signed_mean(signed, loud_days) * 100, 2, signed=True)}%."
        )
        bare = _filter_named(table, "без фильтра")
        if bare is not None:
            alone_helped = []
            for name in alone:
                row = _filter_named(table, name)
                if row is not None and _volume_helps(row, bare):
                    alone_helped.append(name)
            if alone_helped:
                lines.append(
                    "Без ценовой паузы относительно позиции без фильтра счёт улучшают: "
                    + ", ".join(f"«{name}»" for name in alone_helped)
                    + "."
                )
            else:
                lines.append("Без ценовой паузы те же четыре правила относительно позиции без фильтра счёт не улучшают.")
    lines.extend(
        [
            "",
            report.md(_filter_view(table), index=False),
            "",
            "Net выбранной паузы и фильтров громкости поверх неё по календарным годам.\n",
            report.md(_filter_years(curves, (PRICE_PAUSE, *overlays)), index=False),
            "",
            f"Из фильтров громкости оставляем «{PAUSE_NO_CLIMAX}». Широкий день добавлен к этому правилу в разделе 12. В живую функцию он не переносится.",
        ]
    )
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _straight_section(series: cm.Series, study: Study) -> list[str]:
    """Прямота L-дневного хода на выбранном фильтре и отдельно на позиции без него."""
    del study
    lines = [
        "## 14. Фильтры по прямоте\n",
        (
            "Прямота — модуль чистого хода за L дней, делённый на сумму модулей дневных ходов за те же дни. "
            "Единица — ход без единого отката, ноль — путь съел весь чистый сдвиг. "
            f"Считается на закрытии дня решения и держится H дней. "
            f"Коридор {_comma(STRAIGHT_LOW, 2)}–{_comma(STRAIGHT_HIGH, 1)} — тот же, что у 30-минутного окна. "
            f"Порог {_comma(STRAIGHT_FLOOR, 2)} — округление 1/√{PRIMARY[0]}, масштаб случайного блуждания за L дней. "
            "Пила — дни ниже этого порога. Граница 0,25 остаётся у «прямее 0,25»."
        ),
    ]
    volume = contract_daily_volume(series)
    if not np.any(volume > 0):
        lines.append("Для сравнения с выбранным фильтром нужен дневной объём, на этом ряду его нет.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    table, curves = straight_book(series, volume)
    base = _filter_named(table, PAUSE_NO_CLIMAX)
    overlays = (
        "выбранный и коридор 0,15–0,5",
        "выбранный и прямее 0,5",
        "выбранный и прямее 0,25",
        "выбранный и пила ниже 0,25",
    )
    alone = ("коридор 0,15–0,5", "прямее 0,5", "прямее 0,25", "пила ниже 0,25")
    efficiency = path_efficiency(series, PRIMARY[0])
    known = efficiency == efficiency
    if np.any(known):
        lines.append(
            f"Медиана прямоты за {PRIMARY[0]} дней — {_comma(float(np.median(efficiency[known])), 2)}. "
            f"Ниже {_comma(STRAIGHT_FLOOR, 2)} оказывается {_comma(float((efficiency[known] < STRAIGHT_FLOOR).mean() * 100), 0)}% дней, "
            f"в коридоре {_comma(STRAIGHT_LOW, 2)}–{_comma(STRAIGHT_HIGH, 1)} — {_comma(float(((efficiency[known] >= STRAIGHT_LOW) & (efficiency[known] <= STRAIGHT_HIGH)).mean() * 100), 0)}%, "
            f"прямее {_comma(STRAIGHT_HIGH, 1)} — {_comma(float((efficiency[known] > STRAIGHT_HIGH).mean() * 100), 0)}%."
        )
    lines.append(
        f"Счёт тот же: {_lots(ACCOUNT_CASH)} ₽, плечо {CHOSEN_LEVERAGE}, фильтр «{PAUSE_NO_CLIMAX}». "
        "Прямота смотрелась до широкого дня. Каждое правило смотрится отдельно и с другими не складывается."
    )
    if base is not None:
        helped = [name for name in overlays if (row := _filter_named(table, name)) is not None and _volume_helps(row, base)]
        if helped:
            lines.append(
                "На выбранном фильтре счёт улучшают: "
                + ", ".join(f"«{name}»" for name in helped)
                + ". Улучшение — финиш выше без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при финише не ниже 98% выбранного счёта."
            )
        else:
            lines.append(
                "На выбранном фильтре ни одно из четырёх правил прямоты счёт не улучшает: "
                "финиш не растёт без более глубокой просадки, и просадка не становится меньше хотя бы на 2 пункта при близком финише."
            )
        band = _filter_named(table, overlays[0])
        if band is not None and base is not None:
            band_2024 = _year_changes(curves[overlays[0]]).get(2024, float("nan"))
            lines.append(
                f"Коридор 0,15–0,5 снижает просадку до {_comma(band['max_dd_pct_peak'], 1)}% пика и поднимает Калмар до {_comma(band['calmar'], 2)}, "
                f"финиш при этом {_rub(band['end'])} ₽. В 2024 году коридор даёт {_rub(band_2024, signed=True)} ₽."
            )
        positions = dict(straight_filter_positions(series, volume))
        percent = daily_percent(series)
        signed = positions[PAUSE_NO_CLIMAX] * percent
        bare_signed = positions["без фильтра"] * percent
        bits = []
        for label, stacked, plain in (
            ("коридор 0,15–0,5", overlays[0], alone[0]),
            ("прямее 0,5", overlays[1], alone[1]),
            ("прямее 0,25", overlays[2], alone[2]),
            ("пила ниже 0,25", overlays[3], alone[3]),
        ):
            off = (positions[PAUSE_NO_CLIMAX] != 0) & (positions[stacked] == 0)
            kept = (positions[PAUSE_NO_CLIMAX] != 0) & (positions[stacked] != 0)
            bare_off = (positions["без фильтра"] != 0) & (positions[plain] == 0)
            bare_on = (positions["без фильтра"] != 0) & (positions[plain] != 0)
            bits.append(
                f"«{label}» снимает с выбранной позиции дни со средним {_comma(_signed_mean(signed, off) * 100, 2, signed=True)}% "
                f"против {_comma(_signed_mean(signed, kept) * 100, 2, signed=True)}% у оставшихся. "
                f"Без фильтра те же дни дают {_comma(_signed_mean(bare_signed, bare_off) * 100, 2, signed=True)}% "
                f"против {_comma(_signed_mean(bare_signed, bare_on) * 100, 2, signed=True)}%."
            )
        lines.append(" ".join(bits))
        bare = _filter_named(table, "без фильтра")
        if bare is not None:
            alone_helped = [name for name in alone if (row := _filter_named(table, name)) is not None and _volume_helps(row, bare)]
            if alone_helped:
                lines.append(
                    "Без выбранного фильтра относительно позиции без фильтра счёт улучшают: "
                    + ", ".join(f"«{name}»" for name in alone_helped)
                    + "."
                )
            else:
                lines.append("Без выбранного фильтра те же четыре правила относительно позиции без фильтра счёт не улучшают.")
    lines.extend(
        [
            "",
            report.md(_filter_view(table), index=False),
            "",
            "Net выбранного фильтра и правил прямоты поверх него по календарным годам.\n",
            report.md(_filter_years(curves, (PAUSE_NO_CLIMAX, *overlays)), index=False),
            "",
            "В живую функцию фильтр по прямоте не переносится.",
        ]
    )
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _extra_section(series: cm.Series, study: Study) -> list[str]:
    """Фильтры, которых ещё не было: откат, продолжение, ролл, доля дней и широкий день."""
    del study
    need = int(np.ceil(DAY_SHARE * PRIMARY[0]))
    lines = [
        "## 15. Другие фильтры\n",
        (
            "Порог хода, знак 5/60/120 дней, волатильность, пауза после 2σ, громкость и прямота уже смотрелись. "
            "Здесь пять правил, которых в тех таблицах не было. "
            "Откат — в день решения дневной ход против знака L, следующие H дней берётся этот знак. "
            "Продолжение — дневной ход в тот же знак. "
            "Без ролла — в день смены контракта позиция ноль: дата ролла известна заранее, контанго в доходность и так не входит, но круг на ролле списывается. "
            f"60% дней — не меньше {need} дней из {PRIMARY[0]} закрылись в знак сигнала. "
            f"Без широкого дня — (high − low) / close дня решения больше {RANGE_SPIKE:.0f} медиан предыдущих {VOLUME_WINDOW} дней, и следующие H дней позиция не открывается. "
            "Пороги заданы заранее: 60% и та же двойка, что у всплеска объёма. Правила друг с другом не складываются."
        ),
    ]
    volume = contract_daily_volume(series)
    day_range = contract_daily_range(series)
    if not np.any(volume > 0):
        lines.append("Для сравнения с выбранным фильтром нужен дневной объём, на этом ряду его нет.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    table, curves = extra_book(series, volume, day_range)
    base = _filter_named(table, PAUSE_NO_CLIMAX)
    overlays = (
        "выбранный и откат",
        "выбранный и продолжение",
        "выбранный и 60% дней",
        "выбранный и без широкого дня",
        "выбранный и без ролла",
    )
    alone = ("откат", "продолжение", "60% дней", "без широкого дня", "без ролла")
    lines.append(
        f"Счёт тот же: {_lots(ACCOUNT_CASH)} ₽, плечо {CHOSEN_LEVERAGE}, фильтр «{PAUSE_NO_CLIMAX}». "
        "Знак Si и Eu в отдельный фильтр не добавлялся: в первом отчёте он совпадает со знаком CR на 97% и 90%."
    )
    if base is not None:
        helped = [name for name in overlays if (row := _filter_named(table, name)) is not None and _volume_helps(row, base)]
        if helped:
            lines.append(
                "На выбранном фильтре счёт улучшают: "
                + ", ".join(f"«{name}»" for name in helped)
                + ". Улучшение — финиш выше без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при финише не ниже 98% выбранного счёта."
            )
        else:
            lines.append(
                "На выбранном фильтре ни одно из пяти правил счёт не улучшает: "
                "финиш не растёт без более глубокой просадки, и просадка не становится меньше хотя бы на 2 пункта при близком финише."
            )
        positions = dict(extra_filter_positions(series, volume, day_range))
        percent = daily_percent(series)
        signed = positions[PAUSE_NO_CLIMAX] * percent
        bits = []
        for name in overlays:
            off = (positions[PAUSE_NO_CLIMAX] != 0) & (positions[name] == 0)
            kept = (positions[PAUSE_NO_CLIMAX] != 0) & (positions[name] != 0)
            bits.append(
                f"«{name}» снимает дни со средним {_comma(_signed_mean(signed, off) * 100, 2, signed=True)}% "
                f"против {_comma(_signed_mean(signed, kept) * 100, 2, signed=True)}% у оставшихся"
            )
        lines.append("На выбранной позиции " + "; ".join(bits) + ".")
        bare = _filter_named(table, "без фильтра")
        if bare is not None:
            alone_helped = [name for name in alone if (row := _filter_named(table, name)) is not None and _volume_helps(row, bare)]
            if alone_helped:
                lines.append(
                    "Без выбранного фильтра относительно позиции без фильтра счёт улучшают: "
                    + ", ".join(f"«{name}»" for name in alone_helped)
                    + "."
                )
            else:
                lines.append("Без выбранного фильтра те же пять правил относительно позиции без фильтра счёт не улучшают.")
    lines.extend(
        [
            "",
            report.md(_filter_view(table), index=False),
            "",
            "Net выбранного фильтра и новых правил поверх него по календарным годам.\n",
            report.md(_filter_years(curves, (PAUSE_NO_CLIMAX, *overlays)), index=False),
            "",
            f"Из этих правил в выбранный фильтр «{CHOSEN_FILTER}» входит только широкий день. "
            "Остальные четыре в правило не входят. В живую функцию ничего не переносится.",
        ]
    )
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


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


def _rub(value: float, digits: int = 0, signed: bool = False) -> str:
    """Рубли с пробелом между тысячами. Знак только если signed."""
    if value != value:
        return "—"
    sign = "+" if signed and value > 0 else ("-" if value < 0 else "")
    text = f"{abs(value):,.{digits}f}".replace(",", " ").replace(".", ",")
    return sign + text


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
        *_account_section(series, study),
        *_leverage_section(series, study),
        *_filter_section(series, study),
        *_chosen_section(series, study),
        *_volume_section(series, study),
        *_straight_section(series, study),
        *_extra_section(series, study),
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


def _month_end(frame: pd.DataFrame, months: list[str], column: str = "equity") -> list[float]:
    """Значение на конец месяца. До первого месяца ряда — стартовое, дальше держится последнее."""
    monthly = frame.groupby(frame["day"].dt.to_period("M"), sort=True)[column].last()
    out: list[float] = []
    last = ACCOUNT_CASH
    for label in months:
        period = pd.Period(label, freq="M")
        if period in monthly.index:
            last = float(monthly.loc[period])
        out.append(round(last, 2))
    return out


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
        "account": _account_payload(series, study),
        "leverage": _leverage_payload(series, study),
        "filters": _filter_payload(series, study),
        "volume": _volume_payload(series, study),
        "straight": _straight_payload(series, study),
        "extra": _extra_payload(series, study),
    }


def _account_payload(series: cm.Series, study: Study) -> dict[str, object]:
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    book = describe_account(frame, series.level)
    return {
        "cash": ACCOUNT_CASH,
        "risk": ACCOUNT_RISK,
        "margin_day": EXCHANGE_MARGIN_DAY,
        "margin_rub": EXCHANGE_MARGIN_RUB,
        "margin_rate": round(MARGIN_RATE, 6),
        "chosen": book.chosen,
        "per_contract_dd": round(book.per_contract_dd, 2),
        "max_by_margin": book.max_by_margin,
        "dd_from": book.dd_from,
        "dd_to": book.dd_to,
        "go_min": round(book.go_min, 2),
        "go_max": round(book.go_max, 2),
        "by_year": [{key: _json_value(value) for key, value in row.items()} for _, row in book.by_year.iterrows()],
        "grid": [{key: _json_value(value) for key, value in row.items()} for _, row in book.grid.iterrows()],
    }


def _leverage_payload(series: cm.Series, study: Study) -> dict[str, object]:
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    table, curves = leverage_table(frame, series.level, series.ret, series.roll)
    months = [str(period) for period in frame.groupby(frame["day"].dt.to_period("M"), sort=True).groups]
    return {
        "rows": [{key: _json_value(value) for key, value in row.items()} for _, row in table.iterrows()],
        "months": months,
        "equity": {str(leverage): _month_end(curve, months) for leverage, curve in curves.items()},
    }


def _filter_payload(series: cm.Series, study: Study) -> dict[str, object]:
    table, curves = filter_book(series)
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    months = [str(period) for period in frame.groupby(frame["day"].dt.to_period("M"), sort=True).groups]
    shown = ("без фильтра", "порог 50 б.п.", "порог 100 б.п.", PRICE_PAUSE)
    return {
        "leverage": CHOSEN_LEVERAGE,
        "chosen": CHOSEN_FILTER,
        "price_pause": PRICE_PAUSE,
        "shock": FILTER_SHOCK,
        "rows": [{key: _json_value(value) for key, value in row.items()} for _, row in table.iterrows()],
        "months": months,
        "equity": {name: _month_end(curves[name], months) for name in shown if name in curves},
    }


def _volume_payload(series: cm.Series, study: Study) -> dict[str, object]:
    volume = contract_daily_volume(series)
    if not np.any(volume > 0):
        return {"available": False}
    table, curves = volume_book(series, volume)
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    months = [str(period) for period in frame.groupby(frame["day"].dt.to_period("M"), sort=True).groups]
    shown = (
        PRICE_PAUSE,
        "пауза и громкое решение",
        PAUSE_NO_CLIMAX,
        "пауза и растущий объём",
        "пауза и пауза объёма",
    )
    return {
        "available": True,
        "leverage": CHOSEN_LEVERAGE,
        "window": VOLUME_WINDOW,
        "spike": VOLUME_SPIKE,
        "rows": [{key: _json_value(value) for key, value in row.items()} for _, row in table.iterrows()],
        "months": months,
        "equity": {name: _month_end(curves[name], months) for name in shown if name in curves},
    }


def _straight_payload(series: cm.Series, study: Study) -> dict[str, object]:
    volume = contract_daily_volume(series)
    if not np.any(volume > 0):
        return {"available": False}
    table, curves = straight_book(series, volume)
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    months = [str(period) for period in frame.groupby(frame["day"].dt.to_period("M"), sort=True).groups]
    shown = (
        PAUSE_NO_CLIMAX,
        "выбранный и коридор 0,15–0,5",
        "выбранный и прямее 0,5",
        "выбранный и прямее 0,25",
        "выбранный и пила ниже 0,25",
    )
    return {
        "available": True,
        "leverage": CHOSEN_LEVERAGE,
        "low": STRAIGHT_LOW,
        "high": STRAIGHT_HIGH,
        "floor": STRAIGHT_FLOOR,
        "rows": [{key: _json_value(value) for key, value in row.items()} for _, row in table.iterrows()],
        "months": months,
        "equity": {name: _month_end(curves[name], months) for name in shown if name in curves},
    }


def _extra_payload(series: cm.Series, study: Study) -> dict[str, object]:
    volume = contract_daily_volume(series)
    day_range = contract_daily_range(series)
    if not np.any(volume > 0):
        return {"available": False}
    table, curves = extra_book(series, volume, day_range)
    frame = study.frames[("hold", PRIMARY[0], PRIMARY[1])]
    months = [str(period) for period in frame.groupby(frame["day"].dt.to_period("M"), sort=True).groups]
    shown = (
        PAUSE_NO_CLIMAX,
        "выбранный и откат",
        "выбранный и продолжение",
        "выбранный и 60% дней",
        "выбранный и без широкого дня",
        "выбранный и без ролла",
    )
    equity = {name: _month_end(curves[name], months) for name in shown if name in curves}
    if "выбранный и без широкого дня" in curves:
        equity[CHOSEN_FILTER] = equity["выбранный и без широкого дня"]
    return {
        "available": True,
        "leverage": CHOSEN_LEVERAGE,
        "locked": CHOSEN_FILTER,
        "day_share": DAY_SHARE,
        "range_spike": RANGE_SPIKE,
        "rows": [{key: _json_value(value) for key, value in row.items()} for _, row in table.iterrows()],
        "months": months,
        "equity": equity,
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
    account = describe_account(study.frames[("hold", PRIMARY[0], PRIMARY[1])], series.level)
    account.grid.to_csv(report.OUT / "out" / "cr_momentum_account.csv", index=False)
    leverage, _ = leverage_table(study.frames[("hold", PRIMARY[0], PRIMARY[1])], series.level, series.ret, series.roll)
    leverage.to_csv(report.OUT / "out" / "cr_momentum_leverage.csv", index=False)
    filters, _ = filter_book(series)
    filters.to_csv(report.OUT / "out" / "cr_momentum_filters.csv", index=False)
    volume = contract_daily_volume(series)
    if np.any(volume > 0):
        volume_table, _ = volume_book(series, volume)
        volume_table.to_csv(report.OUT / "out" / "cr_momentum_volume.csv", index=False)
        straight_table, _ = straight_book(series, volume)
        straight_table.to_csv(report.OUT / "out" / "cr_momentum_straight.csv", index=False)
        extra_table, _ = extra_book(series, volume, contract_daily_range(series))
        extra_table.to_csv(report.OUT / "out" / "cr_momentum_extra.csv", index=False)
    from research.cr_momentum_mix import main as mix_main
    from research.cr_momentum_senior import main as senior_main
    from research.cr_momentum_veto import main as veto_main

    mix_text = mix_main()
    if mix_text:
        path = report.OUT / "REPORT_cr_momentum_params.md"
        path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" + mix_text, encoding="utf-8")
    senior_main()
    veto_main()


if __name__ == "__main__":
    main()
