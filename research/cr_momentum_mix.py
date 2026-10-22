"""Как выбранный моментум стоит рядом с пятью минутными окнами.

Окна те же, что гоняет `backtest.py`: отдельные счета по 100 000 ₽, не один общий.
Моментум — свой счёт 100 000 ₽, плечо 3, фильтр «пауза, без кульминации и без широкого дня».
Два способа стыковки заданы заранее и не складываются: сумма отдельных счетов и запрет
минутной цели, если её знак не совпадает со знаком моментума этого дня. Знак моментума
известен до открытия дня. В живую функцию ничего не переносится.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import backtest
from cnyrub.engine import _FillBook, marked_equity, step_minute

import research.core as core
import research.cr_momentum as cm
import research.cr_momentum_params as cp
import research.report as report

NAMES = ("short", "long", "thirty", "fortyfive", "sixty")
LABELS = {
    "short": "короткое 525",
    "long": "длинное 12 420",
    "thirty": "30 минут",
    "fortyfive": "45 минут",
    "sixty": "60 минут",
}
MOMENTUM = "моментум"
FIVE = "пять окон"
GATED = "знак моментума"


def clamp_target(target: int, allowed: int | None) -> int:
    """Минутная цель остаётся, только если её знак совпадает с разрешённым. None — не трогать."""
    if allowed is None or target == 0:
        return target
    if allowed == 0 or (target > 0) != (allowed > 0):
        return 0
    return target


def _window(name: str) -> backtest.Window:
    return backtest.WINDOWS[NAMES.index(name)]


def _momentum_book(series: cm.Series) -> tuple[pd.DataFrame, dict[date, int]]:
    """Дневной счёт выбранного фильтра и знак позиции, известный на этот день."""
    volume = cp.contract_daily_volume(series)
    day_range = cp.contract_daily_range(series)
    position = cp.chosen_account_position(series, volume, day_range)
    frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
    curve = cp.leverage_curve(frame, series.level, series.ret, series.roll, cp.CHOSEN_LEVERAGE, cp.ACCOUNT_CASH)
    curve = curve.copy()
    curve["day"] = pd.to_datetime(curve["day"]).dt.date
    signs = {day: int(np.sign(value)) for day, value in zip(curve["day"], position, strict=True)}
    return curve, signs


def _replay_contract(
    secid: str,
    frame: pd.DataFrame,
    window: backtest.Window,
    *,
    cash: float,
    trade_from: date | None,
    day_sign: dict[date, int] | None,
    view: pd.DataFrame | None = None,
) -> tuple[float, list[dict[str, float | date | int]]]:
    """Один контракт. Дни до trade_from только прогревают окно и в таблицу не попадают."""
    if view is None:
        view = backtest.channel_view(frame, window.channel, window.exit_channel)
    opened = view["open"].to_numpy(dtype=float)
    high = view["high"].to_numpy(dtype=float)
    low = view["low"].to_numpy(dtype=float)
    close = view["close"].to_numpy(dtype=float)
    volume = view["volume"].to_numpy(dtype=float)
    prior_high = view["prior_high"].to_numpy(dtype=float)
    prior_low = view["prior_low"].to_numpy(dtype=float)
    prior_vol = view["prior_vol"].to_numpy(dtype=float)
    prior_range = view["prior_range"].to_numpy(dtype=float)
    exit_high = view["exit_high"].to_numpy(dtype=float)
    exit_low = view["exit_low"].to_numpy(dtype=float)
    clock_typical = view["clock_vol"].to_numpy(dtype=float)
    ready = view["entry_ready"].to_numpy(dtype=float)
    drift = view["drift"].to_numpy(dtype=float)
    path = view["path"].to_numpy(dtype=float)
    surge_vol = view["surge_vol"].to_numpy(dtype=float)
    days = view.index.date
    last_day = days[-1]
    book = _FillBook(secid, cash)
    rows: list[dict[str, float | date | int]] = []
    current: date | None = None
    start_equity = cash
    exposure = 0
    minutes = 0
    held_minutes = 0
    last_equity = cash
    last_held = 0

    def close_day() -> None:
        if current is None:
            return
        rows.append(
            {
                "day": current,
                "net": last_equity - start_equity,
                "equity": last_equity,
                "exposure": exposure,
                "minutes": minutes,
                "held_minutes": held_minutes,
                "eod": last_held,
            }
        )

    for i in range(len(view)):
        day = days[i]
        trading = trade_from is None or day >= trade_from
        if trading and day != current:
            close_day()
            current = day
            start_equity = last_equity
            exposure = 0
            minutes = 0
            held_minutes = 0
        allowed = None if day_sign is None else day_sign.get(day, None)
        book.target = clamp_target(int(book.target), allowed)
        step_minute(
            book,
            i,
            opened=float(opened[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(volume[i]),
            day=day,
            next_day=None if i + 1 >= len(view) else days[i + 1],
            last_day=last_day,
            prior_high=float(prior_high[i]),
            prior_low=float(prior_low[i]),
            prior_vol=float(prior_vol[i]),
            prior_range=float(prior_range[i]),
            exit_high=float(exit_high[i]),
            exit_low=float(exit_low[i]),
            clock_vol=float(clock_typical[i]),
            entry_ready=float(ready[i]),
            clearance=0.0,
            cooldown=0,
            clock_volume=False,
            clock_cap=window.clock_cap,
            size_mode=window.size_mode,
            stop_mult=window.stop_mult,
            loss_bars=window.loss_bars,
            risk_fraction=window.risk_fraction,
            margin=window.margin,
            stop_rub=window.stop_rub,
            breakout_span=window.breakout_span,
            trade_from=trade_from,
            trail=False,
            scale_step=window.scale_step,
            scale_floor=window.scale_floor,
            scale_back=window.scale_back,
            drift=float(drift[i]),
            path=float(path[i]),
            surge_vol=float(surge_vol[i]),
            eff_low=window.eff_low,
            eff_high=window.eff_high,
            surge_cap=window.surge_cap,
            leverage=window.leverage,
            min_stop=None,
            skip_flat=False,
        )
        if not trading:
            continue
        last_equity = marked_equity(book, float(close[i]))
        last_held = int(book.held)
        exposure += last_held
        minutes += 1
        if last_held:
            held_minutes += 1
    close_day()
    settled = marked_equity(book, float(close[-1])) if len(view) else cash
    return settled, rows


def replay_window(
    name: str,
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    day_sign: dict[date, int] | None = None,
    views: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Счёт одного окна по контрактам подряд. day_sign включает фильтр знака моментума."""
    window = _window(name)
    equity = float(window.cash)
    rows: list[dict[str, float | date | int]] = []
    built = views if views is not None else {}
    for secid, frame in frames.items():
        print(f"{name} {secid}", flush=True)
        view = built.get(secid)
        if view is None:
            view = backtest.channel_view(frame, window.channel, window.exit_channel)
            built[secid] = view
        equity, part = _replay_contract(
            secid,
            frame,
            window,
            cash=equity,
            trade_from=starts.get(secid),
            day_sign=day_sign,
            view=view,
        )
        rows.extend(part)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["strategy"] = name
    out["mode"] = "как есть" if day_sign is None else GATED
    return out


def _describe(equity: np.ndarray, net: np.ndarray, cash: float, in_market: float) -> dict[str, float]:
    peak_equity = np.maximum.accumulate(equity) if len(equity) else np.array([cash])
    drawdown = peak_equity - equity if len(equity) else np.array([0.0])
    worst = int(drawdown.argmax()) if len(drawdown) else 0
    dd = float(drawdown[worst]) if len(drawdown) else 0.0
    peak = float(equity[worst] + dd) if len(equity) else cash
    previous = np.r_[cash, equity[:-1]] if len(equity) else np.array([cash])
    returns = np.divide(net, previous, out=np.zeros_like(net, dtype=float), where=previous > 0)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    return {
        "end": float(equity[-1]) if len(equity) else cash,
        "net": float(net.sum()) if len(net) else 0.0,
        "max_dd": dd,
        "max_dd_pct_peak": dd / peak * 100.0 if peak else float("nan"),
        "sharpe": float(returns.mean() / std * np.sqrt(252)) if std and std == std and std > 0 else float("nan"),
        "in_market": in_market,
        "cash": cash,
    }


def _cagr_helps(row: dict[str, float], base: dict[str, float]) -> bool:
    """Годовой процент выше без более глубокой просадки, либо просадка меньше на 2 пункта при годовом проценте не ниже 98%."""
    higher = row["cagr"] > base["cagr"] and row["max_dd_pct_peak"] <= base["max_dd_pct_peak"] + 0.05
    calmer = row["cagr"] >= base["cagr"] * 0.98 and row["max_dd_pct_peak"] <= base["max_dd_pct_peak"] - 2.0
    return bool(higher or calmer)


def _years_between(days: pd.Series) -> float:
    if days.empty:
        return 1.0
    return max((pd.Timestamp(days.iloc[-1]) - pd.Timestamp(days.iloc[0])).days / 365.25, 1 / 365.25)


def _path_summary(frame: pd.DataFrame, cash: float, in_market: float) -> dict[str, float]:
    equity = frame["equity"].to_numpy(dtype=float)
    net = frame["net"].to_numpy(dtype=float)
    summary = _describe(equity, net, cash, in_market)
    summary.update(cp._path_rates(summary, cash, _years_between(frame["day"])))
    return summary


def _align(frames: list[pd.DataFrame], cash_each: list[float]) -> pd.DataFrame:
    """Сумма счетов по дням. Пропущенный день несёт прошлый капитал и ноль результата."""
    days = sorted({day for frame in frames for day in frame["day"]})
    index = pd.Index(days, name="day")
    equity = np.zeros(len(index))
    net = np.zeros(len(index))
    for frame, cash in zip(frames, cash_each, strict=True):
        part = frame.drop_duplicates("day").set_index("day").reindex(index)
        carried = part["equity"].ffill()
        before = carried.isna()
        carried = carried.fillna(cash)
        equity += carried.to_numpy(dtype=float)
        step = carried.diff()
        step.iloc[0] = carried.iloc[0] - cash
        step = step.where(~before, 0.0)
        net += step.to_numpy(dtype=float)
    return pd.DataFrame({"day": days, "equity": equity, "net": net})


def _returns(frame: pd.DataFrame, cash: float) -> pd.Series:
    equity = frame["equity"].to_numpy(dtype=float)
    net = frame["net"].to_numpy(dtype=float)
    previous = np.r_[cash, equity[:-1]]
    returns = np.divide(net, previous, out=np.zeros(len(net)), where=previous > 0)
    return pd.Series(returns, index=pd.Index(frame["day"].to_list(), name="day"))


def _correlation(left: pd.Series, right: pd.Series) -> float:
    both = pd.concat([left.rename("l"), right.rename("r")], axis=1, join="inner").dropna()
    if len(both) < 3 or float(both["l"].std(ddof=1)) == 0 or float(both["r"].std(ddof=1)) == 0:
        return float("nan")
    return float(both["l"].corr(both["r"]))


def _in_market(frame: pd.DataFrame) -> float:
    minutes = float(frame["minutes"].sum()) if "minutes" in frame else 0.0
    held = float(frame["held_minutes"].sum()) if "held_minutes" in frame else 0.0
    if minutes <= 0:
        return float("nan")
    return held / minutes * 100.0


def _agreement(minute: pd.DataFrame, momentum: pd.DataFrame) -> dict[str, float]:
    """Дни, где оба были в позиции: доля совпадения знака и средний дневной ход минутного счёта."""
    left = minute.drop_duplicates("day").set_index("day")
    right = momentum.drop_duplicates("day").set_index("day")[["sign", "ret"]].rename(
        columns={"sign": "momentum_sign", "ret": "momentum_ret"}
    )
    both = left.join(right, how="inner")
    if both.empty:
        return {
            "active_days": 0.0,
            "agree_pct": float("nan"),
            "agree_ret": float("nan"),
            "disagree_ret": float("nan"),
            "flat_ret": float("nan"),
            "flat_days": 0.0,
        }
    minute_sign = np.sign(both["exposure"].to_numpy(dtype=float))
    momentum_sign = both["momentum_sign"].to_numpy(dtype=float)
    active = (minute_sign != 0) & (momentum_sign != 0)
    agree = active & (minute_sign == momentum_sign)
    disagree = active & (minute_sign != momentum_sign)
    flat = (momentum_sign == 0) & (minute_sign != 0)
    previous = both["equity"].shift(1)
    previous.iloc[0] = both["equity"].iloc[0] - both["net"].iloc[0]
    base = previous
    daily = both["net"] / base.where(base > 0)
    return {
        "active_days": float(active.sum()),
        "agree_pct": float(agree.sum() / active.sum() * 100.0) if active.any() else float("nan"),
        "agree_ret": float(daily[agree].mean() * 100.0) if agree.any() else float("nan"),
        "disagree_ret": float(daily[disagree].mean() * 100.0) if disagree.any() else float("nan"),
        "flat_ret": float(daily[flat].mean() * 100.0) if flat.any() else float("nan"),
        "flat_days": float(flat.sum()),
    }


def build_mix(series: cm.Series | None = None) -> dict[str, object]:
    """Прогон пяти окон как есть и с фильтром знака, плюс суммы счетов."""
    core.configure(1000.0, 0.001)
    series = cm.continuous() if series is None else series
    momentum, signs = _momentum_book(series)
    momentum = momentum.copy()
    momentum["sign"] = [signs[day] for day in momentum["day"]]
    momentum["ret"] = _returns(momentum, cp.ACCOUNT_CASH).to_numpy()
    frames = backtest.load_minutes()
    starts = backtest.load_trade_starts()
    plain: dict[str, pd.DataFrame] = {}
    gated: dict[str, pd.DataFrame] = {}
    for name in NAMES:
        views: dict[str, pd.DataFrame] = {}
        plain[name] = replay_window(name, frames, starts, None, views)
        gated[name] = replay_window(name, frames, starts, signs, views)
    return {"momentum": momentum, "plain": plain, "gated": gated}


def _book_table(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    out = []
    for row in rows:
        out.append(
            {
                "счёт": row["name"],
                "старт, ₽": cp._rub(float(row["cash"])),
                "финиш, ₽": cp._rub(float(row["end"])),
                "год, %": cp._comma(float(row["cagr"]), 1),
                "просадка, % пика": cp._comma(float(row["max_dd_pct_peak"]), 1),
                "Калмар": cp._comma(float(row["calmar"]), 2),
                "Шарп": cp._comma(float(row["sharpe"]), 2, signed=True),
            }
        )
    return pd.DataFrame(out)


def _year_table(books: dict[str, pd.DataFrame], cash: dict[str, float]) -> pd.DataFrame:
    changes = {}
    for name, frame in books.items():
        changes[name] = cp._year_changes(
            pd.DataFrame({"day": pd.to_datetime(frame["day"]), "equity": frame["equity"]}),
            cash[name],
        )
    years = sorted({year for column in changes.values() for year in column})
    rows = []
    for year in years:
        item: dict[str, object] = {"год": year}
        for name in books:
            item[name] = cp._rub(changes[name].get(year, float("nan")), signed=True)
        rows.append(item)
    return pd.DataFrame(rows)


def mix_section(bundle: dict[str, object] | None = None, series: cm.Series | None = None) -> list[str]:
    """Текст раздела: корреляция, сумма счетов и фильтр знака."""
    lines = [
        "## 16. Рядом с пятью окнами\n",
        (
            "Пять окон — короткое 525, длинное 12 420, 30, 45 и 60 минут — те же правила, что в `backtest.py`. "
            "У каждого свой счёт 100 000 ₽ и свой залог 1 000 ₽. Сумма окон — это пять счетов рядом, не одна позиция. "
            f"Моментум — отдельный счёт {cp._lots(cp.ACCOUNT_CASH)} ₽, плечо {cp.CHOSEN_LEVERAGE}, фильтр «{cp.CHOSEN_FILTER}». "
            "Два способа стыковки заданы заранее и друг с другом не складываются. "
            "Первый — держать счета рядом и сложить капитал. "
            "Второй — перед каждой минутой обнулять минутную цель, если её знак не совпадает со знаком моментума этого дня. "
            "Знак моментума на день известен с прошлого закрытия. Если моментум в этот день без позиции, минутное окно тоже выходит в ноль."
        ),
    ]
    if not backtest.BARS_DIR.exists():
        lines.append("Минутных файлов нет, стыковку посчитать нельзя.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    bundle = build_mix(series) if bundle is None else bundle
    momentum = bundle["momentum"]
    plain: dict[str, pd.DataFrame] = bundle["plain"]
    gated: dict[str, pd.DataFrame] = bundle["gated"]
    assert isinstance(momentum, pd.DataFrame)
    momentum_summary = _path_summary(momentum, cp.ACCOUNT_CASH, float((momentum["sign"] != 0).mean() * 100.0))
    momentum_ret = pd.Series(momentum["ret"].to_numpy(), index=pd.Index(momentum["day"].to_list()))
    books: dict[str, tuple[pd.DataFrame, float]] = {MOMENTUM: (momentum, cp.ACCOUNT_CASH)}
    pair_rows: list[dict[str, float | str]] = [
        {"name": MOMENTUM, **momentum_summary},
    ]
    overlap_rows = []
    gate_rows = []
    helped: list[str] = []
    beside: list[str] = []
    flat_plus: list[str] = []
    against_minus: list[str] = []
    against_plus: list[str] = []
    for name in NAMES:
        label = LABELS[name]
        raw = plain[name]
        held = gated[name]
        raw_summary = _path_summary(raw, float(_window(name).cash), _in_market(raw))
        gate_summary = _path_summary(held, float(_window(name).cash), _in_market(held))
        pair = _align([raw, momentum], [float(_window(name).cash), cp.ACCOUNT_CASH])
        pair_summary = _path_summary(pair, float(_window(name).cash) + cp.ACCOUNT_CASH, float("nan"))
        books[label] = (raw, float(_window(name).cash))
        books[f"{label} и {MOMENTUM}"] = (pair, float(_window(name).cash) + cp.ACCOUNT_CASH)
        pair_rows.append({"name": label, **raw_summary})
        pair_rows.append({"name": f"{label} и {MOMENTUM}", **pair_summary})
        corr = _correlation(_returns(raw, float(_window(name).cash)), momentum_ret)
        stats = _agreement(raw, momentum)
        overlap_rows.append(
            {
                "окно": label,
                "корреляция": cp._comma(corr, 2, signed=True),
                "дни оба в позиции": cp._rub(stats["active_days"]),
                "тот же знак, %": cp._comma(stats["agree_pct"], 0),
                "ход в тот же знак, %": cp._comma(stats["agree_ret"], 3, signed=True),
                "ход против, %": cp._comma(stats["disagree_ret"], 3, signed=True),
                "ход при паузе моментума, %": cp._comma(stats["flat_ret"], 3, signed=True),
            }
        )
        gate_rows.append(
            {
                "окно": label,
                "финиш как есть, ₽": cp._rub(raw_summary["end"]),
                "финиш со знаком, ₽": cp._rub(gate_summary["end"]),
                "просадка как есть, %": cp._comma(raw_summary["max_dd_pct_peak"], 1),
                "просадка со знаком, %": cp._comma(gate_summary["max_dd_pct_peak"], 1),
                "Шарп как есть": cp._comma(raw_summary["sharpe"], 2, signed=True),
                "Шарп со знаком": cp._comma(gate_summary["sharpe"], 2, signed=True),
            }
        )
        if cp._volume_helps(pd.Series(gate_summary), pd.Series(raw_summary)):
            helped.append(label)
        if _cagr_helps(pair_summary, raw_summary):
            beside.append(label)
        if stats["flat_ret"] == stats["flat_ret"] and stats["flat_ret"] > 0:
            flat_plus.append(label)
        if stats["disagree_ret"] == stats["disagree_ret"] and stats["disagree_ret"] < 0:
            against_minus.append(label)
        elif stats["disagree_ret"] == stats["disagree_ret"] and stats["disagree_ret"] > 0:
            against_plus.append(label)
    five = _align([plain[name] for name in NAMES], [float(_window(name).cash) for name in NAMES])
    five_cash = 5 * float(backtest.WINDOWS[0].cash)
    five_summary = _path_summary(five, five_cash, float("nan"))
    both = _align([five, momentum], [five_cash, cp.ACCOUNT_CASH])
    both_cash = five_cash + cp.ACCOUNT_CASH
    both_summary = _path_summary(both, both_cash, float("nan"))
    books[FIVE] = (five, five_cash)
    books[f"{FIVE} и {MOMENTUM}"] = (both, both_cash)
    pair_rows.extend(
        [
            {"name": FIVE, **five_summary},
            {"name": f"{FIVE} и {MOMENTUM}", **both_summary},
        ]
    )
    five_ret = _returns(five, five_cash)
    five_corr = _correlation(five_ret, momentum_ret)
    portfolio_helps = _cagr_helps(both_summary, five_summary)
    if helped:
        lines.append(
            "Фильтр знака моментума улучшает минутный счёт: "
            + ", ".join(f"«{name}»" for name in helped)
            + ". Улучшение — финиш выше без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при финише не ниже 98% того же окна."
        )
    else:
        lines.append(
            "Фильтр знака моментума ни одно из пяти окон не улучшает: "
            "финиш не растёт без более глубокой просадки, и просадка не становится меньше хотя бы на 2 пункта при близком финише."
        )
    quoted = lambda names: ", ".join(f"«{name}»" for name in names)
    if against_minus:
        lines.append(f"Дни против знака моментума в среднем убыточны у окон {quoted(against_minus)}.")
    if against_plus:
        lines.append(f"У окон {quoted(against_plus)} дни против знака всё ещё в плюсе.")
    if flat_plus:
        lines.append(
            f"Дни, когда моментум без позиции, в плюсе у окон {quoted(flat_plus)}. "
            "Фильтр выключает и дни против знака, и эти дни паузы, поэтому финиш минутного счёта падает."
        )
    lines.append(
        f"Дневная доходность моментума и сумма пяти окон коррелируют на {cp._comma(five_corr, 2, signed=True)}. "
        f"Пять окон заканчивают на {cp._rub(five_summary['end'])} ₽ со старта {cp._rub(five_cash)} ₽, "
        f"просадка {cp._comma(five_summary['max_dd_pct_peak'], 1)}% пика, Шарп {cp._comma(five_summary['sharpe'], 2, signed=True)}. "
        f"Те же пять плюс моментум: старт {cp._rub(both_cash)} ₽, финиш {cp._rub(both_summary['end'])} ₽, "
        f"{cp._comma(both_summary['cagr'], 1)}% в год, просадка {cp._comma(both_summary['max_dd_pct_peak'], 1)}% пика, "
        f"Шарп {cp._comma(both_summary['sharpe'], 2, signed=True)}."
    )
    if beside:
        lines.append(
            "Рядом с отдельным окном тот же критерий по годовому проценту выполняется у окон "
            + ", ".join(f"«{name}»" for name in beside)
            + "."
        )
        if LABELS["long"] not in beside:
            lines.append("Длинное окно в этот список не входит: годовой процент пары с моментумом ниже, чем у длинного окна без него.")
    else:
        lines.append("Рядом ни с одним отдельным окном годовой процент не растёт без более глубокой просадки.")
    if portfolio_helps:
        lines.append(
            "По доходности на вложенный капитал моментум рядом с пятью окнами счёт улучшает: "
            "годовой процент выше без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при годовом проценте не ниже 98% пяти окон."
        )
    else:
        lines.append(
            "По доходности на вложенный капитал моментум рядом с пятью окнами сумму не улучшает: "
            "годовой процент не растёт без более глубокой просадки, и просадка не становится меньше хотя бы на 2 пункта при близком годовом проценте."
        )
    lines.extend(
        [
            "",
            report.md(pd.DataFrame(overlap_rows), index=False),
            "",
            "Ход — средний дневной результат минутного счёта в процентах от капитала на начало дня. "
            "«Против» — знак экспозиции за день не совпал со знаком моментума.\n",
            report.md(pd.DataFrame(gate_rows), index=False),
            "",
            "Счета рядом. «И моментум» — сумма двух капиталов, позиции не сальдируются.\n",
            report.md(_book_table(pair_rows), index=False),
            "",
            "Net моментума, пяти окон и их суммы по календарным годам.\n",
            report.md(
                _year_table(
                    {name: books[name][0] for name in (MOMENTUM, FIVE, f"{FIVE} и {MOMENTUM}")},
                    {name: books[name][1] for name in (MOMENTUM, FIVE, f"{FIVE} и {MOMENTUM}")},
                ),
                index=False,
            ),
            "",
            "В живую функцию стыковка не переносится. Окна по-прежнему торгуют отдельно от моментума.",
        ]
    )
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _json_ready(bundle: dict[str, object]) -> dict[str, object]:
    """Месячные капиталы для графика. Дневные ряды остаются в csv."""
    momentum: pd.DataFrame = bundle["momentum"]
    plain: dict[str, pd.DataFrame] = bundle["plain"]
    five = _align([plain[name] for name in NAMES], [float(_window(name).cash) for name in NAMES])
    five_cash = 5 * float(backtest.WINDOWS[0].cash)
    both = _align([five, momentum], [five_cash, cp.ACCOUNT_CASH])
    frames = {
        MOMENTUM: momentum,
        FIVE: five,
        f"{FIVE} и {MOMENTUM}": both,
    }
    months = sorted({str(pd.Timestamp(day).to_period("M")) for day in momentum["day"]})
    equity = {}
    for name, frame in frames.items():
        indexed = frame.drop_duplicates("day").set_index("day")
        indexed.index = pd.to_datetime(indexed.index)
        month_end = indexed["equity"].groupby(indexed.index.to_period("M")).last()
        equity[name] = [round(float(month_end.get(period, np.nan)), 2) for period in months]
    return {"months": months, "equity": equity}


def main() -> None:
    core.configure(1000.0, 0.001)
    series = cm.continuous()
    bundle = build_mix(series)
    text = "\n".join(mix_section(bundle, series)).strip() + "\n"
    out = report.OUT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cr_momentum_mix.md").write_text(text, encoding="utf-8")
    pieces = [bundle["momentum"].assign(strategy=MOMENTUM, mode="как есть")]
    for name in NAMES:
        pieces.append(bundle["plain"][name])
        pieces.append(bundle["gated"][name])
    pd.concat(pieces, ignore_index=True).to_csv(out / "cr_momentum_mix_daily.csv", index=False)
    (out / "cr_momentum_mix.json").write_text(json.dumps(_json_ready(bundle), ensure_ascii=False), encoding="utf-8")
    print(text)
    return text


if __name__ == "__main__":
    main()
