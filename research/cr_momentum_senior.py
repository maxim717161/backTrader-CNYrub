"""Моментум в цепочке «не открывать новую позицию против старшего окна».

Старшее окно — следующее более длинное. Новая позиция из нуля не ставится против
его знака. Если старший без позиции, младший открывается свободно. Уже открытую
позицию правило не закрывает. Счета остаются отдельными. В живую функцию ничего
не переносится: в коде пяти окон такого запрета нет, здесь проверяется сама схема.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import backtest
from research.cr_momentum_mix import (
    LABELS,
    MOMENTUM,
    NAMES,
    _align,
    _cagr_helps,
    _in_market,
    _path_summary,
    _replay_contract,
    _window,
)

import research.core as core
import research.cr_momentum as cm
import research.cr_momentum_params as cp
import research.report as report

# От младшего к старшему. Моментум вставляется в заранее выбранные места и с ними не складывается.
MINUTE_CHAIN = ("thirty", "fortyfive", "sixty", "short", "long")
PLACEMENTS = (
    ("пять окон", MINUTE_CHAIN),
    ("между коротким и длинным", ("thirty", "fortyfive", "sixty", "short", MOMENTUM, "long")),
    ("между 60 и коротким", ("thirty", "fortyfive", "sixty", MOMENTUM, "short", "long")),
    ("старше длинного", (*MINUTE_CHAIN, MOMENTUM)),
    ("младше 30 минут", (MOMENTUM, *MINUTE_CHAIN)),
)
ASKED = "между коротким и длинным"


def immediate_senior(order: tuple[str, ...], name: str) -> str | None:
    """Следующее имя к старшему краю. У самого старшего старшего нет."""
    index = order.index(name)
    if index + 1 >= len(order):
        return None
    return order[index + 1]


def block_momentum_opens(
    position: np.ndarray,
    days: np.ndarray,
    senior_eod: dict[date, int],
) -> tuple[np.ndarray, int, np.ndarray]:
    """День не открывает моментум против знака старшего на закрытии предыдущего дня.

    Продолжение того же знака остаётся, даже если старший за это время развернулся.
    Возвращает позицию, число запрещённых входов подряд как один запрет и маску таких дней.
    """
    out = np.array(position, dtype=float, copy=True)
    blocked_days = np.zeros(len(out), dtype=bool)
    previous = 0.0
    episodes = 0
    streak = False
    for index, day in enumerate(days):
        desired = float(out[index])
        senior = 0
        if index > 0:
            senior = int(senior_eod.get(pd.Timestamp(days[index - 1]).date(), 0))
        opening = (previous == 0.0 and desired != 0.0) or previous * desired < 0.0
        if opening and senior != 0 and np.sign(desired) != np.sign(senior):
            out[index] = 0.0
            blocked_days[index] = True
            if not streak:
                episodes += 1
                streak = True
        else:
            streak = False
        previous = float(out[index])
    return out, episodes, blocked_days


def _ensure_views(
    name: str,
    frames: dict[str, pd.DataFrame],
    cache: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    built = cache.get(name)
    if built is not None:
        return built
    window = _window(name)
    built = {}
    for secid, frame in frames.items():
        print(f"окно {LABELS[name]} {secid}", flush=True)
        built[secid] = backtest.channel_view(frame, window.channel, window.exit_channel)
    cache[name] = built
    return built


def _replay_minute(
    name: str,
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    views: dict[str, pd.DataFrame],
    senior_bars: dict[str, np.ndarray] | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[date, int], int]:
    """Счёт окна. senior_bars — знак старшего на каждой минуте того же контракта."""
    window = _window(name)
    equity = float(window.cash)
    rows: list[dict[str, float | date | int]] = []
    held_map: dict[str, np.ndarray] = {}
    episodes = 0
    for secid, frame in frames.items():
        view = views[secid]
        senior = None if senior_bars is None else senior_bars[secid]
        if senior is not None and len(senior) != len(view):
            raise ValueError(f"{name} {secid}: знак старшего не совпал с минутками")
        held = np.zeros(len(view), dtype=np.int32)
        box: list[int] = []
        print(f"{LABELS[name]} {secid}", flush=True)
        equity, part = _replay_contract(
            secid,
            frame,
            window,
            cash=equity,
            trade_from=starts.get(secid),
            day_sign=None,
            view=view,
            entry_senior=senior,
            held_out=held,
            blocked_box=box,
        )
        rows.extend(part)
        held_map[secid] = held
        episodes += sum(box)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["strategy"] = name
    eod = {row["day"]: int(np.sign(row["eod"])) for row in rows}
    return out, held_map, eod, episodes


def _bars_from_days(view: pd.DataFrame, sign_by_day: dict[date, int]) -> np.ndarray:
    days = pd.Series(view.index.date)
    mapped = days.map(sign_by_day).fillna(0)
    return mapped.to_numpy(dtype=np.int8)


def _momentum_curve(series: cm.Series, position: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
    curve = cp.leverage_curve(frame, series.level, series.ret, series.roll, cp.CHOSEN_LEVERAGE, cp.ACCOUNT_CASH)
    curve = curve.copy()
    curve["day"] = pd.to_datetime(curve["day"]).dt.date
    return curve


def _sign_by_day(days: np.ndarray, position: np.ndarray) -> dict[date, int]:
    return {pd.Timestamp(day).date(): int(np.sign(value)) for day, value in zip(days, position, strict=True)}


class _Books:
    """Кэш прогонов: то же окно с тем же старшим не считается второй раз."""

    def __init__(self, series: cm.Series, base_position: np.ndarray) -> None:
        self.series = series
        self.base_position = base_position
        self.days = series.days
        self.frames = backtest.load_minutes()
        self.starts = backtest.load_trade_starts()
        self.views: dict[str, dict[str, pd.DataFrame]] = {}
        self.minute_cache: dict[tuple[str, int], tuple[pd.DataFrame, dict[str, np.ndarray], dict[date, int], int]] = {}
        self.momentum_cache: dict[int, tuple[pd.DataFrame, np.ndarray, int, np.ndarray]] = {}

    def minute(
        self,
        name: str,
        senior_bars: dict[str, np.ndarray] | None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[date, int], int]:
        key = (name, 0 if senior_bars is None else id(senior_bars))
        cached = self.minute_cache.get(key)
        if cached is not None:
            print(f"{LABELS[name]} уже посчитано", flush=True)
            return cached
        views = _ensure_views(name, self.frames, self.views)
        result = _replay_minute(name, self.frames, self.starts, views, senior_bars)
        self.minute_cache[key] = result
        return result

    def momentum(
        self,
        senior_eod: dict[date, int] | None,
    ) -> tuple[pd.DataFrame, np.ndarray, int, np.ndarray]:
        key = 0 if senior_eod is None else id(senior_eod)
        cached = self.momentum_cache.get(key)
        if cached is not None:
            return cached
        if senior_eod is None:
            position = np.array(self.base_position, dtype=float, copy=True)
            episodes = 0
            mask = np.zeros(len(position), dtype=bool)
        else:
            position, episodes, mask = block_momentum_opens(self.base_position, self.days, senior_eod)
        result = (_momentum_curve(self.series, position), position, episodes, mask)
        self.momentum_cache[key] = result
        return result

    def senior_bars(self, senior: str, held: dict[str, np.ndarray] | None, position: np.ndarray | None) -> dict[str, np.ndarray] | None:
        if senior is None:
            return None
        if senior == MOMENTUM:
            assert position is not None
            signs = _sign_by_day(self.days, position)
            return {secid: _bars_from_days(frame, signs) for secid, frame in self.frames.items()}
        assert held is not None
        return held


def run_placement(order: tuple[str, ...], books: _Books) -> dict[str, dict[str, object]]:
    """Старшие считаются раньше младших. У каждого счёта один непосредственный старший."""
    ready: dict[str, dict[str, object]] = {}
    for name in reversed(order):
        senior = immediate_senior(order, name)
        if name == MOMENTUM:
            eod = None if senior is None else ready[senior]["eod"]
            curve, position, episodes, mask = books.momentum(eod if isinstance(eod, dict) else None)
            ready[name] = {"frame": curve, "position": position, "blocked": episodes, "mask": mask, "eod": None, "held": None}
            continue
        held = None if senior is None or senior == MOMENTUM else ready[senior]["held"]
        position = None if senior != MOMENTUM else ready[MOMENTUM]["position"]
        bars = books.senior_bars(senior, held if isinstance(held, dict) else None, position if isinstance(position, np.ndarray) else None)
        frame, held_map, eod_map, episodes = books.minute(name, bars)
        ready[name] = {"frame": frame, "held": held_map, "eod": eod_map, "blocked": episodes, "position": None, "mask": None}
    return ready


def _account_summary(name: str, frame: pd.DataFrame) -> dict[str, float]:
    if name == MOMENTUM:
        in_market = float((frame["contracts"] != 0).mean() * 100.0) if len(frame) else float("nan")
        return _path_summary(frame, cp.ACCOUNT_CASH, in_market)
    return _path_summary(frame, float(_window(name).cash), _in_market(frame))


def _cash(name: str) -> float:
    return cp.ACCOUNT_CASH if name == MOMENTUM else float(_window(name).cash)


def _label(name: str) -> str:
    return "моментум" if name == MOMENTUM else LABELS[name]


def build_senior(series: cm.Series | None = None) -> dict[str, object]:
    """Пять заранее заданных цепочек. База сравнения — тот же счёт без запрета."""
    core.configure(1000.0, 0.001)
    series = cm.continuous() if series is None else series
    volume = cp.contract_daily_volume(series)
    day_range = cp.contract_daily_range(series)
    base_position = cp.chosen_account_position(series, volume, day_range)
    books = _Books(series, base_position)
    free_momentum = books.momentum(None)
    free_minutes = {name: books.minute(name, None) for name in NAMES}
    placements = []
    for title, order in PLACEMENTS:
        print(f"цепочка: {title}", flush=True)
        placements.append((title, order, run_placement(order, books)))
    percent = cp.daily_percent(series)
    return {
        "series": series,
        "percent": percent,
        "base_position": base_position,
        "free_momentum": free_momentum,
        "free_minutes": free_minutes,
        "placements": placements,
    }


def _helps_text(summary: dict[str, float], base: dict[str, float], same: bool) -> str:
    if same:
        return "без изменений"
    if cp._volume_helps(pd.Series(summary), pd.Series(base)):
        return "да"
    return "нет"


def _same_account(summary: dict[str, float], base: dict[str, float], blocked: int) -> bool:
    return blocked == 0 and abs(summary["end"] - base["end"]) < 0.5


def _rows_for(
    title: str,
    order: tuple[str, ...],
    ready: dict[str, dict[str, object]],
    free_momentum: tuple[pd.DataFrame, np.ndarray, int, np.ndarray],
    free_minutes: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], dict[date, int], int]],
) -> list[dict[str, object]]:
    rows = []
    for name in order:
        frame = ready[name]["frame"]
        assert isinstance(frame, pd.DataFrame)
        summary = _account_summary(name, frame)
        blocked = int(ready[name]["blocked"])
        if name == MOMENTUM:
            base_frame = free_momentum[0]
        else:
            base_frame = free_minutes[name][0]
        base = _account_summary(name, base_frame)
        same = _same_account(summary, base, blocked)
        senior = immediate_senior(order, name)
        rows.append(
            {
                "place": title,
                "name": name,
                "label": _label(name),
                "senior": "—" if senior is None else _label(senior),
                "summary": summary,
                "base": base,
                "blocked": blocked,
                "same": same,
                "helps": _helps_text(summary, base, same),
                "frame": frame,
                "cash": _cash(name),
            }
        )
    return rows


def _table(rows: list[dict[str, object]]) -> pd.DataFrame:
    out = []
    for row in rows:
        summary = row["summary"]
        assert isinstance(summary, dict)
        out.append(
            {
                "место": row["place"],
                "счёт": row["label"],
                "старший": row["senior"],
                "финиш, ₽": cp._rub(float(summary["end"])),
                "год, %": cp._comma(float(summary["cagr"]), 1),
                "просадка, % пика": cp._comma(float(summary["max_dd_pct_peak"]), 1),
                "Шарп": cp._comma(float(summary["sharpe"]), 2, signed=True),
                "запретов": cp._lots(float(row["blocked"])),
                "улучшает свой счёт": row["helps"],
            }
        )
    return pd.DataFrame(out)


def _portfolio(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, float, dict[str, float]]:
    frames = [row["frame"] for row in rows]
    cash = [float(row["cash"]) for row in rows]
    assert all(isinstance(frame, pd.DataFrame) for frame in frames)
    total = _align(frames, cash)
    return total, sum(cash), _path_summary(total, sum(cash), float("nan"))


def _free_portfolio(
    order: tuple[str, ...],
    free_momentum: tuple[pd.DataFrame, np.ndarray, int, np.ndarray],
    free_minutes: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], dict[date, int], int]],
) -> tuple[pd.DataFrame, float]:
    frames = []
    cash = []
    for name in order:
        if name == MOMENTUM:
            frames.append(free_momentum[0])
            cash.append(cp.ACCOUNT_CASH)
        else:
            frames.append(free_minutes[name][0])
            cash.append(float(_window(name).cash))
    return _align(frames, cash), sum(cash)


def _days(count: int) -> str:
    number = abs(int(count))
    if number % 10 == 1 and number % 100 != 11:
        word = "день"
    elif number % 10 in (2, 3, 4) and number % 100 not in (12, 13, 14):
        word = "дня"
    else:
        word = "дней"
    return f"{cp._lots(number)} {word}"


def _pick(rows: list[dict[str, object]], place: str, name: str) -> dict[str, object]:
    return next(row for row in rows if row["place"] == place and row["name"] == name)


def _versus(row: dict[str, object]) -> str:
    summary = row["summary"]
    base = row["base"]
    assert isinstance(summary, dict) and isinstance(base, dict)
    return (
        f"{cp._rub(float(summary['end']))} ₽ против {cp._rub(float(base['end']))} ₽, "
        f"просадка {cp._comma(float(summary['max_dd_pct_peak']), 1)}% пика против {cp._comma(float(base['max_dd_pct_peak']), 1)}%, "
        f"Шарп {cp._comma(float(summary['sharpe']), 2, signed=True)} против {cp._comma(float(base['sharpe']), 2, signed=True)}"
    )


def _momentum_mechanism(bundle: dict[str, object], title: str, ready: dict[str, dict[str, object]]) -> str:
    """Что делали дни, которые запрет снял с моментума. Знак — желаемая позиция до запрета."""
    if MOMENTUM not in ready:
        return ""
    mask = ready[MOMENTUM]["mask"]
    if not isinstance(mask, np.ndarray) or not mask.any():
        return "Моментум в этой цепочке не пропустил ни одного входа."
    percent = bundle["percent"]
    position = bundle["base_position"]
    assert isinstance(percent, np.ndarray) and isinstance(position, np.ndarray)
    signed = position * percent
    blocked = mask & (position != 0)
    kept = ~mask & (position != 0)
    blocked_mean = float(signed[blocked].mean() * 100.0) if blocked.any() else float("nan")
    kept_mean = float(signed[kept].mean() * 100.0) if kept.any() else float("nan")
    days = int(mask.sum())
    return (
        f"В цепочке «{title}» запрет снимает с моментума {_days(days)}. "
        f"Эти дни в среднем дают {cp._comma(blocked_mean, 2, signed=True)}% со знаком желаемой позиции, "
        f"остальные дни в позиции — {cp._comma(kept_mean, 2, signed=True)}%."
    )


def senior_section(bundle: dict[str, object] | None = None, series: cm.Series | None = None) -> list[str]:
    """Текст раздела про цепочку старшего окна."""
    lines = [
        "## 17. Моментум в цепочке старшего окна\n",
        (
            "Между пятью окнами здесь проверяется одно правило: новая позиция не открывается против старшего окна. "
            "Старший — следующее более длинное окно: 30 минут, затем 45, 60, короткое 525 и длинное 12 420. "
            "Если старший без позиции, младший открывается в любую сторону. Уже открытую позицию правило не закрывает и не разворачивает. "
            "Это не фильтр знака из раздела 16: тот обнулял цель всю минуту, в том числе когда моментум был без позиции. "
            "В коде пяти окон такого запрета нет, в живую функцию он не переносится. "
            "Счета отдельные, позиции не сальдируются. "
            "Моментум — тот же счёт 100 000 ₽, плечо 3 и фильтр «"
            + cp.CHOSEN_FILTER
            + "». Места вставки заданы заранее и друг с другом не складываются: "
            "между коротким и длинным, между 60 минутами и коротким, старше длинного, младше 30 минут. "
            "Плюс цепочка одних пяти окон, без моментума. "
            "Знак минутного старшего на решении — его позиция после исполнения этой же минуты, новая цель старшего ещё не позиция. "
            "Знак моментума на день известен до первой минуты. "
            "Когда старший минутный, а младший — моментум, берётся позиция старшего на закрытии предыдущего дня."
        ),
    ]
    if not backtest.BARS_DIR.exists():
        lines.append("Минутных файлов нет, цепочку посчитать нельзя.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    bundle = build_senior(series) if bundle is None else bundle
    free_momentum = bundle["free_momentum"]
    free_minutes = bundle["free_minutes"]
    assert isinstance(free_momentum, tuple) and isinstance(free_minutes, dict)
    all_rows: list[dict[str, object]] = []
    asked_rows: list[dict[str, object]] = []
    asked_ready: dict[str, dict[str, object]] | None = None
    portfolio_bits = []
    for title, order, ready in bundle["placements"]:
        assert isinstance(order, tuple) and isinstance(ready, dict)
        rows = _rows_for(title, order, ready, free_momentum, free_minutes)
        all_rows.extend(rows)
        if title == ASKED:
            asked_rows = rows
            asked_ready = ready
        _frame, cash, summary = _portfolio(rows)
        free_total, free_cash = _free_portfolio(order, free_momentum, free_minutes)
        free_summary = _path_summary(free_total, free_cash, float("nan"))
        portfolio_bits.append((title, summary, free_summary, cash, _cagr_helps(summary, free_summary)))
    lines.append(
        "Свой счёт улучшается, если финиш выше без более глубокой просадки, либо просадка меньше хотя бы на 2 пункта при финише не ниже 98% того же счёта без запрета."
    )
    five_short = _pick(all_rows, "пять окон", "short")
    five_sixty = _pick(all_rows, "пять окон", "sixty")
    five_fortyfive = _pick(all_rows, "пять окон", "fortyfive")
    five_thirty = _pick(all_rows, "пять окон", "thirty")
    lines.append(
        "В цепочке пяти окон без моментума свои счета улучшают «45 минут» и «60 минут». "
        f"«60 минут» при старшем коротком: {_versus(five_sixty)}. "
        f"«45 минут» при старшем окне 60 минут: {_versus(five_fortyfive)}. "
        f"Короткое при старшем длинном даёт {_versus(five_short)}. Финиш выше, просадка глубже, поэтому улучшением этот счёт не считается. "
        f"«30 минут» тоже поднимает финиш, просадка {cp._comma(float(five_thirty['summary']['max_dd_pct_peak']), 1)}% против {cp._comma(float(five_thirty['base']['max_dd_pct_peak']), 1)}%. "
        "Длинное окно старшего не имеет."
    )
    if asked_ready is not None:
        lines.append(_momentum_mechanism(bundle, ASKED, asked_ready))
        short_row = next(row for row in asked_rows if row["name"] == "short")
        momentum_row = next(row for row in asked_rows if row["name"] == MOMENTUM)
        long_row = next(row for row in asked_rows if row["name"] == "long")
        lines.append(
            "Между коротким и длинным короткое окно больше не смотрит на длинное: его старший — моментум. "
            f"Короткое заканчивает на {cp._rub(float(short_row['summary']['end']))} ₽ "
            f"против {cp._rub(float(short_row['base']['end']))} ₽ без запрета, "
            f"просадка {cp._comma(float(short_row['summary']['max_dd_pct_peak']), 1)}% пика "
            f"против {cp._comma(float(short_row['base']['max_dd_pct_peak']), 1)}%, "
            f"Шарп {cp._comma(float(short_row['summary']['sharpe']), 2, signed=True)} "
            f"против {cp._comma(float(short_row['base']['sharpe']), 2, signed=True)}. "
            f"Запрещённых входов {cp._lots(float(short_row['blocked']))}. "
            f"Моментум при старшем длинном окне заканчивает на {cp._rub(float(momentum_row['summary']['end']))} ₽ "
            f"против {cp._rub(float(momentum_row['base']['end']))} ₽, "
            f"просадка {cp._comma(float(momentum_row['summary']['max_dd_pct_peak']), 1)}% пика "
            f"против {cp._comma(float(momentum_row['base']['max_dd_pct_peak']), 1)}%, "
            f"Шарп {cp._comma(float(momentum_row['summary']['sharpe']), 2, signed=True)} "
            f"против {cp._comma(float(momentum_row['base']['sharpe']), 2, signed=True)}. "
            f"Длинное окно в этой цепочке старшего не имеет и остаётся на {cp._rub(float(long_row['summary']['end']))} ₽."
        )
        asked_sixty = _pick(all_rows, ASKED, "sixty")
        asked_fortyfive = _pick(all_rows, ASKED, "fortyfive")
        lines.append(
            "У «45 минут» и «60 минут» в этой вставке старший по-прежнему минутный. "
            "Отметка «да» у них сравнивает счёт с вариантом вообще без запрета, а не с цепочкой пяти окон. "
            f"Относительно цепочки пяти окон оба финиша ниже: «60 минут» {cp._rub(float(asked_sixty['summary']['end']))} ₽ против {cp._rub(float(five_sixty['summary']['end']))} ₽, "
            f"«45 минут» {cp._rub(float(asked_fortyfive['summary']['end']))} ₽ против {cp._rub(float(five_fortyfive['summary']['end']))} ₽."
        )
        above = _pick(all_rows, "старше длинного", "long")
        between_fast = _pick(all_rows, "между 60 и коротким", "sixty")
        below = _pick(all_rows, "младше 30 минут", MOMENTUM)
        lines.append(
            f"Моментум старше длинного снижает длинное окно: {_versus(above)}. "
            f"Моментум между 60 минутами и коротким снижает «60 минут»: {_versus(between_fast)}. "
            f"Моментум младше 30 минут снижает свой счёт: {_versus(below)}. "
            "Эти три места свой счёт по этому правилу не улучшают."
        )
    for title, summary, free_summary, cash, helps in portfolio_bits:
        verdict = "улучшает" if helps else "не улучшает"
        lines.append(
            f"Сумма отдельных счетов «{title}» {verdict} сумму тех же счетов без запрета по годовому проценту: "
            f"{cp._comma(float(summary['cagr']), 1)}% в год и просадка {cp._comma(float(summary['max_dd_pct_peak']), 1)}% пика "
            f"против {cp._comma(float(free_summary['cagr']), 1)}% и {cp._comma(float(free_summary['max_dd_pct_peak']), 1)}%. "
            f"Старт {cp._rub(cash)} ₽, финиш {cp._rub(float(summary['end']))} ₽ против {cp._rub(float(free_summary['end']))} ₽, "
            f"Шарп {cp._comma(float(summary['sharpe']), 2, signed=True)} против {cp._comma(float(free_summary['sharpe']), 2, signed=True)}."
        )
    lines.extend(
        [
            "",
            "«Улучшает свой счёт» сравнивает финиш и просадку с тем же счётом без запрета. «Без изменений» — старший ни одного входа не снял.\n",
            report.md(_table(all_rows), index=False),
        ]
    )
    if asked_rows:
        year_books = {}
        year_cash = {}
        for row, key in (
            (next(row for row in asked_rows if row["name"] == MOMENTUM), "моментум между коротким и длинным"),
            (next(row for row in asked_rows if row["name"] == "short"), "короткое, старший моментум"),
            (next(row for row in asked_rows if row["name"] == "long"), "длинное"),
        ):
            frame = row["frame"]
            assert isinstance(frame, pd.DataFrame)
            stamped = frame.copy()
            stamped["day"] = pd.to_datetime(stamped["day"])
            year_books[key] = stamped
            year_cash[key] = float(row["cash"])
        base_momentum = free_momentum[0].copy()
        base_momentum["day"] = pd.to_datetime(base_momentum["day"])
        base_short = free_minutes["short"][0].copy()
        base_short["day"] = pd.to_datetime(base_short["day"])
        year_books = {
            "моментум без запрета": base_momentum,
            "моментум между коротким и длинным": year_books["моментум между коротким и длинным"],
            "короткое без запрета": base_short,
            "короткое, старший моментум": year_books["короткое, старший моментум"],
        }
        year_cash = {
            "моментум без запрета": cp.ACCOUNT_CASH,
            "моментум между коротким и длинным": cp.ACCOUNT_CASH,
            "короткое без запрета": float(_window("short").cash),
            "короткое, старший моментум": float(_window("short").cash),
        }
        changes = {name: cp._year_changes(frame, year_cash[name]) for name, frame in year_books.items()}
        years = sorted({year for column in changes.values() for year in column})
        year_rows = []
        for year in years:
            item: dict[str, object] = {"год": year}
            for name in year_books:
                item[name] = cp._rub(changes[name].get(year, float("nan")), signed=True)
            year_rows.append(item)
        lines.extend(
            [
                "",
                "Net по календарным годам для места между коротким и длинным. Длинное окно в этой вставке не меняется.\n",
                report.md(pd.DataFrame(year_rows), index=False),
            ]
        )
    lines.append("")
    lines.append("В живую функцию цепочка не переносится. Фильтр моментума не менялся.")
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _month_end(frame: pd.DataFrame) -> pd.Series:
    indexed = frame.drop_duplicates("day").set_index("day")
    indexed.index = pd.to_datetime(indexed.index)
    return indexed["equity"].groupby(indexed.index.to_period("M")).last()


def _json_ready(bundle: dict[str, object]) -> dict[str, object]:
    free_momentum = bundle["free_momentum"]
    free_minutes = bundle["free_minutes"]
    assert isinstance(free_momentum, tuple) and isinstance(free_minutes, dict)
    series = bundle["series"]
    assert isinstance(series, cm.Series)
    months = [str(period) for period in pd.to_datetime(series.days).to_period("M").unique()]
    equity: dict[str, list[float | None]] = {}

    def put(name: str, frame: pd.DataFrame) -> None:
        month_end = _month_end(frame)
        equity[name] = [None if period not in month_end.index else round(float(month_end[period]), 2) for period in months]

    put("моментум без запрета", free_momentum[0])
    put("короткое без запрета", free_minutes["short"][0])
    put("длинное без запрета", free_minutes["long"][0])
    for title, _order, ready in bundle["placements"]:
        assert isinstance(ready, dict)
        if MOMENTUM in ready:
            frame = ready[MOMENTUM]["frame"]
            assert isinstance(frame, pd.DataFrame)
            put(f"моментум: {title}", frame)
        if "short" in ready:
            frame = ready["short"]["frame"]
            assert isinstance(frame, pd.DataFrame)
            put(f"короткое: {title}", frame)
    rows = []
    for title, order, ready in bundle["placements"]:
        assert isinstance(order, tuple) and isinstance(ready, dict)
        rows.extend(_rows_for(title, order, ready, free_momentum, free_minutes))
    table = []
    for row in rows:
        summary = row["summary"]
        base = row["base"]
        assert isinstance(summary, dict) and isinstance(base, dict)
        table.append(
            {
                "place": row["place"],
                "account": row["label"],
                "senior": row["senior"],
                "end": round(float(summary["end"]), 2),
                "base_end": round(float(base["end"]), 2),
                "cagr": round(float(summary["cagr"]), 4),
                "dd": round(float(summary["max_dd_pct_peak"]), 4),
                "base_dd": round(float(base["max_dd_pct_peak"]), 4),
                "sharpe": round(float(summary["sharpe"]), 4),
                "blocked": int(row["blocked"]),
                "helps": row["helps"],
            }
        )
    return {"months": months, "equity": equity, "rows": table}


def append_report(text: str) -> None:
    """Раздел 17 дописывается в отчёт. Повторный запуск заменяет уже записанный раздел."""
    path = report.OUT / "REPORT_cr_momentum_params.md"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## 17."
    if marker in current:
        current = current[: current.index(marker)].rstrip()
    path.write_text(current.rstrip() + "\n" + text.strip() + "\n", encoding="utf-8")


def main() -> str:
    core.configure(1000.0, 0.001)
    if not backtest.BARS_DIR.exists():
        text = "\n".join(senior_section({})).strip() + "\n"
        print(text)
        return text
    series = cm.continuous()
    bundle = build_senior(series)
    text = "\n".join(senior_section(bundle, series)).strip() + "\n"
    out = report.OUT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cr_momentum_senior.md").write_text(text, encoding="utf-8")
    payload = _json_ready(bundle)
    (out / "cr_momentum_senior.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(payload["rows"]).to_csv(out / "cr_momentum_senior.csv", index=False)
    append_report(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
