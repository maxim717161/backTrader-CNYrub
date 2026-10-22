"""Короткое окно смотрит на моментум тем же запретом, что и на длинное.

Правило пяти окон — «старшее, 2 дня, только плюс» из основной ветки (`consensus.ADOPTED_RULE`).
Моментум в этот обмен не входит: окна его не глушат, и он не глушит окна. Сверх правила
короткое окно перед открытием смотрит на позицию моментума так же, как на длинное:
два торговых дня в плюсе, и свежий вход в обратную сторону не ставится.
Для 30, 45 и 60 минут то же взгляд проверяется отдельно, поверх взгляда короткого окна.
Счета отдельные. В живую функцию ничего не переносится.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import backtest
from consensus import (
    ADOPTED_RULE,
    NAMES,
    RULES,
    OpenState,
    _arrays,
    _day_index,
    build_views,
    confirms,
    open_pnl,
    position_state,
    still_in,
    suppress_entry,
    veto_side,
)
from cnyrub.engine import _FillBook, marked_equity, step_minute
from research.cr_momentum_mix import LABELS, _in_market, _path_summary, _window

import research.core as core
import research.cr_momentum as cm
import research.cr_momentum_params as cp
import research.report as report

BASE_RULE = next(rule for rule in RULES if rule.id == "base")
FIVE = "пять окон"
SHORT_LOOK = "короткое смотрит"
EXPERIMENTS = (
    ("30 минут смотрит", frozenset({"short", "thirty"})),
    ("45 минут смотрит", frozenset({"short", "fortyfive"})),
    ("60 минут смотрит", frozenset({"short", "sixty"})),
)


def momentum_watch(days: np.ndarray, nets: np.ndarray, contracts: np.ndarray) -> dict[date, OpenState]:
    """Позиция моментума, известная до открытия дня: возраст эпизода и плюс к вчерашнему закрытию.

    День входа имеет возраст 0 и нулевой плюс. Запрет «2 дня, только плюс» включается
    не раньше третьего дня эпизода и только если результат прошлых дней этого знака больше нуля.
    """
    out: dict[date, OpenState] = {}
    entry: int | None = None
    entry_sign = 0
    for index, day in enumerate(days):
        sign = int(np.sign(contracts[index]))
        if sign == 0:
            entry = None
            entry_sign = 0
            continue
        if entry is None or sign != entry_sign:
            entry = index
            entry_sign = sign
        held_days = index - entry
        pnl = float(nets[entry:index].sum()) if index > entry else 0.0
        out[pd.Timestamp(day).date()] = OpenState("momentum", 0, sign, held_days, pnl, None, None, 0.0)
    return out


def combine_veto(window_veto: int, momentum_veto: int) -> int:
    """1 — не покупать, -1 — не продавать, 2 — ни туда ни туда."""
    if window_veto == 0:
        return momentum_veto
    if momentum_veto == 0 or momentum_veto == window_veto:
        return window_veto
    return 2


def momentum_veto(state: OpenState | None) -> int:
    """Тот же порог, что у длинного окна: два торговых дня и открытый плюс."""
    if state is None or not confirms(state, ADOPTED_RULE):
        return 0
    if state.side > 0:
        return -1
    if state.side < 0:
        return 1
    return 0


def _hits(veto: int, side: int) -> bool:
    return veto == 2 or veto == side


def _run(
    frames: dict[str, pd.DataFrame],
    starts: dict[str, date],
    views: dict[str, dict[str, pd.DataFrame]],
    rule,
    listeners: frozenset[str],
    watch: dict[date, OpenState],
) -> dict[str, object]:
    """Пять счетов. listeners дополнительно глушат свежий вход по позиции моментума."""
    names = NAMES
    windows = backtest.WINDOWS
    channels = {name: window.channel for name, window in zip(names, windows, strict=True)}
    equity = {name: float(window.cash) for name, window in zip(names, windows, strict=True)}
    rows: dict[str, list[dict[str, float | date | int]]] = {name: [] for name in names}
    blocked = dict.fromkeys(names, 0)
    from_momentum = dict.fromkeys(names, 0)
    for secid, frame in frames.items():
        print(f"{rule.id} {secid}", flush=True)
        prepared = views[secid]
        sample = prepared[names[0]]
        days = sample.index.date
        day_index = _day_index(days)
        last_day = days[-1]
        packs = {name: _arrays(prepared[name]) for name in names}
        books = {name: _FillBook(secid, equity[name]) for name in names}
        prev_pnl: dict[str, float | None] = dict.fromkeys(names)
        prev2_pnl: dict[str, float | None] = dict.fromkeys(names)
        streak = dict.fromkeys(names, 0)
        front = starts.get(secid)
        count = len(sample)
        current: date | None = None
        start_equity = dict(equity)
        last_mark = dict(equity)
        minutes = dict.fromkeys(names, 0)
        held_minutes = dict.fromkeys(names, 0)

        def flush() -> None:
            if current is None:
                return
            for name in names:
                end = float(last_mark[name])
                rows[name].append(
                    {
                        "day": current,
                        "net": end - float(start_equity[name]),
                        "equity": end,
                        "minutes": int(minutes[name]),
                        "held_minutes": int(held_minutes[name]),
                    }
                )

        for i in range(count):
            trading = front is None or days[i] >= front
            if trading and days[i] != current:
                flush()
                current = days[i]
                start_equity = dict(last_mark)
                minutes = dict.fromkeys(names, 0)
                held_minutes = dict.fromkeys(names, 0)
            close = float(packs[names[0]]["close"][i])
            for name, window in zip(names, windows, strict=True):
                pack = packs[name]
                step_minute(
                    books[name],
                    i,
                    opened=float(pack["open"][i]),
                    high=float(pack["high"][i]),
                    low=float(pack["low"][i]),
                    close=float(pack["close"][i]),
                    volume=float(pack["volume"][i]),
                    day=days[i],
                    next_day=None if i + 1 >= count else days[i + 1],
                    last_day=last_day,
                    prior_high=float(pack["prior_high"][i]),
                    prior_low=float(pack["prior_low"][i]),
                    prior_vol=float(pack["prior_vol"][i]),
                    prior_range=float(pack["prior_range"][i]),
                    exit_high=float(pack["exit_high"][i]),
                    exit_low=float(pack["exit_low"][i]),
                    clock_vol=float(pack["clock_vol"][i]),
                    entry_ready=float(pack["entry_ready"][i]),
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
                    trade_from=front,
                    trail=False,
                    scale_step=window.scale_step,
                    scale_floor=window.scale_floor,
                    scale_back=window.scale_back,
                    drift=float(pack["drift"][i]),
                    path=float(pack["path"][i]),
                    surge_vol=float(pack["surge_vol"][i]),
                    eff_low=window.eff_low,
                    eff_high=window.eff_high,
                    surge_cap=window.surge_cap,
                    leverage=window.leverage,
                )
            states = []
            for name, window in zip(names, windows, strict=True):
                state = position_state(
                    name,
                    window.channel,
                    books[name],
                    i,
                    day_index,
                    close,
                    prev_pnl[name],
                    prev2_pnl[name],
                )
                if state is not None:
                    states.append(state)
            mom = momentum_veto(watch.get(days[i]))
            for name in names:
                book = books[name]
                window_veto = veto_side(states, rule, name, channels[name])
                extra = mom if name in listeners else 0
                veto = combine_veto(window_veto, extra)
                fresh_side = 1 if book.target > 0 else -1
                if suppress_entry(book, veto):
                    if streak[name] != fresh_side:
                        blocked[name] += 1
                        streak[name] = fresh_side
                        if _hits(extra, fresh_side) and not _hits(window_veto, fresh_side):
                            from_momentum[name] += 1
                else:
                    streak[name] = 0
                last_mark[name] = marked_equity(book, close)
                if trading:
                    minutes[name] += 1
                    if book.held:
                        held_minutes[name] += 1
            if i + 1 == count or days[i + 1] != days[i]:
                for name in names:
                    book = books[name]
                    prev2_pnl[name] = prev_pnl[name]
                    prev_pnl[name] = open_pnl(book, close) if still_in(book) else None
        flush()
        for name in names:
            equity[name] = float(last_mark[name])
    frames_out = {}
    for name in names:
        frame = pd.DataFrame(rows[name])
        cash = float(_window(name).cash)
        gap = abs(float(frame["net"].sum()) - (float(frame["equity"].iloc[-1]) - cash)) if len(frame) else 0.0
        if gap > 1.0:
            raise AssertionError(f"{name}: дневной результат не сходится со счётом на {gap:.2f}")
        frames_out[name] = frame
    return {"frames": frames_out, "blocked": blocked, "from_momentum": from_momentum}


def _summary(name: str, frame: pd.DataFrame) -> dict[str, float]:
    return _path_summary(frame, float(_window(name).cash), _in_market(frame))


def _watch_counts(watch: dict[date, OpenState]) -> tuple[int, int, int]:
    ready = [state for state in watch.values() if confirms(state, ADOPTED_RULE)]
    longs = sum(state.side > 0 for state in ready)
    shorts = sum(state.side < 0 for state in ready)
    return len(ready), longs, shorts


def build_veto(series: cm.Series | None = None) -> dict[str, object]:
    """Пять прогонов: правило основной ветки, взгляд короткого окна и три опыта."""
    core.configure(1000.0, 0.001)
    series = cm.continuous() if series is None else series
    volume = cp.contract_daily_volume(series)
    day_range = cp.contract_daily_range(series)
    position = cp.chosen_account_position(series, volume, day_range)
    curve = cp.leverage_curve(
        pd.DataFrame({"day": pd.to_datetime(series.days), "position": position}),
        series.level,
        series.ret,
        series.roll,
        cp.CHOSEN_LEVERAGE,
        cp.ACCOUNT_CASH,
    )
    watch = momentum_watch(curve["day"].to_numpy(), curve["net"].to_numpy(dtype=float), curve["contracts"].to_numpy(dtype=float))
    frames = backtest.load_minutes()
    starts = backtest.load_trade_starts()
    views = build_views(frames)
    plan = (
        ("без запрета", BASE_RULE, frozenset()),
        (FIVE, ADOPTED_RULE, frozenset()),
        (SHORT_LOOK, ADOPTED_RULE, frozenset({"short"})),
        *tuple((title, ADOPTED_RULE, listeners) for title, listeners in EXPERIMENTS),
    )
    runs = {}
    for title, rule, listeners in plan:
        print(f"прогон {title}", flush=True)
        runs[title] = _run(frames, starts, views, rule, listeners, watch)
    return {"series": series, "curve": curve, "watch": watch, "runs": runs}


def _line(title: str, name: str, run: dict[str, object], base: dict[str, float] | None) -> dict[str, object]:
    frame = run["frames"][name]
    assert isinstance(frame, pd.DataFrame)
    summary = _summary(name, frame)
    blocked = run["blocked"][name]
    extra = run["from_momentum"][name]
    same = base is not None and abs(summary["end"] - base["end"]) < 0.5 and int(extra) == 0
    if base is None or same:
        helps = "без изменений" if same else "—"
    elif cp._volume_helps(pd.Series(summary), pd.Series(base)):
        helps = "да"
    else:
        helps = "нет"
    return {
        "title": title,
        "name": name,
        "label": LABELS[name],
        "summary": summary,
        "base": base,
        "blocked": int(blocked),
        "from_momentum": int(extra),
        "helps": helps,
        "frame": frame,
    }


def _table(rows: list[dict[str, object]]) -> pd.DataFrame:
    out = []
    for row in rows:
        summary = row["summary"]
        assert isinstance(summary, dict)
        out.append(
            {
                "прогон": row["title"],
                "счёт": row["label"],
                "финиш, ₽": cp._rub(float(summary["end"])),
                "год, %": cp._comma(float(summary["cagr"]), 1),
                "просадка, % пика": cp._comma(float(summary["max_dd_pct_peak"]), 1),
                "Шарп": cp._comma(float(summary["sharpe"]), 2, signed=True),
                "запретов": cp._lots(float(row["blocked"])),
                "из них моментум": cp._lots(float(row["from_momentum"])),
                "улучшает счёт": row["helps"],
            }
        )
    return pd.DataFrame(out)


def _versus(summary: dict[str, float], base: dict[str, float]) -> str:
    return (
        f"{cp._rub(summary['end'])} ₽ против {cp._rub(base['end'])} ₽, "
        f"просадка {cp._comma(summary['max_dd_pct_peak'], 1)}% пика против {cp._comma(base['max_dd_pct_peak'], 1)}%, "
        f"Шарп {cp._comma(summary['sharpe'], 2, signed=True)} против {cp._comma(base['sharpe'], 2, signed=True)}"
    )


def veto_section(bundle: dict[str, object] | None = None, series: cm.Series | None = None) -> list[str]:
    """Текст раздела: правило основной ветки и взгляд на моментум."""
    lines = [
        "## 18. Короткое окно смотрит на моментум\n",
        (
            "Правило взаимодействия пяти окон взято из основной ветки: «старшее, 2 дня, только плюс». "
            "Запрет ставит только окно с более длинным каналом. Позиция должна жить уже два торговых дня, "
            "и открытый результат по ней больше нуля. Тогда свежий вход в обратную сторону не открывается. "
            "Свой стоп, выход и уже открытая позиция не трогаются. Короткое окно не глушит длинное. "
            "30 минут не глушит никого: короче него окон нет. "
            "Моментум в этот обмен не входит. Ни одно окно не глушит моментум, и моментум не глушит окна. "
            f"Его счёт остаётся прежним: 100 000 ₽, плечо {cp.CHOSEN_LEVERAGE}, фильтр «{cp.CHOSEN_FILTER}». "
            "Сверх этого правила короткое окно перед открытием смотрит на позицию моментума так же, как на длинное окно: "
            "два торговых дня текущего знака и плюс к закрытию вчерашнего дня. День входа и следующий день запрета ещё не дают. "
            "Для 30, 45 и 60 минут тот же взгляд проверяется отдельно. Каждый опыт добавляется к взгляду короткого окна и не складывается с другими. "
            "Счета отдельные."
        ),
    ]
    if not backtest.BARS_DIR.exists():
        lines.append("Минутных файлов нет, правило посчитать нельзя.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    bundle = build_veto(series) if bundle is None else bundle
    runs = bundle["runs"]
    watch = bundle["watch"]
    curve = bundle["curve"]
    assert isinstance(runs, dict) and isinstance(watch, dict) and isinstance(curve, pd.DataFrame)
    ready, longs, shorts = _watch_counts(watch)
    momentum_end = float(curve["equity"].iloc[-1])
    lines.append(
        f"Моментум заканчивает на {cp._rub(momentum_end)} ₽. "
        f"Дней, когда его позиция уже два дня в плюсе и может запретить вход: {cp._lots(ready)}, "
        f"из них лонг {cp._lots(longs)}, шорт {cp._lots(shorts)}."
    )
    five = runs[FIVE]
    plain = runs["без запрета"]
    looked = runs[SHORT_LOOK]
    rows: list[dict[str, object]] = []
    for name in NAMES:
        base_summary = _summary(name, plain["frames"][name])
        rows.append(_line(FIVE, name, five, base_summary))
    short_line = _line("short-only", "short", looked, _summary("short", five["frames"]["short"]))
    lines.append(
        "Сравнение «улучшает счёт» для правила пяти окон — с теми же счетами без запрета. "
        "Для взгляда на моментум — со счетами, где правило пяти окон уже работает, а моментум ещё молчит."
    )
    helped = [str(row["label"]) for row in rows if row["helps"] == "да"]
    if helped:
        lines.append("Правило пяти окон улучшает свои счета: " + ", ".join(f"«{name}»" for name in helped) + ".")
    else:
        lines.append("Правило пяти окон ни один свой счёт по этому порогу не улучшает.")
    lines.append(
        f"Короткое окно со взглядом на моментум: {_versus(short_line['summary'], short_line['base'])}. "
        f"Запретов {cp._lots(float(short_line['blocked']))}, из них моментум добавил {cp._lots(float(short_line['from_momentum']))} "
        "сверх тех, что и так ставило более длинное окно. "
        + (
            "Свой счёт это улучшает."
            if short_line["helps"] == "да"
            else "Свой счёт это не улучшает."
            if short_line["helps"] == "нет"
            else "Счёт короткого окна от взгляда на моментум не меняется."
        )
    )
    spilled = []
    for name in ("thirty", "fortyfive", "sixty"):
        after = _summary(name, looked["frames"][name])
        before = _summary(name, five["frames"][name])
        spilled.append(f"«{LABELS[name]}» {cp._rub(after['end'])} ₽ против {cp._rub(before['end'])} ₽")
    lines.append(
        "Позиция короткого окна при этом меняется, и через запрет старшего окна снижаются более короткие счета: "
        + ", ".join(spilled)
        + ". Моментум в этом прогоне им запретов не ставил."
    )
    experiment_rows = []
    useful = []
    idle = []
    for title, _listeners in EXPERIMENTS:
        run = runs[title]
        focus = "thirty" if title.startswith("30") else "fortyfive" if title.startswith("45") else "sixty"
        base = _summary(focus, looked["frames"][focus])
        row = _line(title, focus, run, base)
        experiment_rows.append(row)
        others = []
        for name in NAMES:
            if name == focus:
                continue
            changed = _summary(name, run["frames"][name])
            previous = _summary(name, looked["frames"][name])
            if abs(changed["end"] - previous["end"]) >= 0.5:
                others.append(LABELS[name])
        if row["from_momentum"] == 0:
            idle.append(LABELS[focus])
        elif row["helps"] == "да":
            useful.append(LABELS[focus])
        tail = ""
        if others:
            tail = " Вместе с ним сдвинулись счета " + ", ".join(f"«{name}»" for name in others) + ": запрет старшего окна зависит от его позиции."
        lines.append(
            f"«{LABELS[focus]}» со взглядом на моментум: {_versus(row['summary'], row['base'])}. "
            f"Моментум добавил {cp._lots(float(row['from_momentum']))} запретов. "
            + (
                "Свой счёт это улучшает."
                if row["helps"] == "да"
                else "Свой счёт это не улучшает."
                if row["helps"] == "нет"
                else "Счёт не меняется."
            )
            + tail
        )
    if useful:
        lines.append("Смотреть на моментум стоит окнам " + ", ".join(f"«{name}»" for name in useful) + ".")
    else:
        lines.append("Ни 30, ни 45, ни 60 минут смотреть на моментум по этому правилу не стоит: свой счёт ни у одного не улучшается.")
    if idle:
        lines.append("Моментум не добавил ни одного запрета окнам " + ", ".join(f"«{name}»" for name in idle) + ".")
    show = []
    for name in NAMES:
        show.append(_line(FIVE, name, five, _summary(name, plain["frames"][name])))
        show.append(_line(SHORT_LOOK, name, looked, _summary(name, five["frames"][name])))
    for title, _listeners in EXPERIMENTS:
        focus = "thirty" if title.startswith("30") else "fortyfive" if title.startswith("45") else "sixty"
        show.append(_line(title, focus, runs[title], _summary(focus, looked["frames"][focus])))
    lines.extend(
        [
            "",
            "«Из них моментум» — запреты, которых не было у более длинного окна. «Улучшает счёт» для строки «пять окон» сравнивает с прогоном без запрета, для остальных — со строкой, где моментум ещё не смотрит это окно.\n",
            report.md(_table(show), index=False),
        ]
    )
    year_names = ("без запрета", FIVE, SHORT_LOOK)
    year_books = {}
    for title in year_names:
        frame = runs[title]["frames"]["short"].copy()
        frame["day"] = pd.to_datetime(frame["day"])
        year_books[title] = frame
    changes = {title: cp._year_changes(frame, float(_window("short").cash)) for title, frame in year_books.items()}
    year_rows = []
    for year in sorted({year for column in changes.values() for year in column}):
        item: dict[str, object] = {"год": year}
        for title in year_names:
            item[title] = cp._rub(changes[title].get(year, float("nan")), signed=True)
        year_rows.append(item)
    lines.extend(
        [
            "",
            "Net короткого окна по календарным годам. «Пять окон» — запрет старшего окна без моментума. «Короткое смотрит» — то же плюс позиция моментума.\n",
            report.md(pd.DataFrame(year_rows), index=False),
            "",
            "В живую функцию взгляд короткого окна на моментум не переносится. Фильтр моментума не менялся.",
        ]
    )
    del experiment_rows
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _month_end(frame: pd.DataFrame) -> pd.Series:
    indexed = frame.drop_duplicates("day").set_index("day")
    indexed.index = pd.to_datetime(indexed.index)
    return indexed["equity"].groupby(indexed.index.to_period("M")).last()


def _json_ready(bundle: dict[str, object]) -> dict[str, object]:
    series = bundle["series"]
    runs = bundle["runs"]
    assert isinstance(series, cm.Series) and isinstance(runs, dict)
    months = [str(period) for period in pd.to_datetime(series.days).to_period("M").unique()]
    equity: dict[str, list[float | None]] = {}

    def put(name: str, frame: pd.DataFrame) -> None:
        month_end = _month_end(frame)
        equity[name] = [None if period not in month_end.index else round(float(month_end.loc[period]), 2) for period in months]

    curve = bundle["curve"]
    assert isinstance(curve, pd.DataFrame)
    put("моментум", curve)
    for title in ("без запрета", FIVE, SHORT_LOOK):
        for name in NAMES:
            put(f"{LABELS[name]}: {title}", runs[title]["frames"][name])
    for title, _listeners in EXPERIMENTS:
        focus = "thirty" if title.startswith("30") else "fortyfive" if title.startswith("45") else "sixty"
        put(f"{LABELS[focus]}: {title}", runs[title]["frames"][focus])
    return {"months": months, "equity": equity}


def append_report(text: str) -> None:
    path = report.OUT / "REPORT_cr_momentum_params.md"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## 18."
    if marker in current:
        current = current[: current.index(marker)].rstrip()
    path.write_text(current.rstrip() + "\n" + text.strip() + "\n", encoding="utf-8")


def main() -> str:
    core.configure(1000.0, 0.001)
    if not backtest.BARS_DIR.exists():
        text = "\n".join(veto_section({})).strip() + "\n"
        print(text)
        return text
    series = cm.continuous()
    bundle = build_veto(series)
    text = "\n".join(veto_section(bundle, series)).strip() + "\n"
    out = report.OUT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cr_momentum_veto.md").write_text(text, encoding="utf-8")
    payload = _json_ready(bundle)
    (out / "cr_momentum_veto.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    append_report(text)
    print(text)
    return text


if __name__ == "__main__":
    main()

