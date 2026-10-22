"""Общий счёт шести стратегий: правило пяти окон не меняется, взгляд на моментум — отдельными опытами.

База — пять окон по «старшее, 2 дня, только плюс» и моментум без запрета. Опыты заданы заранее
и не складываются. Свежий вход — только из нуля, уже открытую позицию запрет не закрывает.
Моментум видит окно на закрытии прошлого торгового дня. Окно видит моментум так же, как в разделе 18:
два дня текущего знака и плюс к вчерашнему закрытию. В живую функцию ничего не переносится.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import backtest
from consensus import ADOPTED_RULE, NAMES, ConsensusRule, OpenState, veto_side
from research.cr_momentum_mix import LABELS, _align, _path_summary
from research.cr_momentum_veto import _run, momentum_veto, momentum_watch

import research.core as core
import research.cr_momentum as cm
import research.cr_momentum_params as cp
import research.report as report

BOOK = "как есть"
MOM_LONG = "моментум смотрит на длинное"
MOM_ALL = "моментум смотрит на все окна"
LONG_MOM = "длинное смотрит на моментум"
ALL_MOM = "все окна смотрят на моментум"
MUTUAL = "длинное и моментум друг на друга"
SHORT_BOOK = "короткое смотрит на моментум"
LONG_RULE = ConsensusRule("mom-sees-long", "только длинное окно", 2, "no", source="long")
CASH_EACH = cp.ACCOUNT_CASH


def _book_cash() -> float:
    return float(sum(window.cash for window in backtest.WINDOWS) + CASH_EACH)


class MomentumAccount:
    """Тот же счёт плеча 3, но свежий вход из нуля снимается, если запрет смотрит в его сторону."""

    def __init__(self, n: int, cash: float = CASH_EACH) -> None:
        self.n = n
        self.cash = cash
        self.balance = cash
        self.previous = 0
        self.dead = False
        self.equity = np.zeros(n)
        self.net = np.zeros(n)
        self.contracts = np.zeros(n)
        self.done = 0
        self.streak = 0
        self.blocked = 0

    def step(
        self,
        t: int,
        sign: float,
        level: float,
        ret: float,
        roll: bool,
        veto: int,
        leverage: float = cp.CHOSEN_LEVERAGE,
        margin_rate: float = cp.MARGIN_RATE,
    ) -> None:
        if self.previous == 0 and sign != 0.0:
            side = 1 if sign > 0 else -1
            if veto == 2 or veto == side:
                if self.streak != side:
                    self.blocked += 1
                    self.streak = side
                sign = 0.0
            else:
                self.streak = 0
        else:
            self.streak = 0
        if self.dead or self.balance <= 0:
            self.dead = True
            self.balance = 0.0
            self.equity[t] = self.net[t] = self.contracts[t] = 0.0
            self.previous = 0
            self.done = t + 1
            return
        price_in = level - ret
        notional = price_in * core.MULTIPLIER
        target = cp._target_lots(sign, self.balance, notional, leverage, margin_rate)
        sides = cp._resize_sides(self.previous, target, roll)
        cost = sides * cp.side_cost()
        gross = target * ret * core.MULTIPLIER
        balance_after = self.balance + gross - cost
        required = abs(target) * margin_rate * level * core.MULTIPLIER
        carry = target
        if target != 0 and balance_after < required:
            balance_after -= abs(target) * cp.side_cost()
            cost += abs(target) * cp.side_cost()
            carry = 0
            required = 0.0
        if t == self.n - 1 and carry and balance_after > 0:
            balance_after -= abs(carry) * cp.side_cost()
            cost += abs(carry) * cp.side_cost()
            carry = 0
        if balance_after <= 0:
            self.dead = True
            balance_after = 0.0
            carry = 0
        self.net[t] = balance_after - self.balance
        self.contracts[t] = target
        self.balance = balance_after
        self.previous = carry
        self.equity[t] = self.balance
        self.done = t + 1

    def frame(self, days: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "day": [pd.Timestamp(day).date() for day in days],
                "equity": self.equity,
                "net": self.net,
                "contracts": self.contracts,
            }
        )


def replay_momentum(
    days: np.ndarray,
    sign: np.ndarray,
    level: np.ndarray,
    ret: np.ndarray,
    roll: np.ndarray,
    veto_of,
) -> MomentumAccount:
    """veto_of(day) -> запрет на этот день. День — дата ряда."""
    account = MomentumAccount(len(days))
    for t, day in enumerate(days):
        account.step(t, float(sign[t]), float(level[t]), float(ret[t]), bool(roll[t]), int(veto_of(pd.Timestamp(day).date())))
    return account


def _prior_veto(states: dict[date, list[OpenState]], rule: ConsensusRule, channel: int):
    ordered = sorted(states)
    cursor = 0
    latest: list[OpenState] = []

    def veto_of(day: date) -> int:
        nonlocal cursor, latest
        while cursor < len(ordered) and ordered[cursor] < day:
            latest = states[ordered[cursor]]
            cursor += 1
        if not latest:
            return 0
        return veto_side(latest, rule, "momentum", channel)

    return veto_of


def _portfolio(window_frames: dict[str, pd.DataFrame], momentum: pd.DataFrame) -> pd.DataFrame:
    frames = [window_frames[name] for name in NAMES]
    cash = [float(window.cash) for window in backtest.WINDOWS]
    frames.append(momentum)
    cash.append(CASH_EACH)
    return _align(frames, cash)


def _book_summary(frame: pd.DataFrame) -> dict[str, float]:
    return _path_summary(frame, _book_cash(), float("nan"))


def _mom_summary(frame: pd.DataFrame) -> dict[str, float]:
    held = frame["contracts"].to_numpy() != 0 if "contracts" in frame else np.zeros(len(frame), dtype=bool)
    share = float(held.mean() * 100.0) if len(frame) else float("nan")
    return _path_summary(frame, CASH_EACH, share)


class _LiveMomentum:
    """Моментум идёт вместе с окнами: запрет длинного окна известен до открытия дня."""

    def __init__(self, series: cm.Series, sign: np.ndarray) -> None:
        self.days = [pd.Timestamp(day).date() for day in series.days]
        self.sign = sign
        self.level = np.asarray(series.level, dtype=float)
        self.ret = np.asarray(series.ret, dtype=float)
        self.roll = np.asarray(series.roll, dtype=bool)
        self.account = MomentumAccount(len(self.days))
        self.snapshot: list[OpenState] = []
        self.watch: dict[date, OpenState] = {}
        self.seen: set[date] = set()
        self.index = 0

    def on_day_open(self, day: date) -> None:
        if day in self.seen:
            return
        self.seen.add(day)
        while self.index < len(self.days) and self.days[self.index] <= day:
            veto = veto_side(self.snapshot, LONG_RULE, "momentum", 0)
            t = self.index
            self.account.step(t, float(self.sign[t]), self.level[t], self.ret[t], bool(self.roll[t]), veto)
            self.index += 1
        self.watch.clear()
        self.watch.update(
            momentum_watch(
                np.array(self.days[: self.account.done], dtype="datetime64[D]"),
                self.account.net[: self.account.done],
                self.account.contracts[: self.account.done],
            )
        )

    def on_day_close(self, day: date, states: list[OpenState]) -> None:
        del day
        self.snapshot = states

    def finish(self) -> None:
        while self.index < len(self.days):
            t = self.index
            veto = veto_side(self.snapshot, LONG_RULE, "momentum", 0)
            self.account.step(t, float(self.sign[t]), self.level[t], self.ret[t], bool(self.roll[t]), veto)
            self.index += 1


def build_cross(series: cm.Series | None = None) -> dict[str, object]:
    """Пять прогонов окон и два запрета моментуму по уже записанным позициям окон."""
    core.configure(1000.0, 0.001)
    series = cm.continuous() if series is None else series
    volume = cp.contract_daily_volume(series)
    day_range = cp.contract_daily_range(series)
    position = cp.chosen_account_position(series, volume, day_range)
    sign_frame = pd.DataFrame({"day": pd.to_datetime(series.days), "position": position})
    plain = cp.leverage_curve(sign_frame, series.level, series.ret, series.roll, cp.CHOSEN_LEVERAGE, CASH_EACH)
    watch = momentum_watch(plain["day"].to_numpy(), plain["net"].to_numpy(dtype=float), plain["contracts"].to_numpy(dtype=float))
    frames = backtest.load_minutes()
    starts = backtest.load_trade_starts()
    views = backtest_views(frames)
    states: dict[date, list[OpenState]] = {}

    def keep(_day: date, snap: list[OpenState]) -> None:
        states[_day] = snap

    print("прогон как есть", flush=True)
    base = _run(frames, starts, views, ADOPTED_RULE, frozenset(), watch, on_day_close=keep)
    print("прогон короткое смотрит", flush=True)
    short = _run(frames, starts, views, ADOPTED_RULE, frozenset({"short"}), watch)
    print("прогон длинное смотрит", flush=True)
    long = _run(frames, starts, views, ADOPTED_RULE, frozenset({"long"}), watch)
    print("прогон все окна смотрят", flush=True)
    every = _run(frames, starts, views, ADOPTED_RULE, frozenset(NAMES), watch)
    live = _LiveMomentum(series, position)
    print("прогон длинное и моментум друг на друга", flush=True)
    mutual = _run(
        frames,
        starts,
        views,
        ADOPTED_RULE,
        frozenset({"long"}),
        live.watch,
        on_day_open=live.on_day_open,
        on_day_close=live.on_day_close,
    )
    live.finish()
    days = plain["day"].to_numpy()
    level = np.asarray(series.level, dtype=float)
    ret = np.asarray(series.ret, dtype=float)
    roll = np.asarray(series.roll, dtype=bool)
    mom_long = replay_momentum(days, position, level, ret, roll, _prior_veto(states, LONG_RULE, 0))
    mom_all = replay_momentum(days, position, level, ret, roll, _prior_veto(states, ADOPTED_RULE, 0))
    plain_days = plain.copy()
    plain_days["day"] = pd.to_datetime(plain_days["day"]).dt.date
    books = {
        BOOK: (base, plain_days, 0),
        MOM_LONG: (base, mom_long.frame(days), mom_long.blocked),
        MOM_ALL: (base, mom_all.frame(days), mom_all.blocked),
        LONG_MOM: (long, plain_days, 0),
        ALL_MOM: (every, plain_days, 0),
        MUTUAL: (mutual, live.account.frame(series.days), live.account.blocked),
        SHORT_BOOK: (short, plain_days, 0),
    }
    return {"series": series, "books": books, "plain_end": float(plain["equity"].iloc[-1])}


def backtest_views(frames: dict[str, pd.DataFrame]):
    from consensus import build_views

    return build_views(frames)


def _window_blocks(run: dict[str, object]) -> int:
    blocked = run["from_momentum"]
    return int(sum(blocked.values()))


def cross_section(bundle: dict[str, object] | None = None, series: cm.Series | None = None) -> list[str]:
    """Текст раздела: общий счёт и заранее заданные взгляды."""
    lines = [
        "## 19. Общий счёт и моментум\n",
        (
            "Правило пяти окон остаётся «старшее, 2 дня, только плюс» и между собой не меняется. "
            "Моментум в этот обмен не входит. Ниже заранее заданные добавки, каждая сама по себе. "
            "Общий счёт — сумма шести отдельных счетов по 100 000 ₽, старт 600 000 ₽. "
            "Свежий вход моментума снимается только если вчера контрактов не было: "
            "уже открытую позицию запрет не закрывает. Моментум смотрит на окно по закрытию прошлого торгового дня. "
            "Окно смотрит на моментум так же, как короткое смотрит на длинное: два дня в плюсе к вчерашнему закрытию. "
            "Улучшает общий счёт то же правило, что и отдельный счёт: финиш выше и просадка не глубже чем на 0,05 пункта, "
            "либо финиш не ниже 98% базы и просадка меньше хотя бы на 2 пункта."
        ),
    ]
    if not backtest.BARS_DIR.exists():
        lines.append("Минутных файлов нет, общий счёт посчитать нельзя.")
        body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
        return ["\n" + body]
    bundle = build_cross(series) if bundle is None else bundle
    books = bundle["books"]
    assert isinstance(books, dict)
    base_run, base_mom, _ = books[BOOK]
    base_port = _portfolio(base_run["frames"], base_mom)
    base_summary = _book_summary(base_port)
    base_mom_summary = _mom_summary(base_mom)
    lines.append(
        f"Без добавки общий счёт заканчивает на {cp._rub(base_summary['end'])} ₽, "
        f"{cp._comma(base_summary['cagr'], 1)}% в год, просадка {cp._comma(base_summary['max_dd_pct_peak'], 1)}% пика, "
        f"Шарп {cp._comma(base_summary['sharpe'], 2, signed=True)}. "
        f"Моментум в этой сумме — {cp._rub(base_mom_summary['end'])} ₽."
    )
    rows = []
    helped = []
    order = (BOOK, MOM_LONG, MOM_ALL, LONG_MOM, ALL_MOM, MUTUAL, SHORT_BOOK)
    summaries = {BOOK: base_summary}
    ports = {BOOK: base_port}
    for title in order:
        run, momentum, mom_blocks = books[title]
        port = base_port if title == BOOK else _portfolio(run["frames"], momentum)
        summary = base_summary if title == BOOK else _book_summary(port)
        mom_summary = _mom_summary(momentum)
        summaries[title] = summary
        ports[title] = port
        window_blocks = _window_blocks(run)
        same = abs(summary["end"] - base_summary["end"]) < 0.5 and int(mom_blocks) == 0 and window_blocks == 0
        if title == BOOK:
            helps = "база"
        elif same:
            helps = "без изменений"
        elif cp._volume_helps(pd.Series(summary), pd.Series(base_summary)):
            helps = "да"
            helped.append(title)
        else:
            helps = "нет"
        rows.append(
            {
                "прогон": title,
                "финиш, ₽": cp._rub(summary["end"]),
                "год, %": cp._comma(summary["cagr"], 1),
                "просадка, % пика": cp._comma(summary["max_dd_pct_peak"], 1),
                "Шарп": cp._comma(summary["sharpe"], 2, signed=True),
                "моментум, ₽": cp._rub(mom_summary["end"]),
                "запретов моментуму": cp._lots(float(mom_blocks)),
                "моментум добавил окнам": cp._lots(float(window_blocks)),
                "улучшает общий счёт": helps,
            }
        )
        if title == BOOK:
            continue
        lines.append(
            f"«{title}»: {cp._rub(summary['end'])} ₽ против {cp._rub(base_summary['end'])} ₽, "
            f"просадка {cp._comma(summary['max_dd_pct_peak'], 1)}% против {cp._comma(base_summary['max_dd_pct_peak'], 1)}%, "
            f"Шарп {cp._comma(summary['sharpe'], 2, signed=True)} против {cp._comma(base_summary['sharpe'], 2, signed=True)}. "
            f"Моментум {cp._rub(mom_summary['end'])} ₽, запретов ему {cp._lots(float(mom_blocks))}, "
            f"окнам моментум добавил {cp._lots(float(window_blocks))}. "
            + (
                "Общий счёт это улучшает."
                if helps == "да"
                else "Запрет ни разу не сработал, общий счёт тот же."
                if helps == "без изменений"
                else "Общий счёт это не улучшает."
            )
        )
    if helped:
        lines.append("Общий счёт улучшают: " + ", ".join(f"«{title}»" for title in helped) + ".")
    else:
        lines.append("Ни одна заранее заданная добавка общий счёт не улучшает.")
    lines.extend(["", report.md(pd.DataFrame(rows), index=False), ""])
    focus = [BOOK, *helped] if helped else [BOOK, MOM_LONG, MOM_ALL, LONG_MOM]
    year_rows = []
    changes = {}
    for title in focus:
        frame = ports[title].copy()
        frame["day"] = pd.to_datetime(frame["day"])
        changes[title] = cp._year_changes(frame, _book_cash())
    years = sorted({year for column in changes.values() for year in column})
    for year in years:
        item: dict[str, object] = {"год": year}
        for title in focus:
            item[title] = cp._rub(changes[title].get(year, float("nan")), signed=True)
        year_rows.append(item)
    lines.append("Net общего счёта по календарным годам.\n")
    lines.append(report.md(pd.DataFrame(year_rows), index=False))
    moved = []
    for title in (LONG_MOM, ALL_MOM, MUTUAL, SHORT_BOOK):
        run, _, _ = books[title]
        bits = []
        for name in NAMES:
            end = float(run["frames"][name]["equity"].iloc[-1])
            base_end = float(base_run["frames"][name]["equity"].iloc[-1])
            if abs(end - base_end) >= 0.5:
                bits.append(f"«{LABELS[name]}» {cp._rub(end)} ₽ против {cp._rub(base_end)} ₽")
        if bits:
            moved.append("«" + title + "»: " + ", ".join(bits) + ".")
    if moved:
        lines.append("Окна, у которых сменился финиш. " + " ".join(moved))
    lines.append("В живую функцию эти взгляды не переносятся. Фильтр моментума и правило пяти окон не менялись.")
    del summaries
    body = "\n\n".join(part.strip("\n") for part in lines if part.strip())
    return ["\n" + body]


def _month_end(frame: pd.DataFrame) -> pd.Series:
    indexed = frame.drop_duplicates("day").set_index("day")
    indexed.index = pd.to_datetime(indexed.index)
    return indexed["equity"].groupby(indexed.index.to_period("M")).last()


def _json_ready(bundle: dict[str, object]) -> dict[str, object]:
    series = bundle["series"]
    books = bundle["books"]
    assert isinstance(series, cm.Series) and isinstance(books, dict)
    months = [str(period) for period in pd.to_datetime(series.days).to_period("M").unique()]
    equity: dict[str, list[float | None]] = {}
    for title, (run, momentum, _blocks) in books.items():
        port = _portfolio(run["frames"], momentum)
        month_end = _month_end(port)
        equity[title] = [None if period not in month_end.index else round(float(month_end.loc[period]), 2) for period in months]
    return {"months": months, "equity": equity, "plain_end": bundle["plain_end"]}


def append_report(text: str) -> None:
    path = report.OUT / "REPORT_cr_momentum_params.md"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## 19."
    if marker in current:
        current = current[: current.index(marker)].rstrip()
    path.write_text(current.rstrip() + "\n" + text.strip() + "\n", encoding="utf-8")


def main() -> str:
    core.configure(1000.0, 0.001)
    if not backtest.BARS_DIR.exists():
        text = "\n".join(cross_section({})).strip() + "\n"
        print(text)
        return text
    series = cm.continuous()
    bundle = build_cross(series)
    text = "\n".join(cross_section(bundle, series)).strip() + "\n"
    out = report.OUT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cr_momentum_cross.md").write_text(text, encoding="utf-8")
    payload = _json_ready(bundle)
    (out / "cr_momentum_cross.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    append_report(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
