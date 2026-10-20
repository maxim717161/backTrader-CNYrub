"""Отдельная проверка недельного моментума на CR (юань–рубль).

Запуск: `PYTHONPATH=src:. python -m research.cr_momentum` пишет `research/REPORT_cr_momentum.md`.

В отчёте по Si/Eu моментум (знак доходности за L дней → позиция на H дней) был значим, но L = 20
выбрано по результату, а на CR он считался только внутри окна одного контракта (до 63 дней).
Здесь:
1. строится непрерывный дневной ряд CR: внутри контракта доходности по его собственным закрытиям,
   на роллировке — по закрытию нового контракта за предыдущий день (файлы контрактов хранят
   около месяца до начала окна), так что контанго между кварталами не попадает в прибыль и
   можно брать L до 120 дней и держать через роллировку;
2. заново считается сетка L×H со всеми проверками на перебор и случайность;
3. моментум проверяется по-другому: ежедневная перебалансировка с кривой капитала и издержками на
   каждую смену позиции, циклический сдвиг сигнала (перестановочный тест), последовательный
   отбор (walk-forward: параметры выбираются только по прошлому), чувствительность к издержкам;
4. сигнал CR сравнивается с сигналом Si и Eu: это тот же рублёвый фактор или собственный эффект.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import core, report

LOOKBACKS = (5, 10, 20, 40, 60, 120)
HOLDS = (1, 5, 10, 20)
PRIMARY = (20, 5)
SIDE_COST = 6.0
SHIFT_REPS = 5000


@dataclass(frozen=True)
class Series:
    """Непрерывный дневной ряд: доходности в пунктах цены и признак роллировки."""

    days: np.ndarray
    contract: np.ndarray
    ret: np.ndarray
    roll: np.ndarray
    level: np.ndarray

    @property
    def price(self) -> np.ndarray:
        return np.cumsum(self.ret)

    def __len__(self) -> int:
        return len(self.ret)


def _daily_closes(path, weekdays_only: bool = True) -> pd.Series:
    frame = pd.read_parquet(path, columns=["datetime", "close"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if weekdays_only:
        frame = frame[frame["datetime"].dt.dayofweek < 5]
    return frame.groupby(frame["datetime"].dt.normalize())["close"].last()


def continuous(bars_dir=core.BARS_DIR) -> Series:
    document = json.loads((bars_dir.parent / "contracts.json").read_text(encoding="utf-8"))
    days, names, rets, rolls, levels = [], [], [], [], []
    previous_contract = None
    for row in document["contracts"]:
        start, end = row.get("window_start"), row.get("window_end")
        path = bars_dir / f"{row['secid']}.parquet"
        if not start or not end or not path.exists():
            continue
        closes = _daily_closes(path)
        diff = closes.diff()
        window = closes[(closes.index >= pd.Timestamp(start)) & (closes.index <= pd.Timestamp(end))]
        for position, day in enumerate(window.index):
            value = diff.loc[day]
            days.append(day)
            names.append(row["secid"])
            levels.append(float(window.loc[day]))
            rets.append(0.0 if value != value else float(value))
            rolls.append(position == 0 and previous_contract is not None)
        if len(window):
            previous_contract = row["secid"]
    order = np.argsort(np.array(days, dtype="datetime64[ns]"), kind="stable")
    return Series(
        days=np.array(days, dtype="datetime64[ns]")[order],
        contract=np.array(names)[order],
        ret=np.array(rets)[order],
        roll=np.array(rolls)[order],
        level=np.array(levels)[order],
    )


def signal(series: Series, lookback: int) -> np.ndarray:
    """Знак доходности за lookback дней на закрытии дня i (0, пока истории не хватает)."""
    price = series.price
    out = np.zeros(len(price))
    out[lookback:] = np.sign(price[lookback:] - price[:-lookback])
    return out


def trades(series: Series, lookback: int, hold: int, sign: int = 1, delay: int = 0, extra_side_cost: float = 0.0, fixed_side: int = 0) -> pd.DataFrame:
    """Сделки на H дней без пересечения; издержки — круг и ещё круг за каждую роллировку внутри позиции."""
    price = series.price
    sig = signal(series, lookback)
    cost = core.round_trip_cost() + 2 * extra_side_cost
    rows = []
    i = lookback
    while i + delay + hold < len(price):
        side = fixed_side or sign * int(sig[i])
        if side:
            enter, leave = i + delay, i + delay + hold
            rolls = int(series.roll[enter + 1 : leave + 1].sum())
            gross = side * (price[leave] - price[enter]) * core.MULTIPLIER
            rows.append(
                {
                    "day": pd.Timestamp(series.days[enter]),
                    "side": side,
                    "rolls": rolls,
                    "gross": gross,
                    "net": gross - cost * (1 + rolls),
                }
            )
        i += hold
    return pd.DataFrame(rows, columns=["day", "side", "rolls", "gross", "net"])


def month_stats(frame: pd.DataFrame, column: str = "net") -> tuple[float, float, int]:
    if frame.empty:
        return float("nan"), float("nan"), 0
    months = frame["day"].dt.to_period("M").astype(str).to_numpy()
    mean, _, t, n = core.cluster_mean(frame[column].to_numpy(), months)
    return mean, t, n


def notional(series: Series) -> float:
    return float(np.median(series.level)) * core.MULTIPLIER


def grid(series: Series) -> tuple[pd.DataFrame, float]:
    base = notional(series)
    days = np.array(series.days)
    rows, columns = [], []
    for lookback, hold in itertools.product(LOOKBACKS, HOLDS):
        frame = trades(series, lookback, hold)
        columns.append(core.daily_pnl(frame, days))
        mean, t, n = month_stats(frame)
        gross_mean, gross_t, _ = month_stats(frame, "gross")
        rows.append(
            {
                "L, дней": lookback,
                "H, дней": hold,
                "сделок": n,
                "gross, б.п.": gross_mean / base * 1e4 if n else float("nan"),
                "net, б.п.": mean / base * 1e4 if n else float("nan"),
                "t": t,
                "плюсовых, %": float((frame["net"] > 0).mean() * 100) if n else float("nan"),
            }
        )
    p, _ = core.reality_check(np.column_stack(columns), reps=1000)
    return pd.DataFrame(rows), float(p)


def daily_rebalanced(series: Series, lookback: int, extra_side_cost: float = 0.0) -> pd.DataFrame:
    """Позиция на каждый день по знаку вчерашнего сигнала; издержки с каждой изменённой единицы позиции."""
    sig = signal(series, lookback)
    position = np.r_[0.0, sig[:-1]]
    gross = position * series.ret * core.MULTIPLIER
    change = np.abs(np.diff(np.r_[0.0, position]))
    rolls = np.where((series.roll) & (position != 0), 2 * (SIDE_COST + extra_side_cost), 0.0)
    cost = change * (SIDE_COST + extra_side_cost) + rolls
    return pd.DataFrame({"day": pd.to_datetime(series.days), "position": position, "gross": gross, "cost": cost, "net": gross - cost})


def equity_stats(frame: pd.DataFrame, base: float) -> dict[str, float]:
    live = frame[frame["position"] != 0]
    net = frame["net"].to_numpy()
    equity = np.cumsum(net)
    drawdown = float((np.maximum.accumulate(equity) - equity).max())
    years = (frame["day"].iloc[-1] - frame["day"].iloc[0]).days / 365.25
    std = float(net.std(ddof=1))
    return {
        "дней в позиции": int(len(live)),
        "перекладок": int((np.abs(np.diff(np.r_[0.0, frame["position"].to_numpy()])) > 0).sum()),
        "итого net, ₽": float(net.sum()),
        "gross, ₽": float(frame["gross"].sum()),
        "издержки, ₽": float(frame["cost"].sum()),
        "годовая доходность от номинала, %": float(net.sum() / years / base * 100),
        "Шарп (годовой)": float(net.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        "макс. просадка, ₽": drawdown,
        "макс. просадка, % номинала": drawdown / base * 100,
    }


def equity_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Рубли и дни без дробной части, доходность и просадка с одним знаком, Шарп с двумя."""
    out = pd.DataFrame(index=stats.index)
    for column in stats.columns:
        if "Шарп" in column:
            out[column] = stats[column].map(lambda v: f"{v:.2f}")
        elif "%" in column:
            out[column] = stats[column].map(lambda v: f"{v:.1f}")
        else:
            out[column] = stats[column].map(lambda v: f"{v:.0f}")
    return out


def shift_test(series: Series, lookback: int, reps: int = SHIFT_REPS, hold: int = 1) -> dict[str, float]:
    """Циклический сдвиг сигнала относительно доходностей: сохраняет автокорреляцию того и другого,
    разрушает их связь. Статистика — средний gross дня при ежедневной перебалансировке (hold=1)."""
    sig = signal(series, lookback)[:-1]
    ret = series.ret[1:]
    n = len(ret)
    observed = float(np.mean(sig * ret))
    rng = np.random.default_rng(core.SEED)
    margin = max(lookback, 20)
    shifts = rng.integers(margin, n - margin, size=reps)
    nulls = np.empty(reps)
    for i, k in enumerate(shifts):
        nulls[i] = np.mean(np.roll(sig, int(k)) * ret)
    p = (1 + int((nulls >= observed).sum())) / (reps + 1)
    return {"наблюдаемый": observed * core.MULTIPLIER, "нуль, среднее": float(nulls.mean() * core.MULTIPLIER), "нуль, 95-й процентиль": float(np.percentile(nulls, 95) * core.MULTIPLIER), "p": p}


def walk_forward(series: Series, first_test_year: int = 2023) -> tuple[pd.DataFrame, dict[str, float]]:
    """Параметры выбираются по лучшему t на всех сделках до года теста, затем торгуется год теста."""
    variants = {(l, h): trades(series, l, h) for l, h in itertools.product(LOOKBACKS, (5, 10, 20))}
    base = notional(series)
    years = sorted({d.year for d in pd.to_datetime(series.days)})
    rows, oos = [], []
    for year in years:
        if year < first_test_year:
            continue
        best, best_t = None, -np.inf
        for key, frame in variants.items():
            past = frame[frame["day"].dt.year < year]
            if len(past) < 20:
                continue
            _, t, _ = month_stats(past)
            if t == t and t > best_t:
                best, best_t = key, t
        if best is None:
            continue
        current = variants[best]
        current = current[current["day"].dt.year == year]
        mean, t, n = month_stats(current)
        oos.append(current)
        rows.append({"год теста": year, "выбрано L": best[0], "выбрано H": best[1], "t в выборке": best_t, "сделок": n, "net, б.п.": mean / base * 1e4 if n else float("nan"), "t": t})
    merged = pd.concat(oos) if oos else pd.DataFrame(columns=["day", "net"])
    mean, t, n = month_stats(merged)
    return pd.DataFrame(rows), {"сделок": n, "net, б.п.": mean / base * 1e4 if n else float("nan"), "t": t}


def cost_sensitivity(series: Series, lookback: int = 20) -> pd.DataFrame:
    base = notional(series)
    rows = []
    for extra in (0, 1, 2, 5, 10, 20, 50):
        row = {"доп. проскальзывание, ₽ на сторону": extra, "круг, ₽": core.round_trip_cost() + 2 * extra}
        for hold in (5, 10, 20):
            frame = trades(series, lookback, hold, extra_side_cost=extra)
            mean, t, n = month_stats(frame)
            row[f"H={hold}: net, б.п."] = mean / base * 1e4
            row[f"H={hold}: t"] = t
        rows.append(row)
    return pd.DataFrame(rows)


def variants_table(series: Series, lookback: int = 20, hold: int = 5) -> pd.DataFrame:
    """Что меняет эффект: зеркало, вход на день позже, только покупки или только продажи, без 2022 года."""
    base = notional(series)
    rows = []
    cases = {
        "правило": trades(series, lookback, hold),
        "зеркало (против сигнала)": trades(series, lookback, hold, sign=-1),
        "вход на день позже": trades(series, lookback, hold, delay=1),
        "вход на 2 дня позже": trades(series, lookback, hold, delay=2),
        "всегда покупка": trades(series, lookback, hold, fixed_side=1),
        "всегда продажа": trades(series, lookback, hold, fixed_side=-1),
    }
    right = cases["правило"]
    cases["только покупки по сигналу"] = right[right["side"] == 1]
    cases["только продажи по сигналу"] = right[right["side"] == -1]
    cases["без 2022 года"] = right[right["day"].dt.year != 2022]
    cases["только 2022 год"] = right[right["day"].dt.year == 2022]
    for name, frame in cases.items():
        mean, t, n = month_stats(frame)
        rows.append({"вариант": name, "сделок": n, "net, б.п.": mean / base * 1e4 if n else float("nan"), "t": t})
    return pd.DataFrame(rows).set_index("вариант")


def by_year(series: Series, lookback: int = 20, hold: int = 5) -> pd.DataFrame:
    base = notional(series)
    frame = trades(series, lookback, hold)
    rows = []
    for year, part in frame.groupby(frame["day"].dt.year):
        mean, t, n = month_stats(part)
        rows.append({"год": year, "сделок": n, "net, б.п.": mean / base * 1e4, "t": t, "плюсовых, %": float((part["net"] > 0).mean() * 100)})
    return pd.DataFrame(rows).set_index("год")


def other_closes(name: str) -> pd.DataFrame:
    from research import instruments

    market, _ = instruments.load(name)
    core.configure(1000.0, 0.001)
    from research import daily

    closes = daily.daily_closes(market)
    return closes


def cross_signal(series: Series, name: str, lookback: int = 20, hold: int = 5) -> dict[str, object]:
    """Торговать CR по знаку 20-дневной доходности Si или Eu внутри одного квартального контракта.

    Контракты сопоставляются по дате экспирации (третий четверг), история сигнала берётся из окна
    контракта Si/Eu целиком, то есть и за дни до начала торгов CR."""
    from research import fetch, instruments

    other, _ = instruments.load(name)
    core.configure(1000.0, 0.001)
    expiry = {w.name: w.expiry for w in fetch.windows(name, pd.Timestamp.today().date())}
    from research import daily

    closes = daily.daily_closes(other)
    closes["expiry"] = closes["contract"].map(expiry)
    sign_by_key = {}
    for _, part in closes.groupby("contract", sort=False):
        price = part["close"].to_numpy()
        for i in range(lookback, len(price)):
            sign_by_key[(part["expiry"].iloc[0], pd.Timestamp(part["day"].iloc[i]))] = int(np.sign(price[i] - price[i - lookback]))

    document = json.loads((core.BARS_DIR.parent / "contracts.json").read_text(encoding="utf-8"))
    end_of = {row["secid"]: pd.Timestamp(row["window_end"]).date() for row in document["contracts"] if row.get("window_end")}
    price = series.price
    own = signal(series, lookback)
    rows = []
    i = lookback
    cost = core.round_trip_cost()
    agree = total = 0
    while i + hold < len(price):
        key = (end_of.get(str(series.contract[i])), pd.Timestamp(series.days[i]))
        other_side = sign_by_key.get(key)
        if other_side:
            gross = other_side * (price[i + hold] - price[i]) * core.MULTIPLIER
            rolls = int(series.roll[i + 1 : i + hold + 1].sum())
            rows.append({"day": pd.Timestamp(series.days[i]), "side": other_side, "gross": gross, "net": gross - cost * (1 + rolls)})
            if own[i]:
                total += 1
                agree += int(own[i] == other_side)
        i += hold
    frame = pd.DataFrame(rows, columns=["day", "side", "gross", "net"])
    mean, t, n = month_stats(frame)
    return {"name": name, "n": n, "net_bp": mean / notional(series) * 1e4 if n else float("nan"), "t": t, "agree": agree / total if total else float("nan")}


def build_report(series: Series) -> str:
    base = notional(series)
    grid_table, white = grid(series)
    primary = grid_table[(grid_table["L, дней"] == PRIMARY[0]) & (grid_table["H, дней"] == PRIMARY[1])].iloc[0]
    daily_20 = daily_rebalanced(series, 20)
    stats_20 = equity_stats(daily_20, base)
    stats_by_l = pd.DataFrame({f"L={l}": equity_stats(daily_rebalanced(series, l), base) for l in LOOKBACKS}).T
    shifts = pd.DataFrame({f"L={l}": shift_test(series, l) for l in (5, 10, 20, 40, 60, 120)}).T
    walk, walk_total = walk_forward(series)
    cross = [cross_signal(series, name) for name in ("Si", "Eu")]
    cross_frame = pd.DataFrame(
        [
            {"сигнал": f"{c['name']} (20 дн.)", "сделок": c["n"], "net, б.п.": c["net_bp"], "t": c["t"], "совпадение знака с собственным сигналом CR": c["agree"]}
            for c in cross
        ]
    ).set_index("сигнал")
    own = variants_table(series)
    out = [
        "# Моментум на CR: отдельная проверка\n",
        METHOD.format(first=pd.Timestamp(series.days[0]).date(), last=pd.Timestamp(series.days[-1]).date(), n=len(series), rolls=int(series.roll.sum()), notional=base, cost=core.round_trip_cost()),
        "## 1. Сетка L × H на непрерывном ряду\n",
        f"Знак доходности за L дней → позиция на H дней, сделки не пересекаются, роллировка внутри позиции стоит ещё один круг. net и gross — в б.п. номинала; t — с кластеризацией по месяцам. Проверка перебора (White) по всей сетке из {len(grid_table)} вариантов: **p = {white:.3f}**.\n",
        report.md(grid_table, 1, index=False),
        f"\nПервичный вариант L = {PRIMARY[0]}, H = {PRIMARY[1]} (выбран по Si/Eu, не по CR): net **{primary['net, б.п.']:+.1f} б.п.**, t = {primary['t']:+.1f}, сделок {int(primary['сделок'])}, gross {primary['gross, б.п.']:+.1f} б.п.\n",
        "### Что меняет эффект (L = 20, H = 5)\n",
        report.md(own, 1),
        "\n### По годам (L = 20, H = 5)\n",
        report.md(by_year(series), 1),
        "\n## 2. Ежедневная перебалансировка с кривой капитала\n",
        f"Позиция каждый день по знаку вчерашнего сигнала, издержки {SIDE_COST:.0f} ₽ за каждую единицу изменения позиции и круг за роллировку в позиции. Результат на 1 контракт.\n",
        report.md(equity_table(stats_by_l), 1),
        "\n## 3. Перестановочный тест (циклический сдвиг сигнала)\n",
        f"Средний gross дня (₽ на контракт) при ежедневной перебалансировке против распределения при {SHIFT_REPS} случайных циклических сдвигах сигнала относительно доходностей. p — доля сдвигов не хуже наблюдаемого.\n",
        report.md(shifts, 3),
        "\n## 4. Последовательный отбор (walk-forward)\n",
        "На каждом году теста L и H выбираются по лучшему t на всех сделках прошлых лет (сетка L × H ∈ {5, 10, 20}), затем торгуется год теста.\n",
        report.md(walk, 1, index=False),
        f"\nВсе тестовые годы вместе: сделок {walk_total['сделок']}, net **{walk_total['net, б.п.']:+.1f} б.п.**, t = {walk_total['t']:+.1f}.\n",
        "## 5. Чувствительность к издержкам (L = 20)\n",
        "Дополнительное проскальзывание в рублях на контракт на каждую сторону (шаг цены CR стоит 1 ₽).\n",
        report.md(cost_sensitivity(series), 1, index=False),
        "\n## 6. Это сигнал CR или общий рублёвый фактор\n",
        "Торговля CR по знаку 20-дневной доходности Si или Eu (H = 5), та же схема и издержки.\n",
        report.md(cross_frame, 2),
        "\n## 7. Выводы\n\n" + CONCLUSIONS,
    ]
    return "\n".join(out)


METHOD = """## Данные и метод

- Непрерывный дневной ряд CR: {first} … {last}, {n} торговых дней (будни), {rolls} роллировок по окнам из `data/contracts.json`. Доходность дня — по закрытию дня внутри контракта; в день роллировки — по закрытию нового контракта за предыдущий день. Разница цен кварталов (около 2–4% из-за разницы ставок) в прибыль не попадает.
- Номинал контракта около {notional:,.0f} ₽; круг {cost:.0f} ₽ (комиссия 5 ₽ + 1 тик проскальзывания с каждой стороны).
- Вход и выход по цене закрытия дня.
"""

CONCLUSIONS = """**Моментум на CR есть, но доказан слабо: знак устойчив, статистическая значимость на границе.**

За моментум:

- Знак эффекта один во всей сетке. При удержании 5 дней и больше net положителен в 17 вариантах из 18 (L от 5 до 120), при H = 1 gross положителен везде, а net отрицателен только из-за издержек.
- Основной вариант L = 20, H = 5 (параметры взяты по Si/Eu, а не подогнаны под CR): net +35 б.п. на сделку, t = 2,4, 219 сделок; при H = 10 и 20 net +74 и +144 б.п., t = 2,6 и 2,2. Зеркальное правило стабильно убыточно (−57 б.п., t = −4,0), а покупка и продажа без сигнала дают ноль или минус (−18 и −4 б.п.), то есть прибыль не от дрейфа рубля.
- Не держится на одном годе или одной стороне: все годы 2022–2026 положительны (23–52 б.п.), без 2022 года +32 б.п. (t = 2,2), продажи по сигналу +39 б.п., покупки +30 б.п. Вход на день или два позже почти ничего не отнимает (+38 и +37 б.п.), то есть эффект медленный и не зависит от точности исполнения.
- Ежедневная перебалансировка при L = 20: +10 тыс. ₽ на контракт за 4,4 года (около 19% номинала в год), Шарп 1,06, максимальная просадка около 2,1 тыс. ₽ (18% номинала), 98 перекладок. Все L дают плюс, лучший по результату L = 20 — тот же, что выбран заранее.
- Перестановочный тест (циклический сдвиг сигнала): для L = 20 p = 0,011, для L = 5 p = 0,044; результат заметно выше случайного сдвига.
- Walk-forward (выбор L и H только по прошлому): сделки 2023–2026 дают +47 б.п., t = 2,0; выбирался всегда L = 20.
- Эффект выдерживает умеренное ухудшение исполнения: при H = 5 до +5 ₽ проскальзывания на сторону net > 0 (t = 1,8), при H = 10 до +20 ₽ на сторону (t = 1,2), при H = 20 даже до +50 ₽.

Против:

- Проверка всей сетки на перебор (White) даёт p = 0,265: из 24 вариантов поправка на то, что мы смотрели все, значимость не оставляет. Точечно значимы только L = 20 (все три H) и L = 40 при H = 10; с поправкой на 6 значений L наименьший p перестановочного теста (0,011) превращается примерно в 0,07.
- Мало независимых наблюдений: 4,4 года и около 55 непересекающихся окон по 20 дней. По отдельным годам t не выше 1,4, в walk-forward по годам не выше 1,3.
- CR торгуется как единый рублёвый фактор с Si и Eu: 20-дневный знак Si или Eu совпадает со знаком CR в 97% и 90% случаев и даёт CR тот же результат (+42 и +39 б.п., t = 2,2 и 2,0). Поэтому Si и Eu не являются независимым подтверждением CR после апреля 2022 года; независимый тест возможен только на Si/Eu до апреля 2022 года, и там эффект есть (см. `REPORT_longer.md`).
- Издержки существенны: при H = 5 эффект около трёх круговых издержек (12 ₽ ≈ 10 б.п. при gross 46 б.п.); при ухудшении исполнения на 20 ₽ за сторону он исчезает.

Итог: по CR моментум — самая устойчивая из всех проверенных идей (трендовая и контртрендовая внутридневные не подтвердились), но подтверждена она в основном общим для Si/Eu рублёвым трендом одного режима. Перед торговлей разумно: медленные варианты (L = 20, H = 10…20), исполнение с проскальзыванием не хуже 1–2 ₽ на сторону и размер позиции, рассчитанный на просадку около 20% номинала; в живую функцию правило не добавляется без отдельного решения.
"""


def main() -> None:
    core.configure(1000.0, 0.001)
    series = continuous()
    text = build_report(series)
    (report.OUT / "REPORT_cr_momentum.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
