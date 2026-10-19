"""Проверка выводов по CR на инструментах с более длинной историей: Si (USD/RUB) и Eu (EUR/RUB).

Запуск: `PYTHONPATH=src:. python -m research.longer [CR Si Eu]` (данные Si/Eu сначала качаются
`python -m research.fetch Si Eu`). Пишет `research/REPORT_longer.md` (сводка и выводы),
`research/REPORT_longer_<актив>.md` (полные отчёты) и `research/out/longer_*.csv`.

Схема, исключающая самообман:
- до 21.04.2022 CR ещё не существовало, значит выводы по CR не могли подсматривать в эти данные:
  для Si и Eu период до этой даты — настоящая проверка «на других данных», а после — на тех
  же рублёвых движениях, что и у CR (коррелированная проверка);
- правила и пороги не подбираются заново: используются те же сетки, что в отчётах по CR, с
  порогами в «тиках сравнения» (одинаковые доли цены);
- кандидат из отчёта по CR — ставка против разрыва вверх ≥20 тиков сравнения — проверяется
  отдельно и заранее заданным способом.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from research import battery, core, countertrend, daily, evaluate, instruments, report, trend
from research.families import Spec, build

CR_START = pd.Timestamp("2022-04-21")
SPLITS = {"CR": pd.Timestamp("2024-07-01"), "Si": CR_START, "Eu": CR_START}
OUT = report.OUT / "out"

METHOD_TEXT = """## Данные и метод

- Ряд: минутки фронтального квартального контракта {title}, окна — от дня после экспирации предыдущего до экспирации (третий четверг месяца), только будни. {first} … {last}: {days} торговых дней, {contracts} контрактов.
- Шаг цены стоит 1 ₽ на контракт, как у CR, поэтому круг тот же — **{cost:.0f} ₽** (комиссия 5 ₽ с каждой стороны + 1 тик проскальзывания с каждой), но номинал контракта — около {notional:,.0f} ₽, то есть круг стоит **{bp:.1f} б.п.** номинала (у CR — 9,6 б.п.).
- Пороги правил заданы так же, как для CR, но в «тиках сравнения»: 1 тик сравнения = {unit:g} пункта цены = 0,008% цены (у CR — ровно один шаг цены). Так «разрыв 20 тиков» везде означает 0,16% цены.
- Сигнал по закрытию минуты, вход по закрытию следующей минуты; доверительные интервалы и t — по дням (кластерно); проверка перебора White.
- Деление на обучение и проверку: **{split}**. {split_note}
"""

SPLIT_NOTE = {
    "CR": "Как в отчётах по CR.",
    "Si": "До этой даты CR не существовало: обучение — данные, которых отчёты по CR не видели; проверка — период, общий с CR.",
    "Eu": "До этой даты CR не существовало: обучение — данные, которых отчёты по CR не видели; проверка — период, общий с CR.",
}


def gap_periods(market: core.Market, data: dict) -> pd.DataFrame:
    """Ставка против ночного разрыва до конца дня: по порогу, направлению и периоду (gross, net, t по gross)."""
    rows = []
    for threshold in (10, 20, 30):
        trades = core.run_entries(data, build(Spec("gap", -1, dict(min_gap=threshold, hold=0)), market, data))
        trades = trades.assign(day=pd.to_datetime(trades["day"]))
        periods = (
            ("до 21.04.2022", trades[trades["day"] < CR_START]),
            ("с 21.04.2022", trades[trades["day"] >= CR_START]),
            ("весь период", trades),
        )
        for name, part in periods:
            for direction, side in (("вверх (продаём)", -1), ("вниз (покупаем)", 1), ("оба", 0)):
                sub = part if side == 0 else part[part["side"] == side]
                mean, _, t, n = core.cluster_mean(sub["gross"].to_numpy(), sub["day"].to_numpy())
                rows.append(
                    {
                        "порог, тиков сравнения": threshold,
                        "период": name,
                        "разрыв": direction,
                        "сделок": n,
                        "gross": mean,
                        "net": mean - core.round_trip_cost() if n else float("nan"),
                        "t (gross)": t,
                    }
                )
    return pd.DataFrame(rows)


def night_family(result: evaluate.GridResult) -> dict[str, float]:
    """Варианты с удержанием через ночь (самый слабый намёк на тренд у CR) — отдельной строкой."""
    table = result.table
    night = table[table.index.str.contains("ночь")]
    return {
        "вариантов": len(night),
        "медиана net, обучение": float(night["обучение: net"].median()),
        "медиана net, проверка": float(night["проверка: net"].median()),
        "в плюсе на обучении": int((night["обучение: net"] > 0).sum()),
        "в плюсе на проверке": int((night["проверка: net"] > 0).sum()),
        "t>2 на проверке": int((night["проверка: t"] > 2).sum()),
    }


CRISES = (("2020-02-15", "2020-04-30"), ("2022-02-15", "2022-04-30"))


def candidates(market: core.Market, data: dict) -> pd.DataFrame:
    """Кандидаты из отчётов по CR и их зеркала (то же правило с обратным знаком) по периодам.

    Зеркало — главный тест на самообман: если прибыль не зависит от знака сигнала, источник
    прибыли не направление, а устройство выхода (стоп, удержание) или дрейф."""
    unit = core.UNIT
    notional = float(market.bars["close"].median()) * core.MULTIPLIER
    plan = [
        ("тренд: пробой 60, стоп 4, ночь", Spec("donchian", 1, dict(channel=60, stop=4, min_stop=8 * unit), False)),
        ("тренд: z-score 60, k=2, стоп 8, ночь", Spec("zscore", 1, dict(window=60, k=2, stop=8, min_stop=8 * unit), False)),
        ("контртренд: разрыв ≥20, до конца дня", Spec("gap", -1, dict(min_gap=20, hold=0))),
    ]
    rows = []
    for title, spec in plan:
        for kind, sign in (("правило", spec.sign), ("зеркало", -spec.sign)):
            use = Spec(spec.family, sign, spec.params, spec.flatten_day)
            trades = core.run_entries(data, build(use, market, data), flatten_day=use.flatten_day)
            trades = trades.assign(day=pd.to_datetime(trades["day"]))
            crisis = np.zeros(len(trades), dtype=bool)
            for start, end in CRISES:
                crisis |= ((trades["day"] >= start) & (trades["day"] <= end)).to_numpy()
            periods = (
                ("до 21.04.2022", trades[trades["day"] < CR_START]),
                ("с 21.04.2022", trades[trades["day"] >= CR_START]),
                ("весь период", trades),
                ("без кризисных эпизодов 2020 и 2022", trades[~crisis]),
            )
            for period, part in periods:
                stats = core.trade_stats(part)
                rows.append(
                    {
                        "кандидат": title,
                        "вид": kind,
                        "период": period,
                        "сделок": stats["n"],
                        "net, ₽": stats["net"],
                        "net, б.п.": stats["net"] / notional * 1e4 if stats["n"] else float("nan"),
                        "t": stats["t"],
                    }
                )
    return pd.DataFrame(rows)


def run_instrument(name: str) -> dict[str, object]:
    market, instrument = instruments.load(name)
    data = core.arrays(market)
    evaluate.SPLIT = SPLITS[name]
    info = instruments.describe(market, instrument)
    print(f"{name}: {info['минуток']} минуток, {info['дней']} дней", flush=True)

    described = battery.run_all(market)
    out: dict[str, object] = {"name": name, "info": info, "battery": described}

    for key, module in (("trend", trend), ("counter", countertrend)):
        plan = module.specs()
        result = evaluate.run_grid(plan, market, data)
        chosen = evaluate.selection(result)
        by_label = {spec.label(): spec for spec in plan}
        best = by_label[chosen["best_in_sample"]]
        out[key] = {
            "plan": plan,
            "result": result,
            "selection": chosen,
            "detail": evaluate.detail(best, market, data),
            "best": best,
            "night": night_family(result),
        }
        OUT.mkdir(exist_ok=True)
        result.table.round(3).to_csv(OUT / f"longer_{name}_{key}.csv", encoding="utf-8")
        print(f"{name}: {key}: готово", flush=True)

    out["momentum_grid"], out["momentum_p"] = daily.grid(market, CR_START)
    out["momentum_drift"] = daily.drift_controls(market)
    out["momentum_robust"] = daily.robustness(market)
    out["momentum_years"] = daily.by_year(market)
    out["candidates"] = candidates(market, data)
    out["gaps"] = gap_periods(market, data)
    out["gap_table"], out["gap_years"] = countertrend.gap_table(market, data)
    out["bounce"] = countertrend.bounce_table(market, data)
    return out


def momentum_section(res: dict[str, object]) -> str:
    grid = res["momentum_grid"].drop(columns=["до: t", "после: t"])
    return f"""## 4. Моментум на дневном масштабе

Знак доходности за L дней определяет позицию на H дней (без стопов; сделки внутри контракта не пересекаются; вход и выход по закрытию дня; круг — {core.round_trip_cost():.0f} ₽). t — с кластеризацией по месяцам; net — в б.п. номинала. «До» — до 21.04.2022. Проверка перебора (White) по всей сетке L×H: **p = {res['momentum_p']:.3f}**.

{report.md(grid, 1, index=False)}

Контроль дрейфа (всегда в одну сторону):

{report.md(res['momentum_drift'], 1, index=False)}

Устойчивость при L = 20 дней:

{report.md(res['momentum_robust'], 1)}

L = 20, H = 5 по годам:

{report.md(res['momentum_years'], 1)}
"""


def instrument_report(res: dict[str, object]) -> str:
    info = res["info"]
    name = res["name"]
    method = METHOD_TEXT.format(
        title=info["инструмент"],
        first=info["с"],
        last=info["по"],
        days=info["дней"],
        contracts=info["контрактов"],
        cost=info["круг, ₽"],
        notional=info["номинал, ₽"],
        bp=info["круг, б.п."],
        unit=info["тик сравнения"],
        split=SPLITS[name].date(),
        split_note=SPLIT_NOTE[name],
    )
    trend_part, counter_part = res["trend"], res["counter"]
    gaps = res["gaps"]
    return "\n".join(
        [
            f"# {info['инструмент']}: те же проверки, что для CR\n",
            method,
            report.battery_section(res["battery"], trend.FOCUS + " " + countertrend.FOCUS),
            report.grid_section("2. Тренд: правила следования за движением", trend.INTRO, trend_part["result"], trend_part["plan"], trend_part["selection"]),
            report.detail_section(trend_part["best"], trend_part["detail"]),
            report.grid_section("3. Контртренд: правила игры против движения", countertrend.INTRO, counter_part["result"], counter_part["plan"], counter_part["selection"]),
            report.detail_section(counter_part["best"], counter_part["detail"]),
            countertrend.bounce_section(res["bounce"]),
            momentum_section(res),
            "### Кандидаты из отчётов по CR и их зеркала\n\nЗеркало — то же правило с обратным знаком; net на контракт, в ₽ и в б.п. номинала.\n\n" + report.md(res["candidates"], 1, index=False) + "\n",
            "### Закрытие ночного разрыва по периодам\n\ngross и net — ₽ на контракт, выход в конце дня.\n\n" + report.md(gaps, 1, index=False) + "\n",
            countertrend.gap_section(res["gap_table"], res["gap_years"]),
        ]
    )


def pick(frame: pd.DataFrame, **conditions) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= (frame[column] == value).to_numpy()
    return frame[mask].iloc[0]


def summary_tables(results: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    passport, memory, trend_rows, counter_rows, gap_rows, night_rows = [], [], [], [], [], []
    for res in results:
        name = res["name"]
        info = dict(res["info"])
        passport.append({"": name, **{k: v for k, v in info.items() if k != "инструмент"}, "инструмент": info["инструмент"]})
        b = res["battery"]
        vr = b["vr"]
        vr30 = vr[(vr["шаг"] == "5 мин") & (vr["горизонт"] == "30 мин")].iloc[0]["VR без шума последней сделки"]
        vr120 = vr[(vr["шаг"] == "5 мин") & (vr["горизонт"] == "120 мин")].iloc[0]["VR без шума последней сделки"]
        years = b["years"]
        memory.append(
            {
                "": name,
                "автокорр. 1 мин, лаг 1": b["autocorr"].loc["1 мин", "лаг 1"],
                "автокорр. 15 мин, лаг 1": b["autocorr"].loc["15 мин", "лаг 1"],
                "VR 30 мин без шума": vr30,
                "VR 120 мин без шума": vr120,
                "Хёрст": b["hurst"],
                "Хёрст по годам, мин…макс": f"{years['Хёрст'].min():.2f}…{years['Хёрст'].max():.2f}",
                "VR 5 дн": b["daily_vr"].loc["VR", "5 дн"],
                "VR 20 дн": b["daily_vr"].loc["VR", "20 дн"],
                "наклон разрыв→первый час": b["gaps"].iloc[0]["наклон"],
            }
        )
        for key, rows in (("trend", trend_rows), ("counter", counter_rows)):
            part = res[key]
            sel = part["selection"]
            best = sel["best_in_sample"]
            table = part["result"].table
            rows.append(
                {
                    "": name,
                    "вариантов": sel["variants"],
                    "gross>0": sel["gross_positive"],
                    "net>0": sel["net_positive"],
                    "t>2 обуч.": sel["is_significant"],
                    "t>2 проверка": sel["oos_significant"],
                    "White p (обуч.)": sel["p_in_sample"],
                    "White p (всё)": sel["p_all"],
                    "лучший по обучению": best,
                    "его net на проверке": table.loc[best, "проверка: net"],
                    "его t на проверке": table.loc[best, "проверка: t"],
                }
            )
        night_rows.append({"": name, **res["trend"]["night"]})
        gaps = res["gaps"]
        for period in ("до 21.04.2022", "с 21.04.2022", "весь период"):
            for direction in ("вверх (продаём)", "вниз (покупаем)", "оба"):
                row = pick(gaps, **{"порог, тиков сравнения": 20, "период": period, "разрыв": direction})
                if name == "CR" and period != "весь период":
                    continue
                gap_rows.append(
                    {
                        "": name,
                        "период": period,
                        "разрыв": direction,
                        "сделок": row["сделок"],
                        "gross": row["gross"],
                        "net": row["net"],
                        "t (gross)": row["t (gross)"],
                    }
                )
    momentum_rows = []
    for res in results:
        grid = res["momentum_grid"]
        valid = grid[grid["всё: сделок"] >= 20]
        main = grid[(grid["L, дней"] == 20) & (grid["H, дней"] == 5)].iloc[0]

        def cell(prefix: str) -> str:
            if not main[f"{prefix}: сделок"]:
                return "—"
            return f"{main[f'{prefix}: net, б.п.']:+.1f} б.п. (t={main[f'{prefix}: t']:+.1f}, n={int(main[f'{prefix}: сделок'])})"

        momentum_rows.append(
            {
                "": res["name"],
                "вариантов L×H": len(valid),
                "net>0 всего": int((valid["всё: net, б.п."] > 0).sum()),
                "t>2 всего": int((valid["всё: t"] > 2).sum()),
                "White p": res["momentum_p"],
                "L=20, H=5: до 21.04.2022": cell("до"),
                "L=20, H=5: после": cell("после"),
                "L=20, H=5: весь период": cell("всё"),
            }
        )
    pivot = {}
    for res in results:
        cand = res["candidates"]
        for _, row in cand.iterrows():
            key = (row["кандидат"], row["вид"], row["период"])
            pivot.setdefault(key, {})[res["name"]] = f"{row['net, б.п.']:+.1f} б.п. (t={row['t']:+.1f}, n={int(row['сделок'])})"
    candidate_frame = pd.DataFrame(pivot).T.fillna("—")
    candidate_frame.index = [" / ".join(k) for k in candidate_frame.index]
    index = lambda rows: pd.DataFrame(rows).set_index("")  # noqa: E731
    return {
        "passport": index(passport),
        "memory": index(memory),
        "trend": index(trend_rows),
        "counter": index(counter_rows),
        "night": index(night_rows),
        "gaps": pd.DataFrame(gap_rows).set_index(""),
        "candidates": candidate_frame,
        "momentum": pd.DataFrame(momentum_rows).set_index(""),
    }


CONCLUSIONS = """**Вердикт: на длинной истории подтвердился не внутридневной тренд и не контртренд, а медленный тренд на масштабе недель: знак доходности за 20 торговых дней предсказывает следующие 5–20 дней. Он значим на Si и Eu, согласован по знаку на CR, переживает издержки с запасом, но выбран из сетки по результату и его независимая проверка (2019 – апрель 2022) пока слабая по t.** Кандидат из отчёта по CR — закрытие ночных разрывов вверх — не подтвердился.

### 0. Поправка к прошлым отчётам

При подготовке этой ветки найдена ошибка в симуляторе сделок: стоп исполнялся ровно по цене стопа, даже если минутка открылась далеко за ним (разрыв ночью, на выходных). Теперь стоп исполняется по худшей из цены стопа и открытия минутки. На CR поправка мала (число прибыльных вариантов тренда 19 → 18, p White 0,98 → 0,99 и 0,78 → 0,84), но отчёты по тренду и контртренду пересчитаны, а на Si и Eu разрывы большие, и без поправки все цифры про удержание через ночь были бы завышены.

### 1. Что дала длинная история

- ISS отдаёт минутки квартальных фьючерсов только с контрактов, выпущенных в конце 2018 года, поэтому Si и Eu сняты с марта 2019: 1975 торговых дней против 1118 у CR (7,8 против 4,4 года), включая 2019–2021 до появления CR и весну 2022. Дальше в прошлое минутной истории нет.
- Круг стоит 12 ₽ на каждом инструменте, но номинал контракта Si/Eu в 6–8 раз больше: 1,3–1,6 б.п. против 10,2 б.п. у CR. Если у CR малый эффект съедался издержками, то у Si/Eu он должен был проявиться.
- Si, Eu и CR — разные выражения одного рублёвого фактора, поэтому после апреля 2022 они не независимы. Независимая проверка — только Si и Eu до 21.04.2022 (3,1 года).

### 2. Память цены

1. Отрицательная автокорреляция минуток у CR (−0,137) оказалась в основном эффектом крупного шага цены: у Si −0,014, у Eu −0,033 (относительный шаг в 6–7 раз меньше). Шум последней сделки — свойство инструмента, а не рынка.
2. Внутри дня Si близок к случайному блужданию: VR 30 минут без шума 0,970 [0,941; 0,996], 120 минут 0,971 [0,923; 1,023], Хёрст 0,507. Eu заметно возвратнее и тоньше: 0,898 [0,859; 0,963] и 0,702 [0,490; 0,998], Хёрст 0,446; у CR 0,862 и 0,816, Хёрст 0,450. Возврат живёт в менее ликвидном Eu и в CR, а не в самом ликвидном Si.
3. Дневной масштаб: VR 5 дней — Si 1,12 [0,89; 1,25], Eu 0,98, CR 1,00; VR 20 дней — Si 1,18 [0,73; 1,64], Eu 1,14 [0,67; 1,65], CR 0,77 [0,48; 1,06]. Интервалы широкие (контрактов 19–32), по одним дисперсиям слабый тренд не виден.
4. Закрытие ночного разрыва за первый час, которое было у CR (наклон −0,37), **не воспроизводится**: Si +0,15 [−0,57; +2,55], Eu +0,17 [+0,03; +0,42] — разрыв скорее продолжается.

### 3. Внутридневные правила

- Контртренд (174 варианта): после издержек в плюсе у Si 0, у Eu 3 из 174 (у CR 2); до издержек в плюсе всего 12 и 14. Проверка перебора p ≈ 1,00. Игра против движения не работает ни до, ни после издержек.
- Закрытие разрыва вверх (кандидат из CR): у Si и Eu ставка против разрыва ≥ 20 тиков сравнения даёт gross −29 ₽ (t −0,2) и −18 ₽ (t −0,2); после апреля 2022 против разрыва вверх: Si −74 ₽, Eu −43 ₽. До апреля 2022 у Eu разрывы вверх дают +495 ₽ (t 1,1), но разрывы вниз −525 ₽ (t −1,7): обе стороны сдвинуты в одну и ту же сторону (цена в такие дни в среднем падала), это не закрытие разрыва. Асимметрия «вверх лучше вниз», которую мы видели на CR, — не свойство разрывов.
- Тренд (82 варианта): у Si 56 из 80 и у Eu 54 из 82 вариантов в плюсе после издержек (у CR 18), на проверке t > 2 у 12 и 21 вариантов; p White по всему периоду 0,08 и 0,07, по обучению (до 21.04.2022) 0,81 и 0,78. Все 26 вариантов с удержанием через ночь в плюсе и на обучении, и на проверке (медиана net +220 и +229 ₽ на обучении, +90 и +97 ₽ на проверке).
- **Но зеркало таких вариантов тоже в плюсе.** Пробой 60 со стопом 4 и ночью: Si правило +18,6 б.п. (t 2,0), зеркало +17,9 б.п. (t 2,0); Eu +13,1 и +16,3 б.п.; z-score 60: Si +34,2 и +30,8, Eu +27,7 и +17,9. У CR зеркало, наоборот, отрицательное (−9,7 б.п., t −1,6). То есть на Si и Eu прибыль даёт не сигнал входа, а устройство выхода: широкий стоп и удержание до стопа ловят любое длинное одностороннее движение. Это согласуется с трендом на масштабе недель, но не со «внутридневным трендом» и не с конкретным сигналом.
- Продолжение разрыва и первого часа, импульс, последнее движение: для Si и Eu t по модулю не больше 1,2 ни в одном варианте, а у скальпинга net = −12 ₽, то есть ровно круг.

### 4. Моментум на дневном масштабе

Знак доходности за L дней определяет позицию на H дней (без стопов, сделки не пересекаются, круг 12 ₽, t с кластеризацией по месяцам). Сетка L ∈ {5, 10, 20, 40} × H ∈ {5, 10, 20}:

- Si: 12 из 12 вариантов в плюсе, 8 со значимостью t > 2; Eu: 12 из 12 и 6; CR: 8 из 11 и 4. Проверка перебора по сетке: Si p = 0,001, Eu p = 0,002, CR p = 0,08.
- Лучший вариант (L = 20, H = 5): Si +70 б.п. на сделку (t 3,0, 244 сделки), Eu +56 б.п. (t 3,3), CR +46 б.п. (t 2,5, 139 сделок). Зеркало (против сигнала) симметрично отрицательное: −73, −59, −66 б.п. (t −3,2, −3,5, −3,6), то есть направление имеет значение. Контроль дрейфа (всегда покупка или всегда продажа на H = 5…10 дней): у Si и Eu около нуля (|net| ≤ 8 б.п., |t| ≤ 0,5); у CR покупка даёт −26…−36 б.п. (t −0,9…−1,3), продажа +6…+16 б.п., то есть заметная часть прибыли на CR при сделках в одну сторону — снос цены, но он в 3–7 раз меньше прибыли моментума.
- Устойчивость L = 20, H = 5: без кризисных эпизодов (февраль–апрель 2020 и 2022) Si +38 б.п. (t 2,6), Eu +28 б.п. (t 2,4); со входом на день позже Si +54 (t 2,4), Eu +47 (t 2,8), CR +31 (t 1,5); лет в плюсе: Si 6 из 8 (минус в 2019 и 2021), Eu 7 из 8, CR 5 из 5.
- Независимая проверка (Si и Eu до 21.04.2022, 3,1 года): L = 20 положителен во всех 6 ячейках (Si +50 / +97 / +121 б.п. при H = 5 / 10 / 20, Eu +54 / +2 / +139), но t по каждой ячейке только 0,0–2,0 (24–102 сделки): направление то же, значимости мало.
- Масштаб: около 31 сделки в год, средняя сделка 46–70 б.п. при стандартном отклонении 233–323 б.п.; по порядку величины это 14–22% годовых от номинала при Шарпе около 1,1–1,2 на каждом инструменте, но все три инструмента — один рублёвый фактор, их результаты нельзя складывать как независимые.

### 5. Что из этого следует

1. Внутридневной тренд-следящий вход на минутках остаётся без преимущества после издержек, так как проверка на Si и Eu это подтвердила; контртренд тоже не работает. Идея «живая стратегия не обгоняет круг» — не артефакт малого CR.
2. Единственная устойчивая находка — недельный моментум. Он ловится не минутками с ночным удержанием (там знак сигнала не важен), а прямой ставкой на несколько дней по знаку 20-дневной доходности. Его стоит проверять и реализовывать отдельно от текущей минутной функции: другая частота (раз в день), другой риск (позиция на неделю, маржа и разрывы), другой отчёт.
3. Прежде чем торговать, нужны: фиксированные L и H заранее (выбор L = 20 сделан по результату), перенос на другие рублёвые и нерублёвые фьючерсы, где история уже другая (индексы, нефть, золото; минутки ISS для RI есть с 2019), и накопление реальных наблюдений вперёд.

### Оговорки

- L = 20 и H = 5 — лучший вариант сетки, поэтому его t завышен; поправка White (p 0,001–0,002 на Si и Eu) учитывает только 12 вариантов сетки, но не наш выбор семейства правил.
- Si, Eu и CR после апреля 2022 — один и тот же рублёвый фактор; «подтверждение на трёх инструментах» — это в основном одно подтверждение плюс слабая независимая проверка 2019 – апрель 2022.
- Вход и выход по цене закрытия дня; со входом на день позже эффект уменьшается, но остаётся.
- Окна контрактов — по 3 месяца, поэтому L = 60 недоступен, а L = 40 даёт мало сделок.
- Выходные исключены, ночные цены берутся из минуток вечерней сессии; прямую потерю на маржу и переносе позиции через экспирацию здесь не считали.
- Данные Si/Eu в репозиторий не входят (≈ 50 МБ parquet): загружаются командой `PYTHONPATH=src:. python -m research.fetch Si Eu` (примерно 30–60 минут из-за лимита ISS).
"""


def summary_report(results: list[dict[str, object]]) -> str:
    tables = summary_tables(results)
    t = battery.table
    return "\n".join(
        [
            "# Продолжение поисков: Si и Eu с историей с 2019 года\n",
            "Полные отчёты по каждому инструменту: " + ", ".join(f"`REPORT_longer_{r['name']}.md`" for r in results) + ". Таблицы сеток — в `research/out/longer_*.csv`.\n",
            "## 1. Данные\n",
            "Минутная история квартальных фьючерсов в ISS Московской биржи начинается с контрактов, выпущенных в конце 2018 года: Si и Eu сняты с марта 2019 (контракты 2017–2018 годов ISS не отдаёт). Это 7,5 лет против 4,4 у CR, включая 2019–2021 до появления CR и февраль 2022.\n",
            report.md(tables["passport"].drop(columns=["инструмент"]).assign(**{"тик сравнения": lambda f: f["тик сравнения"].map(lambda v: f"{v:g}")}), 1),
            "\n## 2. Память цены (кратко; подробно — в отчётах по инструментам)\n",
            t(tables["memory"]),
            "\n## 3. Тренд: сетка правил следования за движением (82 варианта)\n",
            report.md(tables["trend"], 2),
            "\nВарианты с удержанием через ночь (слабый намёк на тренд у CR):\n",
            report.md(tables["night"], 1),
            "\n## 4. Контртренд: сетка правил игры против движения (174 варианта)\n",
            report.md(tables["counter"], 2),
            "\n## 5. Закрытие ночного разрыва (порог 20 тиков сравнения, выход в конце дня)\n",
            "Главный кандидат из отчёта по CR: ставка против разрыва вверх. «До 21.04.2022» для Si и Eu — данные, которых отчёты по CR не видели.\n",
            report.md(tables["gaps"], 1),
            "\n## 6. Моментум на дневном масштабе (знак доходности за L дней → позиция на H дней)\n",
            "Сетка L ∈ {5, 10, 20, 40} × H ∈ {5, 10, 20}; в таблице только варианты с ≥ 20 сделок. Подробности, дрейф-контроль, устойчивость и разбивка по годам — в отчётах по инструментам.\n",
            report.md(tables["momentum"], 3),
            "\n## 7. Кандидаты из отчётов по CR и их зеркала\n",
            "net на сделку в базисных пунктах номинала (сравнимо между инструментами), t по дням. Зеркало — то же правило с обратным знаком: если оно не хуже правила, источник прибыли не направление сигнала.\n",
            report.md(tables["candidates"], 1),
            "\n## 8. Выводы\n\n" + CONCLUSIONS,
        ]
    )


def main(names: list[str]) -> None:
    names = names or ["CR", "Si", "Eu"]
    results = []
    for name in names:
        res = run_instrument(name)
        (report.OUT / f"REPORT_longer_{name}.md").write_text(instrument_report(res), encoding="utf-8")
        results.append(res)
    (report.OUT / "REPORT_longer.md").write_text(summary_report(results), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
