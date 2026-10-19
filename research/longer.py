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

from research import battery, core, countertrend, evaluate, instruments, report, trend
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

    out["gaps"] = gap_periods(market, data)
    out["gap_table"], out["gap_years"] = countertrend.gap_table(market, data)
    out["bounce"] = countertrend.bounce_table(market, data)
    return out


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
                if name == "CR" and period == "до 21.04.2022":
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
    index = lambda rows: pd.DataFrame(rows).set_index("")  # noqa: E731
    return {
        "passport": index(passport),
        "memory": index(memory),
        "trend": index(trend_rows),
        "counter": index(counter_rows),
        "night": index(night_rows),
        "gaps": pd.DataFrame(gap_rows).set_index(""),
    }


CONCLUSIONS = "TODO"


def summary_report(results: list[dict[str, object]]) -> str:
    tables = summary_tables(results)
    t = battery.table
    return "\n".join(
        [
            "# Продолжение поисков: Si и Eu с историей с 2019 года\n",
            "Полные отчёты по каждому инструменту: " + ", ".join(f"`REPORT_longer_{r['name']}.md`" for r in results) + ". Таблицы сеток — в `research/out/longer_*.csv`.\n",
            "## 1. Данные\n",
            "Минутная история квартальных фьючерсов в ISS Московской биржи начинается с контрактов, выпущенных в конце 2018 года: Si и Eu сняты с марта 2019 (контракты 2017–2018 годов ISS не отдаёт). Это 7,5 лет против 4,4 у CR, включая 2019–2021 до появления CR и февраль 2022.\n",
            report.md(tables["passport"].drop(columns=["инструмент"]), 1),
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
            "\n## 6. Выводы\n\n" + CONCLUSIONS,
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
