"""Сборка отчёта: общие таблицы и итоги по семействам правил (общий код для обеих гипотез)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research import battery, core, evaluate
from research.evaluate import GridResult
from research.families import Spec

OUT = Path(__file__).resolve().parent


COUNT_COLUMNS = ("сделок", "вариантов", "net>0", "gross>0", "t>2", "в плюсе", "год", "удержание", "дней", "контрактов", "минуток", "выбрано")


def md(frame: pd.DataFrame, digits: int = 1, index: bool = True) -> str:
    counts = {i for i, c in enumerate(frame.columns) if any(word in str(c) for word in COUNT_COLUMNS)}

    def cell(value, column: int = -1) -> str:
        if column in counts and isinstance(value, (int, float, np.integer, np.floating)) and value == value:
            return f"{int(value)}"
        if isinstance(value, (float, np.floating)):
            if value != value:
                return "—"
            return f"{value:.{digits}f}"
        return str(value)

    columns = ([frame.index.name or ""] if index else []) + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for label, row in frame.iterrows():
        cells = ([str(label)] if index else []) + [cell(v, i) for i, v in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def family_summary(result: GridResult, specs: list[Spec]) -> pd.DataFrame:
    """По семействам: сколько вариантов, медианные gross/net, доля вариантов в плюсе, лучший по обучению."""
    rows = []
    family_of = {spec.label(): spec.family for spec in specs}
    table = result.table.copy()
    table["семейство"] = [family_of[label] for label in table.index]
    for family, part in table.groupby("семейство", sort=False):
        eligible = part[part["обучение: сделок"] >= 30]
        best = eligible["обучение: t"].idxmax() if len(eligible) else None
        rows.append(
            {
                "семейство": family,
                "вариантов": len(part),
                "медиана gross": part["всё: gross"].median(),
                "медиана net": part["всё: net"].median(),
                "net>0 всего": int((part["всё: net"] > 0).sum()),
                "лучший по обучению: net обуч.": part.loc[best, "обучение: net"] if best else float("nan"),
                "его net на проверке": part.loc[best, "проверка: net"] if best else float("nan"),
                "его t на проверке": part.loc[best, "проверка: t"] if best else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("семейство")


def method_section() -> str:
    return f"""## Данные и метод

- Ряд: минутки фронтального контракта по окнам из `data/contracts.json` (CRM2 … CRZ6), только будни: выходные торги разрежены и сделали бы статистики шумными. 2022-04-21 … 2026-09-21.
- Все статистики считаются внутри дня; ночной разрыв изучается отдельно. Минутки внутри дня зависимы, поэтому доверительные интервалы строятся бутстрепом по дням (дневной масштаб — по контрактам), а t-статистики сделок — с кластеризацией по дням.
- Издержки: комиссия 5 ₽ с каждой стороны + проскальзывание 1 тик (0,001 п. = 1 ₽ на контракт) с каждой стороны: **{core.round_trip_cost():.0f} ₽ за круг на контракт**. Сделка открывается по закрытию минуты, следующей за сигналом (так ведёт себя живая функция).
- Результат сделки — рубли на 1 контракт после издержек (`net`) и до них (`gross`). Сделки не накладываются друг на друга, на счёте не больше одной позиции.
- Чтобы не обмануть себя перебором: период делится по {evaluate.SPLIT.date()} на обучение и проверку (выбор — только по обучению); для сетки считается проверка перебора White (случайная смена знаков дневного результата); показывается, сколько вариантов из сетки в плюсе на обеих половинах.
"""


def vr_frame(vr: pd.DataFrame) -> pd.DataFrame:
    frame = vr.copy()
    frame.index = frame["шаг"] + " → " + frame["горизонт"]
    return frame.drop(columns=["шаг", "горизонт"])


def battery_section(results: dict[str, object], focus: str) -> str:
    t = battery.table
    return f"""## 1. Есть ли в минутках память

{focus}

### Автокорреляция доходностей непересекающихся окон (внутри дня)

Отрицательная — возврат, положительная — продолжение. Лаг 1 на коротких окнах загрязнён эффектом «последняя сделка прошла по биду, следующая по аску» (bid-ask bounce): он живёт только на соседних окнах и затухает к лагу 2.

{t(results['autocorr'])}

### Доля соседних движений в одну сторону

0,5 — нет памяти; меньше — движения чередуются.

{md(results['persistence'], 3, index=False)}

### Отношение дисперсий (VR)

VR = Var(за q шагов) / (q · Var(за шаг)). 1 — случайное блуждание, больше 1 — тренд, меньше 1 — возврат. Второй столбец вычитает шум последней сделки (оценённый по ковариации соседних шагов): именно он показывает, остаётся ли возврат в «настоящей» цене.

{t(vr_frame(results['vr']), index_name='шаг → горизонт')}

Показатель Хёрста по внутридневным окнам 5–160 минут: **{results['hurst']:.3f}** (0,5 — блуждание, меньше — возврат, больше — тренд).

### Дневной масштаб (закрытие к закрытию внутри контракта, интервалы по контрактам)

{t(results['daily_acf'])}

{t(results['daily_vr'])}

### Ночной разрыв: продолжается или закрывается

Наклон: на сколько рублей двигается цена после разрыва в 1 ₽ (1 — полное продолжение, −1 — полное закрытие).

{md(results['gaps'], 3, index=False)}

### Ход после пробоя канала (событийное исследование)

Средний ход цены в сторону пробоя за h минут после входа, ₽ на контракт, и t-статистика. Для сравнения: круг стоит {results['cost']:.0f} ₽.

{t(results['events'])}

### Устойчивость по годам

{t(results['years'].set_index('год'), index_name='год')}

### Спокойные и оживлённые дни

{t(results['activity'].set_index('дни'))}
"""


def grid_section(title: str, intro: str, result: GridResult, specs: list[Spec], selected: dict[str, object], top: int = 8) -> str:
    cols = ["всё: сделок", "всё: gross", "всё: net", "всё: t", "обучение: net", "обучение: t", "проверка: net", "проверка: t"]
    best = evaluate.grid_table(result, "обучение: t", top)[cols]
    worst = evaluate.grid_table(result, "обучение: t").tail(3)[cols]
    years = selected["best_years"]
    return f"""## {title}

{intro}

Вариантов в сетке: **{selected['variants']}**.

- В плюсе после издержек на всём периоде: **{selected['net_positive']}** из {selected['variants']}; до издержек: **{selected['gross_positive']}**.
- В плюсе на обучении: {selected['is_positive']}, со значимостью t>2: **{selected['is_significant']}**.
- В плюсе на проверке: {selected['oos_positive']}, со значимостью t>2: **{selected['oos_significant']}**.
- Проверка перебора (White), p-значение лучшего варианта: обучение **{selected['p_in_sample']:.2f}**, весь период **{selected['p_all']:.2f}** (меньше 0,05 — лучший вариант вряд ли случаен).

### Итоги по семействам

{md(family_summary(result, specs), 1)}

### Лучшие по обучению (net и gross — ₽ на контракт за сделку)

{md(best, 1)}

### Худшие по обучению

{md(worst, 1)}

### Лучший по обучению вариант по годам

`{selected['best_in_sample']}`

{md(years, 1) if len(years) else '—'}
"""


def detail_section(spec: Spec, info: dict[str, object]) -> str:
    mirror = info["mirror"]
    return f"""### Разрез лучшего по обучению варианта

`{spec.label()}`; медианная длительность сделки {info['bars_median']:.0f} мин.

{md(info['sides'], 1)}

Как заканчиваются сделки:

{md(info['reasons'], 0)}

Зеркало — то же правило с обратным знаком, `{info['mirror_label']}`: net на сделку **{mirror['net']:.1f} ₽**, t = {mirror['t']:.1f}, сделок {mirror['n']}.
"""


def save(name: str, text: str, result: GridResult | None = None) -> None:
    (OUT / f"REPORT_{name}.md").write_text(text, encoding="utf-8")
    if result is not None:
        out = OUT / "out"
        out.mkdir(exist_ok=True)
        result.table.round(3).to_csv(out / f"grid_{name}.csv", encoding="utf-8")
