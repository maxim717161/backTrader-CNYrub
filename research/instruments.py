"""Инструменты исследования: CR (юань-рубль, 2022+), Si (доллар-рубль) и Eu (евро-рубль) с 2019.

У всех трёх один «рубль на пункт цены»-масштаб: шаг цены стоит 1 ₽ на контракт (CR: 0,001 п. ·
1000, Si/Eu: 1 п. · 1), поэтому комиссия 5 ₽ с каждой стороны и проскальзывание 1 тик дают круг 12 ₽
на каждом. Номинал контракта разный (CR ≈ 12,5 тыс. ₽, Si/Eu ≈ 60–110 тыс. ₽), так что в
относительном выражении издержки на Si/Eu в 4–8 раз меньше: именно поэтому они — честная проверка
того, не съедают ли издержки реальный, но малый эффект.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from research import core, fetch

REL_UNIT = 8e-5


@dataclass(frozen=True)
class Instrument:
    name: str
    title: str
    multiplier: float
    tick: float


INSTRUMENTS = {
    "CR": Instrument("CR", "CNY/RUB (CR)", 1000.0, 0.001),
    "Si": Instrument("Si", "USD/RUB (Si)", 1.0, 1.0),
    "Eu": Instrument("Eu", "EUR/RUB (Eu)", 1.0, 1.0),
}


def comparison_unit(instrument: Instrument, median_price: float) -> float:
    """Один «тик сравнения»: 0,008% цены, не меньше шага (для CR это ровно 0,001)."""
    ticks = max(1, round(median_price * REL_UNIT / instrument.tick))
    return ticks * instrument.tick


def load(name: str, today: date | None = None) -> tuple[core.Market, Instrument]:
    """Загрузить фронтальный ряд и настроить core под инструмент (множитель, шаг, тик сравнения)."""
    instrument = INSTRUMENTS[name]
    if name == "CR":
        core.configure(instrument.multiplier, instrument.tick)
        market = core.load_front()
        core.configure(instrument.multiplier, instrument.tick, comparison_unit(instrument, float(market.bars["close"].median())))
        return market, instrument
    frames = []
    today = today or date.today()
    for window in fetch.windows(name, today):
        path = fetch.path_for(window)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["contract"] = window.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"нет данных {name}: запустите `PYTHONPATH=src:. python -m research.fetch {name}`")
    market = core.assemble(frames)
    core.configure(instrument.multiplier, instrument.tick, comparison_unit(instrument, float(market.bars["close"].median())))
    return market, instrument


def describe(market: core.Market, instrument: Instrument) -> dict[str, object]:
    """Паспорт ряда: период, дни, контракты, номинал и круг в базисных пунктах."""
    bars = market.bars
    price = float(bars["close"].median())
    notional = price * instrument.multiplier
    return {
        "инструмент": instrument.title,
        "с": bars["day"].min().date().isoformat(),
        "по": bars["day"].max().date().isoformat(),
        "дней": int(bars["day"].nunique()),
        "контрактов": int(bars["contract"].nunique()),
        "минуток": len(bars),
        "медианная цена": price,
        "номинал, ₽": notional,
        "круг, ₽": core.round_trip_cost(),
        "круг, б.п.": core.round_trip_cost() / notional * 1e4,
        "тик сравнения": core.UNIT,
    }
