"""График фронтального ряда CNY/RUB.

Запуск из корня репозитория:

    python plot.py

Пишет cny_front.png: дневное закрытие склейки и объём.
Стыки контрактов отмечены вертикальными линиями — цена там не подгонялась.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

DEFAULT_DATA = Path("data/continuous/cny_front_1m.parquet")
DEFAULT_OUTPUT = Path("cny_front.png")


def load_series(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["datetime", "high", "low", "close", "volume", "secid"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values("datetime")


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    day = frame["datetime"].dt.floor("D")
    return frame.groupby(day, sort=True).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        secid=("secid", "last"),
    )


def _volume_label(value: float, _position: object) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.0f} млн"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f} тыс"
    return f"{value:.0f}"


def plot_series(frame: pd.DataFrame, output: Path) -> Path:
    """Сохранить график дневного закрытия и объёма."""
    if frame.empty:
        raise ValueError("В ряде нет свечей")
    daily = _daily(frame)
    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 6.5),
        height_ratios=(3, 1),
        layout="constrained",
    )
    ax_price.fill_between(daily.index, daily["low"], daily["high"], color="#9bb6c9", linewidth=0)
    ax_price.plot(daily.index, daily["close"], color="#1f4b73", linewidth=1.15, label="Закрытие дня")
    rolls = daily["secid"].ne(daily["secid"].shift())
    rolls.iloc[0] = False
    labeled = False
    for stamp in daily.index[rolls]:
        ax_price.axvline(
            stamp,
            color="#b0b0b0",
            linewidth=0.7,
            label="Стык контрактов" if not labeled else None,
        )
        labeled = True
    first = pd.Timestamp(frame["datetime"].iloc[0]).strftime("%d.%m.%Y")
    last = pd.Timestamp(frame["datetime"].iloc[-1]).strftime("%d.%m.%Y")
    ax_price.set_title(f"CNY/RUB, фронтальный квартальный фьючерс\n{first} — {last}, без поправки на спред")
    ax_price.set_ylabel("Рубли за 1 юань")
    ax_price.grid(True, axis="y", linewidth=0.4, alpha=0.7)
    ax_price.legend(loc="upper left", frameon=False)
    ax_volume.bar(daily.index, daily["volume"], width=1.0, color="#8aa0b4")
    ax_volume.set_ylabel("Объём")
    ax_volume.yaxis.set_major_formatter(FuncFormatter(_volume_label))
    ax_volume.grid(True, axis="y", linewidth=0.4, alpha=0.7)
    ax_volume.xaxis.set_major_locator(mdates.YearLocator())
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    if "--show" not in sys.argv:
        plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="График фронтального ряда CNY/RUB")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Parquet со склейкой")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Куда сохранить PNG")
    parser.add_argument("--show", action="store_true", help="Открыть окно с графиком")
    args = parser.parse_args(argv)
    if not args.data.exists():
        print(f"Ошибка: нет файла {args.data}", file=sys.stderr)
        return 1
    output = plot_series(load_series(args.data), args.output)
    print(f"Записано {output}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
