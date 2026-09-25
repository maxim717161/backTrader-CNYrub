"""График квартальных контрактов CNY/RUB по отдельности.

Запуск из корня репозитория:

    python plot.py

Пишет cny_front.png. Каждый контракт — своя линия, соседние выпуски не
соединяются. Месяц до экспирации предыдущего виден как пересечение по времени.
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

DEFAULT_DATA = Path("data/bars")
DEFAULT_OUTPUT = Path("cny_front.png")


def load_series(path: Path) -> pd.DataFrame:
    columns = ["datetime", "high", "low", "close", "volume", "secid"]
    if path.is_dir():
        frames = [pd.read_parquet(item, columns=columns) for item in sorted(path.glob("*.parquet"))]
        if not frames:
            return pd.DataFrame(columns=columns)
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.read_parquet(path, columns=columns)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values(["datetime", "secid"])


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    day = frame["datetime"].dt.floor("D")
    return frame.groupby(day, sort=True).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )


def _volume_label(value: float, _position: object) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.0f} млн"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f} тыс"
    return f"{value:.0f}"


def _ordered_secids(frame: pd.DataFrame) -> list[str]:
    last = frame.groupby("secid", sort=False)["datetime"].max().sort_values()
    return [str(secid) for secid in last.index]


def plot_series(frame: pd.DataFrame, output: Path) -> Path:
    """Сохранить график: своя линия цены и объёма у каждого контракта."""
    if frame.empty:
        raise ValueError("В ряде нет свечей")
    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 6.5),
        height_ratios=(3, 1),
        layout="constrained",
    )
    colors = plt.get_cmap("tab20")
    secids = _ordered_secids(frame)
    for index, secid in enumerate(secids):
        part = frame.loc[frame["secid"] == secid]
        daily = _daily(part)
        color = colors(index % 20)
        ax_price.plot(daily.index, daily["close"], color=color, linewidth=1.05, label=secid)
        ax_volume.plot(daily.index, daily["volume"], color=color, linewidth=0.8, alpha=0.85)
    first = pd.Timestamp(frame["datetime"].iloc[0]).strftime("%d.%m.%Y")
    last = pd.Timestamp(frame["datetime"].iloc[-1]).strftime("%d.%m.%Y")
    ax_price.set_title(f"CNY/RUB, квартальные контракты по отдельности\n{first} — {last}")
    ax_price.set_ylabel("Рубли за 1 юань")
    ax_price.grid(True, axis="y", linewidth=0.4, alpha=0.7)
    ax_price.legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
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
    parser = argparse.ArgumentParser(description="График квартальных контрактов CNY/RUB")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Каталог или parquet с минутками")
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
