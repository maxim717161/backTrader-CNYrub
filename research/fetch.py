"""Загрузка минуток квартальных фьючерсов других инструментов (Si, Eu, RI, ...) по окнам фронта.

Минутная история ISS для квартальных фьючерсов начинается с контрактов, выпущенных в конце
2018 года (до этого свечи уже не отдаются), поэтому окна идут с марта 2019. Окно контракта —
от дня после экспирации предыдущего до экспирации (третий четверг месяца): так же, как у CR
в `data/contracts.json`.

Данные кладутся в `data/other/<актив>/` (в git не входят — слишком велики): пересоздаются
командой `PYTHONPATH=src:. python -m research.fetch Si Eu RI`.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OTHER_DIR = ROOT / "data" / "other"
LETTERS = {3: "H", 6: "M", 9: "U", 12: "Z"}
FIRST_EXPIRY = (2018, 12)
COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Window:
    asset: str
    secid: str
    start: date
    end: date
    expiry: date

    @property
    def name(self) -> str:
        return f"{self.asset}{LETTERS[self.expiry.month]}{self.expiry.year}"


def third_thursday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 14)


def windows(asset: str, today: date) -> list[Window]:
    out: list[Window] = []
    previous = third_thursday(*FIRST_EXPIRY)
    year, month = FIRST_EXPIRY
    while True:
        month += 3
        if month > 12:
            month -= 12
            year += 1
        end = third_thursday(year, month)
        start = previous + timedelta(days=1)
        if start > today:
            break
        out.append(Window(asset, f"{asset}{LETTERS[month]}{year % 10}", start, min(end, today), end))
        previous = end
    return out


def path_for(window: Window, directory: Path = OTHER_DIR) -> Path:
    return directory / window.asset / f"{window.name}.parquet"


def complete(window: Window, directory: Path, today: date) -> bool:
    return window.end < today and path_for(window, directory).exists()


def fetch_window(window: Window, client, directory: Path = OTHER_DIR) -> int:
    frame = client.candles(window.secid, window.start, window.end)
    if frame.empty:
        return 0
    frame = frame[COLUMNS].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].astype("float64")
    frame["volume"] = frame["volume"].astype("int32")
    path = path_for(window, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression="zstd", index=False)
    return len(frame)


def main(argv: list[str]) -> None:
    from cnyrub.iss import IssClient

    assets = argv or ["Si", "Eu"]
    today = date.today()
    client = IssClient(pause=0.08)
    jobs = [w for asset in assets for w in windows(asset, today) if not complete(w, OTHER_DIR, today)]
    print(f"к загрузке окон: {len(jobs)}", flush=True)

    def work(window: Window) -> tuple[str, int]:
        count = fetch_window(window, client)
        print(f"{window.name}: {count} минуток", flush=True)
        return window.name, count

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(work, jobs))


if __name__ == "__main__":
    main(sys.argv[1:])
