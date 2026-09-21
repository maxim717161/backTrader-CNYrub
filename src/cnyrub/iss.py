"""Клиент публичного ISS Московской биржи."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date

import pandas as pd

from cnyrub.bars import BARS_COLUMNS, prepare_bars

CANDLES_URL = (
    "https://iss.moex.com/iss/engines/futures/markets/forts/boards/RFUD/securities/{secid}/candles.json"
)
SECURITY_URL = "https://iss.moex.com/iss/securities/{secid}.json"
PAGE_SIZE = 500
USER_AGENT = "cnyrub/0.1 (front-month CNY/RUB minute history)"


def candles_request(secid: str, start: date, end: date, offset: int) -> tuple[str, dict[str, object]]:
    url = CANDLES_URL.format(secid=secid)
    params: dict[str, object] = {
        "interval": 1,
        "from": start.isoformat(),
        "till": end.isoformat(),
        "start": offset,
        "iss.meta": "off",
    }
    return url, params


def collect_pages(fetch_page: Callable[[int], list[list[object]]]) -> list[list[object]]:
    """Снять страницы ISS, пока не кончится короткая страница или пустой ответ."""
    offset = 0
    rows: list[list[object]] = []
    while True:
        chunk = fetch_page(offset)
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        offset += len(chunk)
    return rows


class _Pacer:
    """Пауза между началами запросов, общая для всех потоков."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            if delay < 0:
                delay = 0.0
            self._next_at = max(self._next_at, now) + self.interval
        if delay:
            time.sleep(delay)


class IssClient:
    def __init__(self, pause: float = 0.2, timeout: float = 60.0, retries: int = 5) -> None:
        self.pause = pause
        self.timeout = timeout
        self.retries = retries
        self._pacer = _Pacer(pause)

    def description(self, secid: str) -> dict[str, str | None] | None:
        payload = self._get_json(SECURITY_URL.format(secid=secid), {"iss.meta": "off"})
        rows = (payload or {}).get("description", {}).get("data") or []
        if not rows:
            return None
        return {row[0]: row[2] for row in rows}

    def candles(self, secid: str, start: date, end: date) -> pd.DataFrame:
        columns: list[str] | None = None
        pages = 0

        def fetch_page(offset: int) -> list[list[object]]:
            nonlocal columns, pages
            url, params = candles_request(secid, start, end, offset)
            payload = self._get_json(url, params)
            table = (payload or {}).get("candles") or {}
            if columns is None:
                columns = list(table.get("columns") or [])
            data = table.get("data") or []
            pages += 1
            if pages == 1 or pages % 10 == 0:
                print(f"  {secid}: страница {pages}, смещение {offset}", flush=True)
            return data

        raw = collect_pages(fetch_page)
        print(f"  {secid}: получено {len(raw)} строк", flush=True)
        if not raw or not columns:
            return prepare_bars(pd.DataFrame(columns=BARS_COLUMNS), secid, start, end)
        frame = pd.DataFrame(raw, columns=columns)
        return prepare_bars(frame, secid, start, end)

    def _get_json(self, url: str, params: dict[str, object]) -> dict[str, object] | None:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            self._pacer.wait()
            request = urllib.request.Request(
                full,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                last_error = error
                if error.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                last_error = error
            time.sleep(self.pause * (2**attempt))
        raise RuntimeError(f"ISS не ответил: {full}") from last_error
