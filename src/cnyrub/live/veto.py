"""Запрет входа от старшего окна на живом счёте.

Правило то же, что «старшее, 2 дня, только плюс» в истории. Каждая стратегия
пишет короткий файл peers/<имя>.json. Чужую книгу целиком не читаем: у длинного
окна там десятки тысяч минуток. Короткое окно не глушит длинное. Своя позиция,
стоп и выход не трогаются — снимается только свежая цель входа против чужого плюса.
"""

from __future__ import annotations

from cnyrub.engine import MULTIPLIER, _FillBook
from cnyrub.live.config import PRESETS
from cnyrub.live.indicators import bar_get

HOLD_DAYS = 2


def peer_key(strategy: str) -> str:
    return f"peers/{strategy}.json"


def held_trading_days(bars: list[object], entry_i: int | None) -> int:
    """Сколько торговых дней прошло после дня входа. День — смена даты в минутках."""
    if entry_i is None or not bars:
        return 0
    start = 0 if entry_i < 0 else min(int(entry_i), len(bars) - 1)
    seen: set[str] = set()
    days = 0
    for bar in bars[start:]:
        day = str(bar_get(bar, "t"))[:10]
        if day in seen:
            continue
        seen.add(day)
        if len(seen) > 1:
            days += 1
    if entry_i < 0:
        # Вход уже выпал из хранимого окна: дней не меньше, чем само это окно, плюс один.
        days += 1
    return days


def cancel_fresh_entry(book: _FillBook) -> None:
    """Снять цель, которую эта минута только что поставила. Исполнения ещё не было."""
    book.target = 0
    book.base = 0
    book.entry = ""
    book.stop_dist = None
    book.stop_px = None
    book.unit = 0.0
    book.best = None
    book.scale_level = 0
    book.scaled = False
    book.reason = ""


def publish_peer(store, strategy: str, channel: int, book: _FillBook | None, bars: list[object]) -> None:
    """Записать позицию, которую видят остальные стратегии. Плоская книга снимает запрет."""
    if book is None or strategy not in PRESETS:
        return
    close = float(bar_get(bars[-1], "c")) if bars else 0.0
    held = int(book.held)
    target = int(book.target)
    same_side = held != 0 and target != 0 and (held > 0) == (target > 0)
    document = {
        "strategy": strategy,
        "channel": int(channel),
        "held": held if same_side else 0,
        "target": target if same_side else 0,
        "held_days": held_trading_days(bars, book.entry_i) if same_side else 0,
        "pnl": (close - float(book.avg)) * held * MULTIPLIER if same_side else 0.0,
    }
    store.save(peer_key(strategy), document)


def veto_source(store, strategy: str, channel: int, book: _FillBook) -> str:
    """Кто запрещает этот свежий вход. Пусто — вход можно открывать."""
    if book.held != 0 or book.target == 0:
        return ""
    side = 1 if book.target > 0 else -1
    blockers: list[str] = []
    for name in PRESETS:
        if name == strategy:
            continue
        peer = _load_peer(store, name)
        if peer is not None and _blocks(peer, int(channel), side):
            blockers.append(name)
    return ", ".join(blockers)


def _load_peer(store, name: str) -> dict[str, object] | None:
    try:
        document = store.load(peer_key(name))
    except Exception:
        return None
    if not isinstance(document, dict) or document.get("strategy") != name:
        return None
    return document


def _blocks(peer: dict[str, object], channel: int, side: int) -> bool:
    """Старшее окно в плюсе двое торговых дней и держит позицию против этого входа."""
    try:
        peer_channel = int(peer.get("channel") or 0)
        held = int(peer.get("held") or 0)
        target = int(peer.get("target") or 0)
        held_days = int(peer.get("held_days") or 0)
        pnl = float(peer.get("pnl") or 0.0)
    except (TypeError, ValueError):
        return False
    if peer_channel <= channel or held == 0 or target == 0 or held_days < HOLD_DAYS or not pnl > 0:
        return False
    if (held > 0) != (target > 0):
        return False
    if side > 0:
        return held < 0
    return held > 0
