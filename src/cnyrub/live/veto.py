"""Запрет свежего входа на живом счёте. Роли те же, что у python backtest.py.

Исследование сюда не импортируется. Каждая стратегия пишет peers/<имя>.json.
Перед свежим входом читаются только те соседи, которых слушает роль этого счёта.
Окно 30: более длинное окно, уже два торговых дня в плюсе. Окно 45: только
1 760, два дня в плюсе. Окно 60: более длинное окно, плюс считается сразу,
моментум не голосует; если старшие смотрят в обе стороны, запрета нет.
Короткое окно: 1 760, 2 160, 12 420 и моментум, все уже день в плюсе и в одну
сторону. Окно 2 160: моментум, два дня в плюсе. Окно 1 760, длинное окно и
моментум чужие файлы не читают. Своя позиция, стоп и выход не трогаются.
"""

from __future__ import annotations

from dataclasses import dataclass

from cnyrub.engine import MULTIPLIER, _FillBook
from cnyrub.live.config import MOMENTUM, PRESETS
from cnyrub.live.indicators import bar_get

# Порядок имён в ответе. Моментум в пресеты окон не входит, но файл свой пишет.
_ORDER = tuple(PRESETS) + (MOMENTUM,)


@dataclass(frozen=True)
class _Role:
    mode: str
    quorum: int = 1
    hold_days: int = 2
    slower_only: bool = True
    momentum_votes: bool = False
    exclusive: bool = True
    voters: frozenset[str] | None = None


# Те же роли, что kept_split в исследовании. Числа продублированы нарочно:
# облачная функция исследование не импортирует.
_ROLES: dict[str, _Role] = {
    "thirty": _Role("legacy", hold_days=2, exclusive=False),
    "fortyfive": _Role(
        "quorum",
        quorum=1,
        hold_days=2,
        slower_only=False,
        voters=frozenset({"m1760"}),
    ),
    "sixty": _Role("quorum", quorum=1, hold_days=0, momentum_votes=False),
    "short": _Role("quorum", quorum=4, hold_days=1, momentum_votes=True),
    "m2160": _Role(
        "quorum",
        quorum=1,
        hold_days=2,
        slower_only=False,
        momentum_votes=True,
        voters=frozenset({MOMENTUM}),
    ),
    "m1760": _Role("none"),
    "long": _Role("none"),
    MOMENTUM: _Role("none"),
}


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
    """Записать позицию в peers/. Плоская книга снимает запрет."""
    if book is None or strategy not in _ROLES:
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
    role = _ROLES.get(strategy)
    if role is None or role.mode == "none":
        return ""
    side = 1 if book.target > 0 else -1
    votes: list[tuple[str, int]] = []
    for name in _ORDER:
        if name == strategy:
            continue
        peer = _load_peer(store, name)
        if peer is None:
            continue
        peer_side = _vote(peer, role, channel)
        if peer_side:
            votes.append((name, peer_side))
    if role.mode == "quorum" and role.exclusive and _both_sides(votes):
        return ""
    opposite = [name for name, peer_side in votes if peer_side != side]
    if role.mode == "quorum" and len(opposite) < role.quorum:
        return ""
    return ", ".join(opposite)


def _both_sides(votes: list[tuple[str, int]]) -> bool:
    return any(peer_side > 0 for _, peer_side in votes) and any(peer_side < 0 for _, peer_side in votes)


def _load_peer(store, name: str) -> dict[str, object] | None:
    try:
        document = store.load(peer_key(name))
    except Exception:
        return None
    if not isinstance(document, dict) or document.get("strategy") != name:
        return None
    return document


def _vote(peer: dict[str, object], role: _Role, channel: int) -> int:
    """Сторона чужой позиции, если она входит в роль. Иначе 0."""
    name = str(peer.get("strategy") or "")
    if role.voters is not None and name not in role.voters:
        return 0
    try:
        peer_channel = int(peer.get("channel") or 0)
        held = int(peer.get("held") or 0)
        target = int(peer.get("target") or 0)
        held_days = int(peer.get("held_days") or 0)
        pnl = float(peer.get("pnl") or 0.0)
    except (TypeError, ValueError):
        return 0
    if name == MOMENTUM:
        if not role.momentum_votes:
            return 0
    elif role.slower_only and peer_channel <= channel:
        return 0
    if held == 0 or target == 0 or (held > 0) != (target > 0):
        return 0
    if held_days < role.hold_days or not pnl > 0:
        return 0
    return 1 if held > 0 else -1
