"""Сообщение в ВКонтакте после исполненной заявки и пробный статус.

Токен и адресат читаются из окружения функции:
VK_TOKEN — ключ доступа сообщества с правом на сообщения,
VK_USER_ID — личный диалог (пользователь должен сам написать сообществу),
VK_PEER_ID — любой peer_id, например беседа (2000000000 + номер беседы).
Если настроек нет, сообщение не отправляется. Сбой ВК сделку не отменяет
и не попадает в ответ функции вместе с токеном.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from collections.abc import Callable, Mapping
from urllib.request import Request, urlopen

_API = "https://api.vk.com/method/messages.send"
_VERSION = "5.199"
_LIMIT = 4000
_REASONS = {
    "up": "пробой вверх",
    "down": "пробой вниз",
    "add": "добор",
    "scale_back": "возврат",
    "scale": "откат",
    "stop": "стоп",
    "channel": "канал",
    "time": "время",
    "expiry": "экспирация",
}


def trade_text(result: Mapping[str, object]) -> str | None:
    """Текст сделки. Пусто, если за эту минуту фьючерс не исполнялся."""
    order = result.get("order")
    if not isinstance(order, dict):
        return None
    signed = order.get("signed")
    if isinstance(signed, bool) or not isinstance(signed, int) or signed == 0:
        return None
    executed = order.get("executed")
    lots = abs(signed if not isinstance(executed, int) else executed)
    side = "покупка" if signed > 0 else "продажа"
    strategy = str(result.get("strategy") or "").strip()
    secid = str(result.get("secid") or "").strip()
    when = order.get("time")
    when_text = when if isinstance(when, str) else ""
    lines: list[str] = []
    if strategy:
        lines.append(f"стратегия {strategy}")
    place = " ".join(part for part in (secid, when_text) if part)
    if place:
        lines.append(place)
    deal = f"{side} {lots} по {_price(order.get('price'))}"
    reason = _REASONS.get(str(order.get("reason") or ""))
    if reason:
        deal = f"{deal}, {reason}"
    lines.append(deal)
    requested = order.get("requested")
    if isinstance(requested, int) and not isinstance(requested, bool) and lots < requested:
        lines.append(f"исполнено частично: {lots} из {requested}, остаток дозаберёт следующая минута")
    pnl = order.get("pnl")
    if isinstance(pnl, (int, float)) and not isinstance(pnl, bool):
        lines.append(f"результат {_money(float(pnl))}")
    held = result.get("held")
    target = result.get("target")
    if isinstance(held, int) and not isinstance(held, bool):
        if isinstance(target, int) and not isinstance(target, bool):
            lines.append(f"позиция {held}, цель {target}")
        else:
            lines.append(f"позиция {held}")
    return "\n".join(lines)


def halt_text(result: Mapping[str, object]) -> str | None:
    """Текст остановки. Пусто, если эта минута стратегию не останавливала."""
    if result.get("halt_new") is not True:
        return None
    strategy = str(result.get("strategy") or "").strip()
    lines = [f"стратегия {strategy} остановлена" if strategy else "стратегия остановлена"]
    account = str(result.get("account_id") or "").strip()
    if account:
        lines.append(f"счёт {account}")
    secid = str(result.get("secid") or "").strip()
    if secid:
        lines.append(secid)
    reason = str(result.get("halted") or "").strip()
    if reason:
        lines.append(f"причина: {reason}")
    held = result.get("held")
    target = result.get("target")
    if isinstance(held, int) and not isinstance(held, bool):
        if isinstance(target, int) and not isinstance(target, bool):
            lines.append(f"позиция {held}, цель {target}")
        else:
            lines.append(f"позиция {held}")
    lines.append("проверьте счёт и пришлите reconcile: true")
    return "\n".join(lines)


def status_text(result: Mapping[str, object]) -> str:
    """Короткая проба: стратегия, фаза, контракт, минутки и позиция."""
    lines = ["проба"]
    strategy = str(result.get("strategy") or "").strip()
    account = str(result.get("account_id") or "").strip()
    if strategy:
        lines.append(f"стратегия {strategy}")
    if account:
        lines.append(f"счёт {account}")
    phase = str(result.get("phase") or "").strip()
    if phase:
        lines.append(f"фаза {phase}")
    if result.get("state") == "не прочитано":
        lines.append("книга не прочитана")
    halted = result.get("halted")
    if isinstance(halted, str) and halted.strip():
        lines.append(f"остановка {halted.strip()}")
    secid = str(result.get("secid") or "").strip()
    bars = result.get("bars")
    place = secid
    if isinstance(bars, int) and not isinstance(bars, bool):
        place = f"{place} минуток {bars}".strip()
    if place:
        lines.append(place)
    held = result.get("held")
    target = result.get("target")
    held_ok = isinstance(held, int) and not isinstance(held, bool)
    target_ok = isinstance(target, int) and not isinstance(target, bool)
    if held_ok and target_ok:
        lines.append(f"позиция {held}, цель {target}")
    elif held_ok:
        lines.append(f"позиция {held}")
    return "\n".join(lines)


def notify_trade(
    result: dict[str, object],
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[[str, str, str], None] | None = None,
) -> None:
    """Отправить текст сделки или остановки. Без настроек ничего не делает."""
    text = trade_text(result) or halt_text(result)
    if text is None:
        return
    notify_vk(result, text, environ=environ, post=post)


def notify_status(
    result: dict[str, object],
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[[str, str, str], None] | None = None,
) -> None:
    """Отправить пробу. Заявку не описывает и сделку не меняет."""
    notify_vk(result, status_text(result), environ=environ, post=post)
    if "vk" not in result:
        result["vk"] = "не настроено"


def notify_vk(
    result: dict[str, object],
    text: str,
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[[str, str, str], None] | None = None,
) -> None:
    """Отправить уже собранный текст. Пустой текст не шлёт."""
    if not text.strip():
        return
    env = os.environ if environ is None else environ
    token = str(env.get("VK_TOKEN") or "").strip()
    peers: list[str] = []
    for name in ("VK_PEER_ID", "VK_USER_ID"):
        value = str(env.get(name) or "").strip()
        if value and value not in peers:
            peers.append(value)
    if not token and not peers:
        return
    if not token or not peers:
        result["vk"] = "не настроено"
        return
    send = post or _post
    try:
        for peer in peers:
            send(token, peer, text)
    except Exception as exc:
        detail = " ".join(str(exc).split())
        if not detail or token in detail or "api.vk.com" in detail:
            detail = "не отправлено"
        print(f"vk: {detail[:180]}", file=sys.stderr)
        result["vk"] = "не отправлено"
        return
    result["vk"] = "отправлено"


def _money(value: float) -> str:
    """Рубли со знаком. Дробная часть остаётся, пока она не нулевая."""
    amount = round(value, 2)
    sign = "-" if amount < 0 else "+" if amount > 0 else ""
    whole, frac = f"{abs(amount):.2f}".split(".")
    groups: list[str] = []
    while whole:
        groups.append(whole[-3:])
        whole = whole[:-3]
    body = " ".join(reversed(groups)) or "0"
    if frac != "00":
        body = f"{body}.{frac.rstrip('0')}"
    return f"{sign}{body} руб."


def _price(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _post(token: str, peer_id: str, text: str) -> None:
    form = urllib.parse.urlencode(
        {
            "peer_id": peer_id,
            "random_id": 0,
            "message": text[:_LIMIT],
            "v": _VERSION,
        }
    ).encode("utf-8")
    request = Request(
        _API,
        data=form,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        raise RuntimeError("вк не ответил") from None
    if isinstance(payload, dict) and "response" in payload:
        return
    message = ""
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error = payload["error"]
        message = f"{error.get('error_code', '')} {error.get('error_msg', '')}".strip()
    raise RuntimeError(_description(message) or "вк не принял сообщение")


def _description(raw: str) -> str:
    return " ".join(raw.split())[:180]
