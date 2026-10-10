"""Сообщение в Telegram после исполненной заявки на фьючерс.

Токен бота и чат читаются из окружения функции:
TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID. Если обоих нет, сообщение
не отправляется. Ошибка Telegram не отменяет уже сохранённую сделку
и не попадает в ответ функции вместе с токеном.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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
    """Отправить текст сделки. Без настроек ничего не делает."""
    text = trade_text(result)
    if text is None:
        return
    _send(result, text, environ=environ, post=post)


def notify_status(
    result: dict[str, object],
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[[str, str, str], None] | None = None,
) -> None:
    """Отправить пробу. Заявку не описывает и сделку не меняет."""
    _send(result, status_text(result), environ=environ, post=post)
    if "telegram" not in result:
        result["telegram"] = "не настроено"


def _send(
    result: dict[str, object],
    text: str,
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[[str, str, str], None] | None = None,
) -> None:
    env = os.environ if environ is None else environ
    token = str(env.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = str(env.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token and not chat_id:
        return
    if not token or not chat_id:
        result["telegram"] = "не настроено"
        return
    send = post or _post
    try:
        send(token, chat_id, text)
    except Exception as exc:
        detail = " ".join(str(exc).split())
        if not detail or token in detail or "api.telegram.org" in detail:
            detail = "не отправлено"
        print(f"telegram: {detail[:180]}", file=sys.stderr)
        result["telegram"] = "не отправлено"
        return
    result["telegram"] = "отправлено"


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


def _post(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_description(raw) or "telegram не ответил") from None
    except Exception:
        raise RuntimeError("telegram не ответил") from None
    if not isinstance(payload, dict) or not payload.get("ok"):
        description = ""
        if isinstance(payload, dict):
            description = str(payload.get("description") or "")
        raise RuntimeError(_description(description) or "telegram не принял сообщение")


def _description(raw: str) -> str:
    text = raw.strip()
    try:
        payload = json.loads(text) if text.startswith("{") else None
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("description"):
        text = str(payload["description"])
    return " ".join(text.split())[:180]
