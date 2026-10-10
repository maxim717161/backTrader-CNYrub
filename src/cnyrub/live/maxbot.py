"""Тот же текст сделки и пробы в мессенджер MAX.

Токен и адресат читаются из окружения функции:
MAX_BOT_TOKEN и MAX_CHAT_ID (чат или канал) либо MAX_USER_ID (личный диалог).
Если настроек нет, сообщение не отправляется. Сбой MAX не отменяет сделку
и не отменяет отправку в Telegram.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from collections.abc import Callable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_API = "https://platform-api2.max.ru/messages"


def notify_max(
    result: dict[str, object],
    text: str,
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[[str, str, str, str], None] | None = None,
) -> None:
    """Отправить уже собранный текст. Пустой текст не шлёт."""
    if not text.strip():
        return
    env = os.environ if environ is None else environ
    token = str(env.get("MAX_BOT_TOKEN") or "").strip()
    chat_id = str(env.get("MAX_CHAT_ID") or "").strip()
    user_id = str(env.get("MAX_USER_ID") or "").strip()
    if not token and not chat_id and not user_id:
        return
    targets = [("chat_id", chat_id)] if chat_id else []
    if user_id:
        targets.append(("user_id", user_id))
    if not token or not targets:
        result["max"] = "не настроено"
        return
    send = post or _post
    try:
        for kind, value in targets:
            send(token, kind, value, text)
    except Exception as exc:
        detail = " ".join(str(exc).split())
        if not detail or token in detail or "platform-api2.max.ru" in detail:
            detail = "не отправлено"
        print(f"max: {detail[:180]}", file=sys.stderr)
        result["max"] = "не отправлено"
        return
    result["max"] = "отправлено"


def _post(token: str, kind: str, value: str, text: str) -> None:
    query = urllib.parse.urlencode({kind: value, "disable_link_preview": "true"})
    url = f"{_API}?{query}"
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Authorization": token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_description(raw) or "max не ответил") from None
    except Exception:
        raise RuntimeError("max не ответил") from None
    if not isinstance(payload, dict) or "message" not in payload:
        description = ""
        if isinstance(payload, dict):
            description = str(payload.get("message") or payload.get("description") or "")
        raise RuntimeError(_description(description) or "max не принял сообщение")


def _description(raw: str) -> str:
    text = raw.strip()
    try:
        payload = json.loads(text) if text.startswith("{") else None
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("description")
        if isinstance(message, str) and message.strip():
            text = message
    return " ".join(text.split())[:180]
