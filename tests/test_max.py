"""Отправка того же текста в MAX без сети."""

from __future__ import annotations

import json

from cnyrub.live.maxbot import notify_max
from cnyrub.live.telegram import notify_trade, trade_text


def test_notify_max_stays_quiet_without_settings():
    result: dict[str, object] = {}
    notify_max(result, "проба", environ={})
    assert "max" not in result
    notify_max(result, "проба", environ={"MAX_BOT_TOKEN": "secret"})
    assert result["max"] == "не настроено"


def test_notify_max_posts_to_a_user_and_keeps_going_after_telegram(capsys, monkeypatch):
    seen: list[tuple[str, str, str, str]] = []

    def post(token: str, kind: str, value: str, text: str) -> None:
        seen.append((token, kind, value, text))

    monkeypatch.setattr("cnyrub.live.maxbot._post", post)

    def refuse(token: str, chat_id: str, text: str) -> None:
        raise RuntimeError(f"bot {token}")

    result = {
        "strategy": "thirty",
        "secid": "CRZ6",
        "held": 0,
        "target": 0,
        "order": {
            "signed": -3,
            "executed": 3,
            "price": 12.8,
            "time": "2026-10-09 12:10",
            "reason": "stop",
            "pnl": -90,
        },
    }
    notify_trade(
        result,
        environ={
            "TELEGRAM_BOT_TOKEN": "tg-secret",
            "TELEGRAM_CHAT_ID": "7",
            "MAX_BOT_TOKEN": "max-secret",
            "MAX_USER_ID": "42",
        },
        post=refuse,
    )
    assert result["telegram"] == "не отправлено"
    assert result["max"] == "отправлено"
    assert seen == [("max-secret", "user_id", "42", trade_text(result))]
    logged = capsys.readouterr().err
    assert "tg-secret" not in logged
    assert "max-secret" not in logged


def test_post_sends_the_chat_and_hides_the_token(monkeypatch):
    from cnyrub.live import maxbot

    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"body": {}}}).encode("utf-8")

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(maxbot, "urlopen", urlopen)
    maxbot._post("max-secret", "chat_id", "100", "проба")
    assert captured["url"] == "https://platform-api2.max.ru/messages?chat_id=100&disable_link_preview=true"
    assert captured["auth"] == "max-secret"
    assert captured["timeout"] == 10
    assert captured["body"] == {"text": "проба"}

    def refuse(request, timeout):
        raise maxbot.HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(maxbot, "urlopen", refuse)
    try:
        maxbot._post("max-secret", "user_id", "42", "x")
    except RuntimeError as error:
        assert "max-secret" not in str(error)
        assert "max не ответил" in str(error)
    else:
        raise AssertionError("отказ MAX должен всплыть")
