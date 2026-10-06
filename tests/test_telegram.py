"""Текст сделки и отправка в Telegram без сети."""

from __future__ import annotations

import json

import pytest

from cnyrub.live.telegram import notify_trade, trade_text


def _fill(**overrides) -> dict[str, object]:
    order = {
        "id": "short-202609281006",
        "signed": 10,
        "executed": 10,
        "price": 10.8,
        "time": "2026-09-28 10:06",
        "reason": "up",
    }
    order.update(overrides.pop("order", {}))
    result = {
        "strategy": "short",
        "secid": "CRZ6",
        "held": 10,
        "target": 19,
        "order": order,
    }
    result.update(overrides)
    return result


def test_trade_text_names_the_strategy_and_the_reason():
    assert trade_text(_fill()) == (
        "стратегия short\nCRZ6 2026-09-28 10:06\nпокупка 10 по 10.8, пробой вверх\nпозиция 10, цель 19"
    )
    assert trade_text(_fill(strategy="sixty")).startswith("стратегия sixty\n")
    closing = _fill(
        held=0,
        target=0,
        order={"signed": -10, "executed": 10, "price": 9.4, "reason": "stop", "pnl": -6010},
    )
    assert trade_text(closing) == (
        "стратегия short\n"
        "CRZ6 2026-09-28 10:06\n"
        "продажа 10 по 9.4, стоп\n"
        "результат -6 010 руб.\n"
        "позиция 0, цель 0"
    )
    partial = _fill(held=6, target=6, order={"signed": -4, "executed": 4, "price": 11.25, "reason": "scale", "pnl": 3996.5})
    assert "результат +3 996.5 руб." in trade_text(partial)
    assert trade_text({"order": None, "phase": "idle"}) is None
    assert trade_text({"phase": "halted"}) is None


def test_notify_stays_quiet_without_telegram_settings():
    result = _fill()
    notify_trade(result, environ={})
    assert "telegram" not in result
    notify_trade(result, environ={"TELEGRAM_BOT_TOKEN": "123:abc"})
    assert result["telegram"] == "не настроено"


def test_notify_posts_the_fill_once():
    seen: list[tuple[str, str, str]] = []

    def post(token: str, chat_id: str, text: str) -> None:
        seen.append((token, chat_id, text))

    result = _fill(order={"reason": "scale", "signed": -4, "executed": 4, "price": 11.25}, held=6, target=6)
    notify_trade(
        result,
        environ={"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "-1001"},
        post=post,
    )
    assert result["telegram"] == "отправлено"
    assert seen == [
        (
            "123:abc",
            "-1001",
            "стратегия short\nCRZ6 2026-09-28 10:06\nпродажа 4 по 11.25, откат\nпозиция 6, цель 6",
        )
    ]


def test_notify_keeps_the_fill_when_telegram_rejects_it(capsys):
    def post(token: str, chat_id: str, text: str) -> None:
        raise RuntimeError(f"https://api.telegram.org/bot{token}/sendMessage")

    result = _fill()
    notify_trade(result, environ={"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "7"}, post=post)
    assert result["telegram"] == "не отправлено"
    assert result["order"]["signed"] == 10
    logged = capsys.readouterr()
    assert "123:abc" not in logged.err
    assert "api.telegram.org" not in logged.err
    assert "не отправлено" in logged.err


def test_post_sends_the_chat_and_hides_the_token(monkeypatch):
    from cnyrub.live import telegram

    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"ok": True, "result": {}}).encode("utf-8")

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(telegram, "urlopen", urlopen)
    telegram._post("123:abc", "-1001", "short CRZ6")
    assert captured["url"] == "https://api.telegram.org/bot123:abc/sendMessage"
    assert captured["timeout"] == 10
    assert captured["body"] == {
        "chat_id": "-1001",
        "text": "short CRZ6",
        "disable_web_page_preview": True,
    }

    def refuse(request, timeout):
        raise telegram.HTTPError(request.full_url, 400, "Bad Request", hdrs=None, fp=None)

    monkeypatch.setattr(telegram, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="telegram не ответил") as error:
        telegram._post("123:abc", "7", "x")
    assert "123:abc" not in str(error.value)
