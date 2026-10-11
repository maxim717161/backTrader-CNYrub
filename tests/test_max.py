"""Текст сделки и отправка в MAX без сети."""

from __future__ import annotations

import json

import pytest

from cnyrub.live.maxbot import notify_max, notify_status, notify_trade, status_text, trade_text


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


def test_status_text_reports_the_saved_book():
    text = status_text(
        {
            "strategy": "thirty",
            "account_id": "2254834906",
            "phase": "probe",
            "halted": None,
            "order": None,
            "secid": "CRZ6",
            "bars": 12231,
            "held": 0,
            "target": 0,
        }
    )
    assert text == (
        "проба\n"
        "стратегия thirty\n"
        "счёт 2254834906\n"
        "фаза probe\n"
        "CRZ6 минуток 12231\n"
        "позиция 0, цель 0"
    )
    halted = status_text({"strategy": "thirty", "phase": "probe", "halted": "позиция на счёте 0, в книге -3"})
    assert "остановка позиция на счёте 0, в книге -3" in halted
    unread = status_text({"strategy": "thirty", "phase": "probe", "state": "не прочитано"})
    assert "книга не прочитана" in unread


def test_notify_stays_quiet_without_max_settings():
    result = _fill()
    notify_trade(result, environ={})
    assert "max" not in result
    notify_trade(result, environ={"MAX_BOT_TOKEN": "secret"})
    assert result["max"] == "не настроено"
    quiet = {"strategy": "thirty", "phase": "probe", "order": None}
    notify_status(quiet, environ={})
    assert quiet["max"] == "не настроено"


def test_notify_posts_the_fill_to_a_user():
    seen: list[tuple[str, str, str, str]] = []

    def post(token: str, kind: str, value: str, text: str) -> None:
        seen.append((token, kind, value, text))

    result = _fill(order={"reason": "scale", "signed": -4, "executed": 4, "price": 11.25}, held=6, target=6)
    notify_trade(
        result,
        environ={"MAX_BOT_TOKEN": "max-secret", "MAX_USER_ID": "42"},
        post=post,
    )
    assert result["max"] == "отправлено"
    assert seen == [("max-secret", "user_id", "42", trade_text(result))]


def test_notify_keeps_the_fill_when_max_rejects_it(capsys):
    def post(token: str, kind: str, value: str, text: str) -> None:
        raise RuntimeError(f"https://platform-api2.max.ru/messages {token}")

    result = _fill()
    notify_trade(result, environ={"MAX_BOT_TOKEN": "max-secret", "MAX_CHAT_ID": "100"}, post=post)
    assert result["max"] == "не отправлено"
    assert result["order"]["signed"] == 10
    logged = capsys.readouterr()
    assert "max-secret" not in logged.err
    assert "platform-api2.max.ru" not in logged.err
    assert "не отправлено" in logged.err


def test_notify_max_stays_quiet_without_settings():
    result: dict[str, object] = {}
    notify_max(result, "проба", environ={})
    assert "max" not in result


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
    with pytest.raises(RuntimeError, match="max не ответил") as error:
        maxbot._post("max-secret", "user_id", "42", "x")
    assert "max-secret" not in str(error.value)
