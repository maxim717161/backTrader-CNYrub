"""Текст сделки и отправка в ВК без сети."""

from __future__ import annotations

import json

import pytest

from cnyrub.live.vk import error_text, halt_text, notify_vk, notify_status, notify_trade, status_text, trade_text


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


def test_notify_stays_quiet_without_vk_settings():
    result = _fill()
    notify_trade(result, environ={})
    assert "vk" not in result
    notify_trade(result, environ={"VK_TOKEN": "secret"})
    assert result["vk"] == "не настроено"
    quiet = {"strategy": "thirty", "phase": "probe", "order": None}
    notify_status(quiet, environ={})
    assert quiet["vk"] == "не настроено"


def test_notify_posts_the_fill_to_a_user_and_a_chat():
    seen: list[tuple[str, str, str]] = []

    def post(token: str, peer_id: str, text: str) -> None:
        seen.append((token, peer_id, text))

    result = _fill(order={"reason": "scale", "signed": -4, "executed": 4, "price": 11.25}, held=6, target=6)
    notify_trade(
        result,
        environ={"VK_TOKEN": "vk-secret", "VK_USER_ID": "42", "VK_PEER_ID": "2000000001"},
        post=post,
    )
    assert result["vk"] == "отправлено"
    text = trade_text(result)
    assert seen == [("vk-secret", "2000000001", text), ("vk-secret", "42", text)]


def test_notify_keeps_the_fill_when_vk_rejects_it(capsys):
    def post(token: str, peer_id: str, text: str) -> None:
        raise RuntimeError(f"https://api.vk.com/method/messages.send {token}")

    result = _fill()
    notify_trade(result, environ={"VK_TOKEN": "vk-secret", "VK_USER_ID": "42"}, post=post)
    assert result["vk"] == "не отправлено"
    assert result["order"]["signed"] == 10
    logged = capsys.readouterr()
    assert "vk-secret" not in logged.err
    assert "api.vk.com" not in logged.err
    assert "не отправлено" in logged.err


def test_notify_vk_stays_quiet_without_settings():
    result: dict[str, object] = {}
    notify_vk(result, "проба", environ={})
    assert "vk" not in result


def test_post_sends_a_form_and_hides_the_token(monkeypatch):
    from urllib.parse import parse_qs

    from cnyrub.live import vk

    captured: dict[str, object] = {}
    reply = {"response": 7}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(reply).encode("utf-8")

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["form"] = parse_qs(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(vk, "urlopen", urlopen)
    vk._post("vk-secret", "42", "проба")
    assert captured["url"] == "https://api.vk.com/method/messages.send"
    assert captured["auth"] == "Bearer vk-secret"
    assert captured["timeout"] == 10
    assert captured["form"] == {"peer_id": ["42"], "random_id": ["0"], "message": ["проба"], "v": ["5.199"]}
    assert "vk-secret" not in captured["url"]

    reply = {"error": {"error_code": 901, "error_msg": "Can't send messages for users without permission"}}
    with pytest.raises(RuntimeError, match="901") as error:
        vk._post("vk-secret", "42", "x")
    assert "vk-secret" not in str(error.value)

    def refuse(request, timeout):
        raise OSError("https://api.vk.com vk-secret")

    monkeypatch.setattr(vk, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="вк не ответил") as error:
        vk._post("vk-secret", "42", "x")
    assert "vk-secret" not in str(error.value)


def test_halt_text_only_for_a_new_halt():
    halted = {
        "strategy": "sixty",
        "account_id": "2254834906",
        "phase": "halted",
        "halted": "позиция на счёте 0, в книге -3",
        "secid": "CRZ6",
        "held": -3,
        "target": -3,
        "halt_new": True,
    }
    assert halt_text(halted) == (
        "стратегия sixty остановлена\n"
        "счёт 2254834906\n"
        "CRZ6\n"
        "причина: позиция на счёте 0, в книге -3\n"
        "позиция -3, цель -3\n"
        "проверьте счёт и пришлите reconcile: true"
    )
    assert halt_text({**halted, "halt_new": False}) is None
    repeated = dict(halted)
    del repeated["halt_new"]
    assert halt_text(repeated) is None


def test_notify_trade_sends_the_halt_alert_once():
    seen: list[str] = []

    def post(token: str, peer_id: str, text: str) -> None:
        seen.append(text)

    result = {"strategy": "thirty", "phase": "halted", "halted": "заявка не подтверждена", "order": None, "halt_new": True}
    notify_trade(result, environ={"VK_TOKEN": "t", "VK_USER_ID": "42"}, post=post)
    assert result["vk"] == "отправлено"
    assert "заявка не подтверждена" in seen[0]
    quiet = {"strategy": "thirty", "phase": "halted", "halted": "заявка не подтверждена", "order": None}
    notify_trade(quiet, environ={"VK_TOKEN": "t", "VK_USER_ID": "42"}, post=post)
    assert "vk" not in quiet and len(seen) == 1


def test_trade_text_marks_a_partial_fill():
    result = _fill(order={"signed": 3, "executed": 3, "requested": 10})
    assert "исполнено частично: 3 из 10" in trade_text(result)
    assert "частично" not in trade_text(_fill(order={"requested": 10}))


def test_error_text_only_when_the_alert_is_due():
    failed = {"strategy": "thirty", "account_id": "2254834906", "phase": "error", "error": "RuntimeError: API  недоступно", "error_new": True}
    text = error_text(failed)
    assert text.startswith("стратегия thirty: сбой запуска\nсчёт 2254834906\nRuntimeError: API недоступно")
    assert error_text({**failed, "error_new": False}) is None
    seen: list[str] = []
    notify_trade(failed, environ={"VK_TOKEN": "t", "VK_USER_ID": "42"}, post=lambda token, peer, body: seen.append(body))
    assert seen == [text]
