"""Обработчик Cloud Functions.

Каждый запуск, и таймер и тестовый вызов, ставит заявки на реальный счёт
Т-Инвестиций. Исключение — поле probe: оно только пишет пробу в ВК.
Токен передаётся полем token в тестовом JSON либо полем
secret_id: тогда он читается из Lockbox, ключ записи token или TOKEN.
В лог и в ответ функции токен не попадает. После каждого запуска
в лог пишется одна строка JSON — тот же ответ: фаза, остановка,
заявка, контракт, число минуток, позиция и цель.

Окружение функции:
  STATE_BUCKET — бакет Object Storage, один JSON на стратегию и счёт
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY — статический ключ бакета
  VK_TOKEN и VK_USER_ID либо VK_PEER_ID — куда писать исполненную заявку,
    остановку стратегии и (раз в сутки) сбой запуска. Если настроек нет,
    сделка просто сохраняется.
  RUN_DEADLINE — секунд до принудительной остановки запуска (по умолчанию 45).
  Поле probe в тестовом JSON шлёт одну пробу в ВК и не ставит заявку.
  В таймер его класть не нужно: пока оно включено, минута не торгует.

Сервисный аккаунт функции должен читать Lockbox. Повторы таймера лучше
выключить: следующая минута подхватит пропуск сама.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable
from datetime import datetime
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from cnyrub.live import trace
from cnyrub.live.broker import TinkoffClient
from cnyrub.live.config import parse_event
from cnyrub.live.service import run_minute
from cnyrub.live.state import StateStore, alert_key, object_store_from_env, state_key
from cnyrub.live.vk import notify_status, notify_trade

_MSK = ZoneInfo("Europe/Moscow")
_METADATA = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
_LOCKBOX = "https://payload.lockbox.api.cloud.yandex.net/lockbox/v1/secrets/{secret_id}/payload"


_DEADLINE = 45.0
_GRACE = 8
_started = 0.0


def _deadline() -> float:
    try:
        value = float(os.environ.get("RUN_DEADLINE") or _DEADLINE)
    except ValueError:
        value = _DEADLINE
    return max(value, 1.0)


def _on_alarm(signum, frame) -> None:
    signal.alarm(_GRACE)
    raise TimeoutError(f"запуск не уложился в {_deadline():g} с, этап: {trace.stage}")


def handle(event: object, context: object = None, **kwargs) -> dict[str, object]:
    """Один запуск с предохранителем: зависший запрос не съедает все 60 секунд функции.

    По истечении RUN_DEADLINE (по умолчанию 45 с) запуск прерывается ошибкой с названием
    этапа. Она попадает в лог и в суточное сообщение ВК, а не обрывается молча.
    """
    global _started
    _started = time.monotonic()
    trace.mark("старт")
    armed = False
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(max(int(_deadline()), 1))
        armed = True
    except (ValueError, AttributeError):
        pass
    try:
        return _handle(event, context, **kwargs)
    finally:
        if armed:
            signal.alarm(0)


def _handle(
    event: object,
    context: object = None,
    *,
    broker_factory: Callable[[str], object] | None = None,
    store: StateStore | None = None,
    now: datetime | None = None,
    secret_reader: Callable[[str], str] | None = None,
    notifier: Callable[[dict[str, object]], None] | None = None,
    status_notifier: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Разобрать событие, достать токен и прогнать одну минуту."""
    request = parse_event(event)
    if request.probe:
        result = _empty_probe(request)
        try:
            if store is None:
                store = object_store_from_env()
            trace.mark("бакет: чтение книги")
            result = _probe_result(request, store)
        except Exception:
            result["state"] = "не прочитано"
        sender = notify_status if status_notifier is None else status_notifier
        try:
            sender(result)
        except Exception:
            result["vk"] = "не отправлено"
        _log_run(result)
        return result
    token = request.token
    try:
        if store is None:
            store = object_store_from_env()
        if token is None:
            trace.mark("Lockbox: чтение токена")
            reader = secret_reader or read_lockbox_token
            token = reader(request.secret_id or "")
        if broker_factory is None:
            broker = TinkoffClient(token)
        else:
            broker = broker_factory(token)
        trace.mark("минута стратегии")
        result = run_minute(request, broker, store, now=now)
    except Exception as exc:
        _report_failure(request, exc, token, store, now, notifier)
        raise
    if request.reconcile:
        _reset_alert(request, store)
    sender = notify_trade if notifier is None else notifier
    trace.mark("ВК: сообщение")
    try:
        sender(result)
    except Exception:
        result["vk"] = "не отправлено"
    result["sec"] = round(time.monotonic() - _started, 1)
    _log_run(result)
    return result


_ALERTED: dict[str, str] = {}


def _today(now: datetime | None) -> str:
    moment = datetime.now(_MSK) if now is None else now.astimezone(_MSK)
    return moment.date().isoformat()


def _alert_due(request, store: StateStore | None, today: str) -> bool:
    """Одно сообщение о сбое в сутки. Нет доступа к бакету — помним в этом процессе."""
    key = alert_key(request.strategy, request.account_id)
    if store is not None:
        try:
            saved = store.load(key)
            if isinstance(saved, dict) and saved.get("date") == today:
                return False
            store.save(key, {"date": today})
            _ALERTED[key] = today
            return True
        except Exception:
            pass
    if _ALERTED.get(key) == today:
        return False
    _ALERTED[key] = today
    return True


def _reset_alert(request, store: StateStore | None) -> None:
    """Перезапуск снимает отметку: следующий сбой снова придёт в ВК."""
    key = alert_key(request.strategy, request.account_id)
    _ALERTED.pop(key, None)
    if store is None:
        return
    try:
        store.save(key, {"date": None})
    except Exception:
        pass


def _report_failure(request, exc: Exception, token, store, now, notifier) -> None:
    detail = f"{type(exc).__name__}: {exc}"
    if token:
        detail = detail.replace(token, "***")
    entry: dict[str, object] = {
        "strategy": request.strategy,
        "account_id": request.account_id,
        "phase": "error",
        "error": detail,
    }
    try:
        if _alert_due(request, store, _today(now)):
            entry["error_new"] = True
            sender = notify_trade if notifier is None else notifier
            try:
                sender(entry)
            except Exception:
                entry["vk"] = "не отправлено"
    finally:
        _log_run(entry)


def _empty_probe(request) -> dict[str, object]:
    return {
        "strategy": request.strategy,
        "account_id": request.account_id,
        "phase": "probe",
        "halted": None,
        "order": None,
        "secid": None,
        "bars": None,
        "held": None,
        "target": None,
    }


def _probe_result(request, store: StateStore | None) -> dict[str, object]:
    """Снимок сохранённой книги. Биржу не спрашивает и заявку не ставит."""
    state: dict[str, object] = {}
    if store is not None:
        loaded = store.load(state_key(request.strategy, request.account_id))
        if isinstance(loaded, dict):
            state = loaded
    instrument = state.get("instrument") if isinstance(state.get("instrument"), dict) else {}
    book = state.get("book") if isinstance(state.get("book"), dict) else {}
    bars = state.get("bars")
    held = book.get("held")
    target = book.get("target")
    return {
        "strategy": request.strategy,
        "account_id": request.account_id,
        "phase": "probe",
        "halted": state.get("halted"),
        "order": None,
        "secid": instrument.get("secid"),
        "bars": len(bars) if isinstance(bars, list) else None,
        "held": held if isinstance(held, int) and not isinstance(held, bool) else None,
        "target": target if isinstance(target, int) and not isinstance(target, bool) else None,
    }


def _log_run(result: dict[str, object]) -> None:
    """Одна строка JSON в stdout. Облако забирает её в лог функции."""
    print(json.dumps(result, ensure_ascii=False), flush=True)


def read_lockbox_token(secret_id: str, get: Callable[[str, dict[str, str]], dict] | None = None) -> str:
    """IAM из метаданных виртуалки, затем поле token секрета Lockbox."""
    fetch = get or _urllib_get
    issued = fetch(_METADATA, {"Metadata-Flavor": "Google"})
    access = issued.get("access_token")
    if not access:
        raise RuntimeError("метаданные не выдали IAM-токен")
    payload = fetch(_LOCKBOX.format(secret_id=secret_id), {"Authorization": f"Bearer {access}"})
    for entry in payload.get("entries") or []:
        if entry.get("key") in {"token", "TOKEN"} and entry.get("textValue"):
            return str(entry["textValue"])
    raise RuntimeError("в секрете Lockbox нет ключа token")


def _urllib_get(url: str, headers: dict[str, str]) -> dict:
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=10) as response:
        document = json.loads(response.read().decode("utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError("пустой ответ Lockbox")
    return document
