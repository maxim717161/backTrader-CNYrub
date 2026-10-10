"""Обработчик Cloud Functions.

Каждый запуск, и таймер и тестовый вызов, ставит заявки на реальный счёт
Т-Инвестиций. Исключение — поле probe: оно только пишет пробу в Telegram.
Токен передаётся полем token в тестовом JSON либо полем
secret_id: тогда он читается из Lockbox, ключ записи token или TOKEN.
В лог и в ответ функции токен не попадает. После каждого запуска
в лог пишется одна строка JSON — тот же ответ: фаза, остановка,
заявка, контракт, число минуток, позиция и цель.

Окружение функции:
  STATE_BUCKET — бакет Object Storage, один JSON на стратегию и счёт
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY — статический ключ бакета
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID — куда писать исполненную заявку
    на фьючерс. Если обоих нет, сделка просто сохраняется.
  Поле probe в тестовом JSON шлёт в этот чат одну пробу и не ставит заявку.
  В таймер его класть не нужно: пока оно включено, минута не торгует.

Сервисный аккаунт функции должен читать Lockbox. Повторы таймера лучше
выключить: следующая минута подхватит пропуск сама.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from urllib.request import Request, urlopen

from cnyrub.live.broker import TinkoffClient
from cnyrub.live.config import parse_event
from cnyrub.live.service import run_minute
from cnyrub.live.state import StateStore, object_store_from_env, state_key
from cnyrub.live.telegram import notify_status, notify_trade

_METADATA = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
_LOCKBOX = "https://payload.lockbox.api.cloud.yandex.net/lockbox/v1/secrets/{secret_id}/payload"


def handle(
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
    if store is None:
        store = object_store_from_env()
    if request.probe:
        result = _probe_result(request, store)
        sender = notify_status if status_notifier is None else status_notifier
        try:
            sender(result)
        except Exception:
            result["telegram"] = "не отправлено"
        _log_run(result)
        return result
    token = request.token
    if token is None:
        reader = secret_reader or read_lockbox_token
        secret_id = request.secret_id or ""
        token = reader(secret_id)
    if broker_factory is None:
        broker = TinkoffClient(token)
    else:
        broker = broker_factory(token)
    result = run_minute(request, broker, store, now=now)
    sender = notify_trade if notifier is None else notifier
    try:
        sender(result)
    except Exception:
        result["telegram"] = "не отправлено"
    _log_run(result)
    return result


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
