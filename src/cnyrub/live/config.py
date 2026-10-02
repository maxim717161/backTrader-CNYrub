"""Параметры запуска: таймер Яндекса или прямой JSON тестового вызова.

Каждый запуск торгует реальный счёт Т-Инвестиций. Токен приходит в тестовом
JSON либо как идентификатор секрета Lockbox в таймере. В ответ функции токен
не попадает.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, replace

FILL_PER_MINUTE_LIMIT = 10


@dataclass(frozen=True)
class StrategyParams:
    """Правила одного окна. Залог и размер счёта сюда не входят: их отдаёт биржа."""

    channel: int
    exit_channel: int
    stop_mult: float | None
    clock_cap: float | None
    size_mode: str
    loss_bars: int | None
    risk_fraction: float | None
    stop_rub: float | None
    breakout_span: float | None
    scale_step: float | None
    scale_back: float | None
    scale_floor: float
    eff_low: float | None = None
    eff_high: float | None = None
    surge_cap: float | None = None
    leverage: float | None = None


# Те же числа, что у WINDOWS в исследовании. Короткое окно без уменьшения
# на откате. Длинное — стоп 22 медианы и уменьшение до половины на 100 медианах.
# Окно 30 минут — прямота 0,15–0,5, плечо 4 и объём тише трёх медиан за 300 минут.
# Окно 45 минут — стоп 8, плечо 4, без фильтра прямоты и без потолка громкости.
# Окно 60 минут — те же правила, плечо 5.
PRESETS: dict[str, StrategyParams] = {
    "short": StrategyParams(
        channel=525,
        exit_channel=525,
        stop_mult=None,
        clock_cap=5.0,
        size_mode="flat",
        loss_bars=1450,
        risk_fraction=0.10,
        stop_rub=285.0,
        breakout_span=None,
        scale_step=None,
        scale_back=None,
        scale_floor=0.5,
    ),
    "long": StrategyParams(
        channel=12_420,
        exit_channel=0,
        stop_mult=22.0,
        clock_cap=None,
        size_mode="span",
        loss_bars=None,
        risk_fraction=None,
        stop_rub=None,
        breakout_span=12.0,
        scale_step=100.0,
        scale_back=50.0,
        scale_floor=0.5,
    ),
    "thirty": StrategyParams(
        channel=30,
        exit_channel=0,
        stop_mult=8.0,
        clock_cap=5.0,
        size_mode="flat",
        loss_bars=None,
        risk_fraction=None,
        stop_rub=None,
        breakout_span=None,
        scale_step=None,
        scale_back=None,
        scale_floor=0.5,
        eff_low=0.15,
        eff_high=0.5,
        surge_cap=3.0,
        leverage=4.0,
    ),
    "fortyfive": StrategyParams(
        channel=45,
        exit_channel=0,
        stop_mult=8.0,
        clock_cap=5.0,
        size_mode="flat",
        loss_bars=None,
        risk_fraction=None,
        stop_rub=None,
        breakout_span=None,
        scale_step=None,
        scale_back=None,
        scale_floor=0.5,
        leverage=4.0,
    ),
    "sixty": StrategyParams(
        channel=60,
        exit_channel=0,
        stop_mult=8.0,
        clock_cap=5.0,
        size_mode="flat",
        loss_bars=None,
        risk_fraction=None,
        stop_rub=None,
        breakout_span=None,
        scale_step=None,
        scale_back=None,
        scale_floor=0.5,
        leverage=5.0,
    ),
}

_OPTIONAL_FLOATS = (
    "stop_mult",
    "clock_cap",
    "risk_fraction",
    "stop_rub",
    "breakout_span",
    "scale_step",
    "scale_back",
    "scale_floor",
    "eff_low",
    "eff_high",
    "surge_cap",
    "leverage",
)
_OPTIONAL_INTS = ("channel", "exit_channel", "loss_bars")


@dataclass(frozen=True)
class RunRequest:
    strategy: str
    account_id: str
    token: str | None
    secret_id: str | None
    fill_per_minute: int
    reconcile: bool
    params: StrategyParams
    cash_ticker: str | None


def preset(strategy: str) -> StrategyParams:
    if strategy not in PRESETS:
        raise ValueError("strategy должен быть short, long, thirty, fortyfive или sixty")
    return PRESETS[strategy]


def parse_event(event: object) -> RunRequest:
    """Принять JSON параметров, конверт таймера или конверт HTTPS-вызова."""
    data = _payload(event)
    strategy = str(data.get("strategy") or "")
    params = preset(strategy)
    account_id = str(data.get("account_id") or "").strip()
    if not account_id:
        raise ValueError("нужен account_id")
    token = _text(data.get("token"))
    secret_id = _text(data.get("secret_id"))
    if token is None and secret_id is None:
        raise ValueError("нужен token или secret_id")
    params = _apply_overrides(params, data)
    fill = data.get("fill_per_minute", FILL_PER_MINUTE_LIMIT)
    fill_per_minute = int(fill)
    if not 1 <= fill_per_minute <= FILL_PER_MINUTE_LIMIT:
        raise ValueError("fill_per_minute от 1 до 10")
    return RunRequest(
        strategy=strategy,
        account_id=account_id,
        token=token,
        secret_id=secret_id,
        fill_per_minute=fill_per_minute,
        reconcile=_flag(data.get("reconcile")),
        params=params,
        cash_ticker=_cash_ticker(data.get("cash_ticker", "LQDT")),
    )


def _apply_overrides(params: StrategyParams, data: dict[str, object]) -> StrategyParams:
    changes: dict[str, object] = {}
    for name in _OPTIONAL_INTS:
        if name not in data:
            continue
        raw = data[name]
        changes[name] = None if raw is None else int(raw)
    for name in _OPTIONAL_FLOATS:
        if name not in data:
            continue
        raw = data[name]
        changes[name] = None if raw is None else float(raw)
    if "size_mode" in data and data["size_mode"] is not None:
        changes["size_mode"] = str(data["size_mode"])
    if "loss_bars" in changes and changes["loss_bars"] == 0:
        changes["loss_bars"] = None
    if not changes:
        return params
    return replace(params, **changes)


_EMPTY_BODY = (
    "тело запроса пустое: в тесте выберите шаблон «Без шаблона» и вставьте JSON "
    "с полями strategy, account_id и token или secret_id"
)
_NOT_JSON = "тело запроса не JSON"
_NOT_OBJECT = "событие должно быть JSON-объектом"
_TIMER_PAYLOAD = "payload таймера должен быть JSON-объектом"


def _payload(event: object) -> dict[str, object]:
    loaded = _json_object(event)
    if _is_http_event(loaded):
        loaded = _json_object(_http_body(loaded))
    return _unwrap_timer(loaded)


def _json_object(event: object, *, empty: str = _EMPTY_BODY) -> dict[str, object]:
    if isinstance(event, (str, bytes)):
        text = _text_of(event)
        if not text:
            raise ValueError(empty)
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(_NOT_JSON) from exc
    elif isinstance(event, dict):
        loaded = event
    else:
        raise ValueError(_NOT_OBJECT)
    if not isinstance(loaded, dict):
        raise ValueError(_NOT_OBJECT)
    return loaded


def _text_of(value: str | bytes) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(_NOT_JSON) from exc
    else:
        text = value
    return text.lstrip("\ufeff").strip()


def _is_http_event(data: dict[str, object]) -> bool:
    if "httpMethod" in data or "isBase64Encoded" in data:
        return True
    return "body" in data and "headers" in data


def _http_body(data: dict[str, object]) -> object:
    body = data.get("body")
    if isinstance(body, str) and data.get("isBase64Encoded"):
        if not body.strip():
            raise ValueError(_EMPTY_BODY)
        try:
            body = base64.b64decode(body, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(_NOT_JSON) from exc
    if body is None or (isinstance(body, (str, bytes)) and not _text_of(body)):
        raise ValueError(_EMPTY_BODY)
    return body


def _unwrap_timer(loaded: dict[str, object]) -> dict[str, object]:
    messages = loaded.get("messages")
    if not isinstance(messages, list) or not messages:
        return loaded
    first = messages[0]
    details = first.get("details") if isinstance(first, dict) else None
    if not isinstance(details, dict):
        raise ValueError(_TIMER_PAYLOAD)
    payload = details.get("payload")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (str, bytes)):
        return _json_object(payload, empty=_TIMER_PAYLOAD)
    raise ValueError(_TIMER_PAYLOAD)


def _cash_ticker(value: object) -> str | None:
    """Фонд денежного рынка на свободные рубли. Пустое значение выключает его."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "NONE", "OFF", "0"}:
        return None
    if text not in {"LQDT", "TMON"}:
        raise ValueError("cash_ticker должен быть LQDT или TMON")
    return text


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
