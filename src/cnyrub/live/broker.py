"""Клиент REST Т-Инвестиций. Песочницы нет: каждый метод бьёт в боевой контур.

База: https://invest-public-api.tbank.ru/rest/
Заголовок Authorization: Bearer <токен>. Токен в сообщения об ошибках не пишется.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from urllib.request import Request, urlopen

from cnyrub.contracts import Contract, front_windows

API = "https://invest-public-api.tbank.ru/rest/"
_TICKER = re.compile(r"^CR[HMUZ][0-9]$")


@dataclass(frozen=True)
class Instrument:
    secid: str
    uid: str
    figi: str
    frsttrade: date
    lsttrade: date
    lot: int


@dataclass(frozen=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class CashFund:
    """Фонд денежного рынка: свободные рубли лежат в нём и дают ставку каждую ночь."""

    ticker: str
    uid: str
    lot: int


@dataclass(frozen=True)
class FillReport:
    order_id: str
    requested: int
    executed: int
    price: float | None


def quotation(value: object) -> float:
    """Цена или сумма вида {units, nano}."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return float(value)
    units = int(value.get("units") or 0)
    nano = int(value.get("nano") or 0)
    return units + nano / 1_000_000_000


def trading_date(value: object) -> date:
    """Календарный день экспирации.

    Полночь UTC оставляем как эту дату: так биржа часто отдаёт lastTradeDate.
    Иначе берём дату в Москве.
    """
    moment = _timestamp(value).astimezone(timezone.utc)
    if moment.hour == 0 and moment.minute == 0 and moment.second == 0:
        return moment.date()
    from zoneinfo import ZoneInfo

    return moment.astimezone(ZoneInfo("Europe/Moscow")).date()


def parse_instrument(row: dict[str, object]) -> Instrument | None:
    """Квартальный фьючерс CNY/RUB, лот 1000, код CR и буква месяца."""
    ticker = str(row.get("ticker") or "")
    if _TICKER.match(ticker) is None:
        return None
    class_code = row.get("classCode")
    if class_code not in (None, "SPBFUT"):
        return None
    try:
        lot = int(row.get("lot") or 0)
    except (TypeError, ValueError):
        return None
    if lot != 1000:
        return None
    asset = str(row.get("basicAsset") or "").upper()
    if "CNY" not in asset or "UCNY" in asset or "MOEX" in asset:
        return None
    uid = str(row.get("uid") or "")
    if not uid:
        return None
    try:
        frsttrade = trading_date(row.get("firstTradeDate"))
        lsttrade = trading_date(row.get("lastTradeDate"))
    except (TypeError, ValueError):
        return None
    if lsttrade < frsttrade:
        return None
    return Instrument(
        secid=ticker,
        uid=uid,
        figi=str(row.get("figi") or ""),
        frsttrade=frsttrade,
        lsttrade=lsttrade,
        lot=lot,
    )


def choose_front(instruments: list[Instrument], today: date) -> tuple[Instrument, date]:
    """Фронтальный контракт и день, с которого в нём можно открывать сделки.

    День начала — следующий после последнего дня предыдущего выпуска.
    У первого выпуска это его первый день торгов.
    """
    paired: list[tuple[Contract, Instrument]] = []
    for item in instruments:
        paired.append(
            (
                Contract(
                    secid=item.secid,
                    shortname=item.secid,
                    assetcode="CNY",
                    frsttrade=item.frsttrade,
                    lsttrade=item.lsttrade,
                    lstdeldate=None,
                    lotsize=item.lot,
                ),
                item,
            )
        )
    windows = [
        window
        for window in front_windows([contract for contract, _item in paired], today)
        if window.contract.lsttrade >= today
    ]
    if not windows:
        raise RuntimeError("нет фронтального контракта CNY/RUB")
    window = windows[-1]
    for contract, instrument in paired:
        if contract.secid == window.secid and contract.lsttrade == window.contract.lsttrade:
            return instrument, window.start
    raise RuntimeError("нет фронтального контракта CNY/RUB")


def parse_candle(row: dict[str, object]) -> Candle | None:
    """Минутная свеча. Незакрытую свечу отбрасываем."""
    if row.get("isComplete") is False:
        return None
    try:
        moment = _timestamp(row.get("time"))
    except (TypeError, ValueError):
        return None
    return Candle(
        time=moment,
        open=quotation(row.get("open")),
        high=quotation(row.get("high")),
        low=quotation(row.get("low")),
        close=quotation(row.get("close")),
        volume=float(row.get("volume") or 0),
    )


def parse_fill(payload: dict[str, object], fallback_id: str) -> FillReport:
    requested = int(payload.get("lotsRequested") or 0)
    executed = int(payload.get("lotsExecuted") or 0)
    price_raw = payload.get("executedOrderPrice")
    if price_raw in (None, {}):
        price_raw = payload.get("initialOrderPrice")
    price = None if price_raw in (None, {}) else quotation(price_raw)
    return FillReport(
        order_id=str(payload.get("orderId") or fallback_id),
        requested=requested,
        executed=executed,
        price=price,
    )


def parse_cash_fund(row: dict[str, object], ticker: str) -> CashFund | None:
    """Биржевой фонд LQDT или TMON с класса TQTF."""
    if str(row.get("ticker") or "").upper() != ticker.upper():
        return None
    kind = str(row.get("instrumentType") or row.get("instrumentKind") or "").lower()
    if kind and "etf" not in kind:
        return None
    class_code = str(row.get("classCode") or "")
    if class_code and class_code != "TQTF":
        return None
    uid = str(row.get("uid") or "")
    if not uid:
        return None
    try:
        lot = int(row.get("lot") or 1)
    except (TypeError, ValueError):
        return None
    if lot < 1:
        return None
    return CashFund(ticker=ticker.upper(), uid=uid, lot=lot)


def margin_rub(payload: dict[str, object]) -> float:
    """Большее из ГО на покупку и на продажу, чтобы хватало в обе стороны."""
    buy = quotation(payload.get("initialMarginOnBuy"))
    sell = quotation(payload.get("initialMarginOnSell"))
    return max(buy, sell)


class TinkoffClient:
    """Боевой REST. transport подменяется в тесте разбора ответов."""

    def __init__(self, token: str, transport=None) -> None:
        self._token = token
        self._transport = transport or _urllib_post

    def cny_futures(self) -> list[Instrument]:
        payload = self._call(
            "InstrumentsService",
            "Futures",
            {"instrumentStatus": "INSTRUMENT_STATUS_ALL"},
        )
        found: list[Instrument] = []
        for row in payload.get("instruments") or []:
            if isinstance(row, dict):
                item = parse_instrument(row)
                if item is not None:
                    found.append(item)
        return found

    def candles(self, uid: str, start: datetime, end: datetime) -> list[Candle]:
        payload = self._call(
            "MarketDataService",
            "GetCandles",
            {
                "instrumentId": uid,
                "from": _stamp(start),
                "to": _stamp(end),
                "interval": "CANDLE_INTERVAL_1_MIN",
                "limit": 2400,
            },
        )
        candles: list[Candle] = []
        for row in payload.get("candles") or []:
            if isinstance(row, dict):
                candle = parse_candle(row)
                if candle is not None:
                    candles.append(candle)
        candles.sort(key=lambda item: item.time)
        return candles

    def margin(self, uid: str) -> float:
        payload = self._call("InstrumentsService", "GetFuturesMargin", {"instrumentId": uid})
        return margin_rub(payload)

    def futures_position(self, account_id: str, uid: str) -> int:
        payload = self._call("OperationsService", "GetPositions", {"accountId": account_id})
        lots = 0
        for row in payload.get("futures") or []:
            if not isinstance(row, dict):
                continue
            if row.get("instrumentUid") == uid or row.get("figi") == uid:
                lots += int(row.get("balance") or 0)
        return lots

    def position_price(self, account_id: str, uid: str) -> float | None:
        payload = self._call("OperationsService", "GetPortfolio", {"accountId": account_id, "currency": "RUB"})
        for row in payload.get("positions") or []:
            if not isinstance(row, dict):
                continue
            if row.get("instrumentUid") != uid and row.get("figi") != uid:
                continue
            price = row.get("averagePositionPrice")
            if isinstance(price, dict):
                return quotation(price)
        return None

    def cash_fund(self, ticker: str) -> CashFund:
        payload = self._call("InstrumentsService", "FindInstrument", {"query": ticker})
        for row in payload.get("instruments") or []:
            if isinstance(row, dict):
                fund = parse_cash_fund(row, ticker)
                if fund is not None:
                    return fund
        raise RuntimeError(f"нет фонда {ticker}")

    def last_price(self, uid: str) -> float:
        payload = self._call("MarketDataService", "GetLastPrices", {"instrumentId": [uid]})
        for row in payload.get("lastPrices") or []:
            if not isinstance(row, dict):
                continue
            if row.get("instrumentUid") in (None, uid):
                return quotation(row.get("price"))
        return 0.0

    def free_rub(self, account_id: str) -> float:
        """Рубли, которые можно забрать: обеспечение фьючерса уже вычтено."""
        payload = self._call("OperationsService", "GetWithdrawLimits", {"accountId": account_id})
        total = 0.0
        for row in payload.get("money") or []:
            if isinstance(row, dict) and str(row.get("currency") or "").lower() == "rub":
                total += quotation(row)
        return total

    def fund_lots(self, account_id: str, uid: str, lot: int) -> int:
        payload = self._call("OperationsService", "GetPositions", {"accountId": account_id})
        shares = 0
        for row in payload.get("securities") or []:
            if not isinstance(row, dict):
                continue
            if row.get("instrumentUid") == uid or row.get("figi") == uid:
                shares += int(row.get("balance") or 0)
        size = lot if lot > 0 else 1
        return shares // size

    def equity(self, account_id: str) -> float:
        payload = self._call("OperationsService", "GetPortfolio", {"accountId": account_id, "currency": "RUB"})
        return quotation(payload.get("totalAmountPortfolio"))

    def market_order(self, account_id: str, uid: str, signed: int, order_id: str) -> FillReport:
        if signed == 0:
            raise ValueError("пустая заявка")
        payload = self._call(
            "OrdersService",
            "PostOrder",
            {
                "quantity": str(abs(signed)),
                "direction": "ORDER_DIRECTION_BUY" if signed > 0 else "ORDER_DIRECTION_SELL",
                "accountId": account_id,
                "orderType": "ORDER_TYPE_MARKET",
                "orderId": order_id,
                "instrumentId": uid,
            },
        )
        return parse_fill(payload, order_id)

    def _call(self, service: str, method: str, body: dict[str, object]) -> dict[str, object]:
        url = f"{API}tinkoff.public.invest.api.contract.v1.{service}/{method}"
        payload = self._transport(url, body, {"Authorization": f"Bearer {self._token}"})
        if not isinstance(payload, dict):
            raise RuntimeError("пустой ответ API")
        if payload.get("code") and payload.get("message") and _looks_like_error(payload):
            raise RuntimeError(str(payload.get("message")))
        return payload


def _looks_like_error(payload: dict[str, object]) -> bool:
    useful = {
        "candles",
        "instruments",
        "futures",
        "positions",
        "orderId",
        "totalAmountPortfolio",
        "lastPrices",
        "money",
    }
    return not any(key in payload for key in useful)


def _stamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, dict):
        seconds = int(value.get("seconds") or 0)
        return datetime.fromtimestamp(seconds, timezone.utc)
    text = str(value).replace("Z", "+00:00")
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _urllib_post(url: str, body: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    data = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))
