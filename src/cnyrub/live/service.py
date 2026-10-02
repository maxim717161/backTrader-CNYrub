"""Один вызов функции — одна минута одного счёта.

История окна качается по одному московскому дню и в этот вызов заявка на
фьючерс не ставится. Первый ордер на фьючерс — на минуте, которая закрылась
уже после того, как окно собрано. Пропущенные минуты дописываются в канал и
проверяются на стоп, шаг считается только по последней закрытой свече, и на
фьючерс уходит не больше одного рыночного ордера. В бакете остаётся окно
индикаторов и десять сессий сверху, более старые минутки стираются.

Свободные рубли всех окон покупают фонд денежного рынка. Залог на сделку
этой минуты — fill_per_minute лотов — остаётся в рублях и с запасом в половину.
Фонд продаётся только если этих рублей не хватает, чтобы увеличить позицию.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from cnyrub.engine import (
    CLOCK_DAYS,
    SURGE_BARS,
    _FillBook,
    export_book,
    load_book,
    reprice_fill,
    restore_book,
    step_minute,
)
from cnyrub.live.broker import Candle, CashFund, Instrument, choose_front
from cnyrub.live.config import RunRequest
from cnyrub.live.indicators import bar_get, bar_levels
from cnyrub.live.state import StateStore, state_key

MSK = ZoneInfo("Europe/Moscow")
FLATTEN_AT = time(23, 40)
# В истории последняя вечерняя свеча — 23:49. Следующая минута уже другой день.
SESSION_LAST = time(23, 49)
# Сессия деривативов длиннее основной: утро с 06:50 и вечер до 23:50.
SESSION_MINUTES = 18 * 60
HISTORY_WALKS = 90
CATCHUP_DAYS = 3
# Десять сессий сверх окна. Новогодние и майские выходные длятся больше недели,
# а день перед праздником часто короче обычной сессии.
KEEP_EXTRA_SESSIONS = 10
TRADE_KEEP = 30
# Запас сверх залога на минутную сделку: в рублях лежит полтора таких залога.
CASH_MARGIN_BUFFER = 1.5


def history_goal(params) -> int:
    """Сколько минуток нужно, чтобы канал (и пять дней объёма короткого окна) ожил."""
    goal = params.channel
    if params.exit_channel:
        goal = max(goal, params.exit_channel)
    if params.clock_cap is not None:
        goal = max(goal, (CLOCK_DAYS + 1) * SESSION_MINUTES)
    if params.surge_cap is not None:
        goal = max(goal, SURGE_BARS)
    return goal


def bars_to_keep(params) -> int:
    """Сколько последних минуток хранить. Старше этого окно уже не смотрит."""
    return history_goal(params) + KEEP_EXTRA_SESSIONS * SESSION_MINUTES


def _next_weekday(day: date) -> date:
    """Следующий день торгов. Субботу и воскресенье перешагиваем."""
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def following_day(bar_time: datetime, last_day: date) -> date | None:
    """День следующей минуты для движка.

    До 23:49 это дата самой свечи. С вечерней 23:49 следующая минута — уже
    следующий торговый день, поэтому сигнал перед экспирацией не открывает
    сделку утром последнего дня, а сигнал перед trade_from открывает.
    В последний день после 23:40 остаток снимается целиком, раньше — по
    десять контрактов в минуту.
    """
    local = bar_time.astimezone(MSK)
    if local.date() == last_day and local.time() >= FLATTEN_AT:
        return None
    if local.time() >= SESSION_LAST:
        return _next_weekday(local.date())
    return local.date()


def make_order_id(strategy: str, when: datetime) -> str:
    """Один и тот же id на эту минуту: повторный вызов не ставит вторую заявку."""
    stamp = when.astimezone(MSK).strftime("%Y%m%d%H%M")
    return f"{strategy}-{stamp}"[:36]


def make_cash_order_id(strategy: str, when: datetime) -> str:
    """Id заявки на фонд в ту же минуту, отдельно от заявки на фьючерс."""
    stamp = when.astimezone(MSK).strftime("%Y%m%d%H%M")
    return f"{strategy}-c-{stamp}"[:36]


def extra_margin(before_held: int, target: int, margin: float) -> float:
    """Сколько ещё обеспечения нужно, чтобы добрать цель, а не сократить её."""
    if margin <= 0 or abs(target) <= abs(before_held):
        return 0.0
    if before_held != 0 and target * before_held < 0:
        return 0.0
    return (abs(target) - abs(before_held)) * margin


def cash_to_keep(fill_per_minute: int, margin: float) -> float:
    """Рубли, которые не уходят в фонд: залог на минутную сделку и запас."""
    if margin <= 0 or fill_per_minute <= 0:
        return 0.0
    return fill_per_minute * margin * CASH_MARGIN_BUFFER


def minute_adds_margin(before_held: int, signed: int) -> bool:
    """Эта минута увеличивает позицию, а не только сокращает её."""
    return abs(before_held + signed) > abs(before_held)


def fund_lots_to_sell(held: int, free_rub: float, price: float, lot: int, extra: float) -> int:
    """Сколько лотов фонда продать, чтобы свободных рублей хватило на обеспечение."""
    if held <= 0 or extra <= free_rub:
        return 0
    lot_cost = price * max(lot, 1)
    if lot_cost <= 0:
        return 0
    need = math.ceil((extra - free_rub) / lot_cost)
    return max(0, min(held, need))


def fund_lots_to_buy(free_rub: float, price: float, lot: int, keep: float = 0.0) -> int:
    """Сколько лотов фонда купить, оставив залог минутной сделки и запас на комиссию."""
    lot_cost = price * max(lot, 1)
    if lot_cost <= 0 or free_rub <= 0:
        return 0
    spendable = free_rub - max(keep, 0.0)
    reserve = max(10.0, lot_cost * 0.005)
    return max(0, int((spendable - reserve) // lot_cost))


def _sell_fund(
    state: dict[str, object],
    broker,
    request: RunRequest,
    when: datetime,
    extra: float,
) -> tuple[dict[str, object] | None, str | None]:
    """Продать фонд перед заявкой. Второй элемент — причина остановиться."""
    if not request.cash_ticker or extra <= 0:
        return None, None
    try:
        fund = _cash_instrument(state, broker, request.cash_ticker)
        held = int(broker.fund_lots(request.account_id, fund.uid, fund.lot))
        free = float(broker.free_rub(request.account_id))
        price = float(broker.last_price(fund.uid)) if held > 0 else 0.0
    except Exception:
        return None, "фонд не прочитан"
    if held > 0 and price <= 0:
        return None, "нет цены фонда"
    lots = fund_lots_to_sell(held, free, price, fund.lot, extra)
    if lots <= 0:
        return None, None
    order_id = make_cash_order_id(request.strategy, when)
    try:
        report = broker.market_order(request.account_id, fund.uid, -lots, order_id)
    except Exception:
        return None, "фонд не продан"
    if report.executed != lots:
        return None, f"фонд продан не целиком: {report.executed} из {lots}"
    return {
        "id": order_id,
        "ticker": fund.ticker,
        "signed": -lots,
        "executed": report.executed,
        "price": None if report.price is None else float(report.price),
    }, None


def _buy_fund(
    state: dict[str, object],
    broker,
    request: RunRequest,
    when: datetime,
    instrument_uid: str,
    today: date,
) -> dict[str, object] | None:
    """Купить фонд на рубли сверх залога минутной сделки. Ошибка не останавливает фьючерс."""
    if not request.cash_ticker:
        return None
    try:
        free = float(broker.free_rub(request.account_id))
        margin = _margin(state, broker, instrument_uid, today, refresh=False)
        keep = cash_to_keep(request.fill_per_minute, margin)
        if margin <= 0 or free <= keep:
            return None
        fund = _cash_instrument(state, broker, request.cash_ticker)
        price = float(broker.last_price(fund.uid))
        lots = fund_lots_to_buy(free, price, fund.lot, keep)
        if lots <= 0:
            return None
        if not broker.book_has_offers(fund.uid):
            return None
        order_id = make_cash_order_id(request.strategy, when)
        report = broker.market_order(request.account_id, fund.uid, lots, order_id)
    except Exception as exc:
        detail = " ".join(str(exc).split()) or "неизвестная ошибка"
        return {"error": f"фонд не куплен: {detail[:180]}"}
    if report.executed != lots:
        return {"error": f"фонд куплен не целиком: {report.executed} из {lots}"}
    return {
        "id": order_id,
        "ticker": fund.ticker,
        "signed": lots,
        "executed": report.executed,
        "price": None if report.price is None else float(report.price),
    }


def run_minute(request: RunRequest, broker, store: StateStore, now: datetime | None = None) -> dict[str, object]:
    moment = _moscow(now)
    key = state_key(request.strategy, request.account_id)
    state = store.load(key) or _empty_state(request)
    result: dict[str, object] = {
        "strategy": request.strategy,
        "account_id": request.account_id,
        "phase": "idle",
        "halted": state.get("halted"),
        "order": None,
    }
    if state.get("halted") and not request.reconcile:
        result["phase"] = "halted"
        return result

    front, trade_from = _resolve_front(state, broker, moment.date())
    result["secid"] = front.secid
    if _adopt_instrument(state, front, trade_from) == "halt":
        return _halt(state, store, key, result, "контракт сменился, пока позиция ещё открыта", None, request.params)

    book = load_book(state.get("book") if isinstance(state.get("book"), dict) else None, front.secid, None)
    if not state.get("ready"):
        _pull_history_day(state, broker, front, moment.date())
        bars = list(state.get("bars") or [])
        goal = history_goal(request.params)
        history_before = state.get("history_before")
        reached_listing = isinstance(history_before, str) and date.fromisoformat(history_before) <= front.frsttrade
        walked = int(state.get("history_walked") or 0)
        enough = len(bars) >= goal or reached_listing or (walked >= HISTORY_WALKS and len(bars) >= request.params.channel)
        if enough:
            state["ready"] = True
            if bars:
                state["last_bar"] = str(bar_get(bars[-1], "t"))
        elif walked >= HISTORY_WALKS:
            state["book"] = export_book(book)
            return _halt(state, store, key, result, "не хватает минуток, чтобы собрать окно", book, request.params)
        _trim_bars(state, book, request.params)
        result["cash_order"] = _buy_fund(state, broker, request, moment, front.uid, moment.date())
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "history"
        result["bars"] = len(state.get("bars") or [])
        result["goal"] = goal
        result["ready"] = bool(state.get("ready"))
        return result

    broker_lots = int(broker.futures_position(request.account_id, front.uid))
    if request.reconcile:
        _adopt_position(book, broker, request, front, broker_lots, list(state.get("bars") or []))
        state["halted"] = None
        _trim_bars(state, book, request.params)
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "reconciled"
        result["held"] = book.held
        result["halted"] = None
        return result
    if broker_lots != book.held:
        return _halt(
            state,
            store,
            key,
            result,
            f"позиция на счёте {broker_lots}, в книге {book.held}",
            book,
            request.params,
        )

    sync_before = state.get("sync_from")
    bars_before = len(state.get("bars") or [])
    fresh = _fresh_candles(state, broker, front.uid, moment)
    if fresh is None:
        result["cash_order"] = _buy_fund(state, broker, request, moment, front.uid, moment.date())
        _save_trimmed(state, store, key, book, request.params)
        result["phase"] = "catchup"
        result["bars"] = len(state.get("bars") or [])
        result["held"] = book.held
        return result
    if not fresh:
        book_changed = _trim_bars(state, book, request.params)
        result["cash_order"] = _buy_fund(state, broker, request, moment, front.uid, moment.date())
        moved = state.get("sync_from") != sync_before or len(state.get("bars") or []) != bars_before
        if book_changed or moved:
            if book_changed:
                state["book"] = export_book(book)
            store.save(key, state)
        result["phase"] = "idle"
        result["bars"] = len(state.get("bars") or [])
        result["held"] = book.held
        result["target"] = book.target
        return result

    if book.held == 0:
        book.equity = float(broker.equity(request.account_id))

    state["bars"] = _merge_bars(list(state.get("bars") or []), [_bar(candle) for candle in fresh])
    if len(fresh) > 1:
        for candle in fresh[:-1]:
            if book.stop_hit(candle.open, candle.high, candle.low):
                book.target = 0
                book.reason = "stop"
                break

    margin = _margin(state, broker, front.uid, moment.date(), refresh=True)
    if margin <= 0:
        return _halt(state, store, key, result, "биржа не вернула гарантийное обеспечение", book, request.params)

    last = fresh[-1]
    before = export_book(book)
    levels = bar_levels(list(state["bars"]), request.params)
    local = last.time.astimezone(MSK)
    step_minute(
        book,
        len(state["bars"]) - 1,
        opened=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        volume=last.volume,
        day=local.date(),
        next_day=following_day(local, front.lsttrade),
        last_day=front.lsttrade,
        prior_high=levels["prior_high"],
        prior_low=levels["prior_low"],
        prior_vol=levels["prior_vol"],
        prior_range=levels["prior_range"],
        exit_high=levels["exit_high"],
        exit_low=levels["exit_low"],
        clock_vol=levels["clock_vol"],
        entry_ready=levels["entry_ready"],
        clearance=0.0,
        cooldown=0,
        clock_volume=False,
        clock_cap=request.params.clock_cap,
        size_mode=request.params.size_mode,
        stop_mult=request.params.stop_mult,
        loss_bars=request.params.loss_bars,
        risk_fraction=request.params.risk_fraction,
        margin=margin,
        stop_rub=request.params.stop_rub,
        breakout_span=request.params.breakout_span,
        trade_from=trade_from,
        trail=False,
        scale_step=request.params.scale_step,
        scale_floor=request.params.scale_floor,
        scale_back=request.params.scale_back,
        fill_per_minute=request.fill_per_minute,
        drift=levels["drift"],
        path=levels["path"],
        surge_vol=levels["surge_vol"],
        eff_low=request.params.eff_low,
        eff_high=request.params.eff_high,
        surge_cap=request.params.surge_cap,
        leverage=request.params.leverage,
    )
    stamp = _bar(last)["t"]
    signed = int(book.last_signed)
    if signed == 0:
        state["last_bar"] = stamp
        _trim_bars(state, book, request.params)
        result["cash_order"] = _buy_fund(state, broker, request, last.time, front.uid, moment.date())
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "signal" if book.target != book.held else "hold"
        result["held"] = book.held
        result["target"] = book.target
        result["bars"] = len(state["bars"])
        return result

    keep = cash_to_keep(request.fill_per_minute, margin) if minute_adds_margin(int(before["held"]), signed) else 0.0
    cash_order, cash_error = _sell_fund(state, broker, request, last.time, keep)
    if cash_error:
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(state, store, key, result, cash_error, book, request.params)
    result["cash_order"] = cash_order

    order_id = make_order_id(request.strategy, last.time)
    try:
        report = broker.market_order(request.account_id, front.uid, signed, order_id)
    except Exception:
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(state, store, key, result, "заявка не подтверждена", book, request.params)
    if report.executed != abs(signed):
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(
            state,
            store,
            key,
            result,
            f"заявка исполнена не целиком: {report.executed} из {abs(signed)}",
            book,
            request.params,
        )

    price = last.close if report.price is None or report.price <= 0 else float(report.price)
    reprice_fill(book, before, last.close, price)
    if book.held == 0:
        try:
            book.equity = float(broker.equity(request.account_id))
        except Exception:
            pass
    state["last_bar"] = stamp
    _trim_bars(state, book, request.params)
    result["phase"] = "order"
    result["held"] = book.held
    result["target"] = book.target
    result["bars"] = len(state["bars"])
    result["order"] = {
        "id": order_id,
        "signed": signed,
        "executed": report.executed,
        "price": price,
    }
    # Продажу фонда в эту минуту не перекупаем. Свободные рубли сверх залога
    # минутной сделки паркуем после сокращения позиции.
    if cash_order is None:
        parked = _buy_fund(state, broker, request, last.time, front.uid, moment.date())
        if parked is not None:
            result["cash_order"] = parked
    state["book"] = export_book(book)
    store.save(key, state)
    return result


def choose_front_safe(instruments: list[Instrument], today: date) -> tuple[Instrument, date]:
    return choose_front(instruments, today)


def _resolve_front(state: dict[str, object], broker, today: date) -> tuple[Instrument, date]:
    """Взять уже записанный контракт, пока он не истёк. Список фьючерсов тяжёлый."""
    current = state.get("instrument")
    if isinstance(current, dict):
        try:
            lsttrade = date.fromisoformat(str(current["lsttrade"]))
            trade_from = date.fromisoformat(str(current["trade_from"]))
            frsttrade = date.fromisoformat(str(current["frsttrade"]))
        except (KeyError, TypeError, ValueError):
            lsttrade = None
        else:
            uid = str(current.get("uid") or "")
            secid = str(current.get("secid") or "")
            checked = state.get("front_checked") == today.isoformat()
            if uid and secid and checked and today <= lsttrade:
                return (
                    Instrument(secid, uid, str(current.get("figi") or ""), frsttrade, lsttrade, 1000),
                    trade_from,
                )
    front, trade_from = choose_front_safe(broker.cny_futures(), today)
    state["front_checked"] = today.isoformat()
    return front, trade_from


def _margin(state: dict[str, object], broker, uid: str, today: date, *, refresh: bool) -> float:
    """ГО на сегодня. Перед заявкой на фьючерс читаем заново, иначе берём запись."""
    raw = state.get("margin")
    day = today.isoformat()
    if not refresh and isinstance(raw, dict) and raw.get("uid") == uid and raw.get("day") == day:
        value = float(raw.get("value") or 0)
        if value > 0:
            return value
    value = float(broker.margin(uid))
    if value > 0:
        state["margin"] = {"uid": uid, "value": value, "day": day}
    return value


def _cash_instrument(state: dict[str, object], broker, ticker: str) -> CashFund:
    """Uid фонда не меняется. Повторный поиск по тикеру не нужен."""
    raw = state.get("cash_fund")
    if isinstance(raw, dict) and str(raw.get("ticker") or "") == ticker and raw.get("uid"):
        try:
            lot = int(raw.get("lot") or 1)
        except (TypeError, ValueError):
            lot = 1
        if lot >= 1:
            return CashFund(ticker, str(raw["uid"]), lot)
    fund = broker.cash_fund(ticker)
    state["cash_fund"] = {"ticker": fund.ticker, "uid": fund.uid, "lot": fund.lot}
    return fund


def _empty_state(request: RunRequest) -> dict[str, object]:
    return {
        "version": 1,
        "strategy": request.strategy,
        "account_id": request.account_id,
        "instrument": None,
        "bars": [],
        "ready": False,
        "last_bar": None,
        "history_before": None,
        "history_walked": 0,
        "halted": None,
        "book": None,
    }


def _adopt_instrument(state: dict[str, object], front: Instrument, trade_from: date) -> str:
    current = state.get("instrument")
    if isinstance(current, dict) and current.get("uid") == front.uid:
        current["trade_from"] = trade_from.isoformat()
        current["lsttrade"] = front.lsttrade.isoformat()
        return "same"
    book = state.get("book") if isinstance(state.get("book"), dict) else {}
    if int(book.get("held") or 0) != 0:
        return "halt"
    state["instrument"] = {
        "secid": front.secid,
        "uid": front.uid,
        "figi": front.figi,
        "frsttrade": front.frsttrade.isoformat(),
        "lsttrade": front.lsttrade.isoformat(),
        "trade_from": trade_from.isoformat(),
    }
    state["bars"] = []
    state["ready"] = False
    state["last_bar"] = None
    state["history_before"] = None
    state["history_walked"] = 0
    state["sync_from"] = None
    state["book"] = export_book(_FillBook(front.secid, None))
    return "reset"


def _pull_history_day(state: dict[str, object], broker, front: Instrument, today: date) -> None:
    walked = int(state.get("history_walked") or 0)
    if walked >= HISTORY_WALKS:
        return
    raw = state.get("history_before")
    before = date.fromisoformat(raw) if isinstance(raw, str) else today + timedelta(days=1)
    day = before - timedelta(days=1)
    start = datetime.combine(day, time.min, tzinfo=MSK)
    end = datetime.combine(before, time.min, tzinfo=MSK)
    candles = [
        candle
        for candle in broker.candles(front.uid, start, end)
        if start <= candle.time.astimezone(MSK) < end
    ]
    state["bars"] = _merge_bars(list(state.get("bars") or []), [_bar(candle) for candle in candles])
    state["history_before"] = day.isoformat()
    state["history_walked"] = walked + 1


def _fresh_candles(state: dict[str, object], broker, uid: str, now: datetime) -> list[Candle] | None:
    """Новые закрытые свечи после last_bar.

    None значит, что до текущей минуты ещё больше трёх суток. Курсор
    запоминается, поэтому следующий вызов продолжает с того же места,
    а не качает те же дни заново. Шаг по этим свечам — когда курсор
    догнал сейчас.
    """
    bars = list(state.get("bars") or [])
    cursor = _sync_cursor(state, bars, now) - timedelta(minutes=2)
    added = False
    for _ in range(CATCHUP_DAYS):
        if cursor >= now - timedelta(minutes=1):
            break
        end = cursor + timedelta(days=1)
        if end > now:
            end = now
        if end <= cursor:
            break
        downloaded = broker.candles(uid, cursor, end)
        extra = [
            _bar(candle)
            for candle in downloaded
            if cursor <= candle.time.astimezone(MSK) < end
        ]
        before = len(bars)
        bars = _merge_bars(bars, extra)
        added = added or len(bars) != before
        cursor = end
    state["bars"] = bars
    if added or _sync_is_stale(state, cursor, now):
        state["sync_from"] = cursor.isoformat()
    if cursor < now - timedelta(minutes=2):
        return None
    last_raw = state.get("last_bar")
    if not isinstance(last_raw, str):
        return []
    last_dt = datetime.fromisoformat(last_raw)
    pending: list[Candle] = []
    for bar in reversed(bars):
        moment = datetime.fromisoformat(str(bar_get(bar, "t")))
        if moment <= last_dt:
            break
        pending.append(_candle_from_bar(bar, moment))
    pending.reverse()
    return pending


def _sync_is_stale(state: dict[str, object], cursor: datetime, now: datetime) -> bool:
    """Тихая минута не двигает курсор. Иначе файл переписывается каждые шестьдесят секунд."""
    previous = state.get("sync_from")
    if not isinstance(previous, str):
        return True
    if cursor < now - timedelta(minutes=2):
        return True
    return cursor - datetime.fromisoformat(previous) > timedelta(hours=1)


def _sync_cursor(state: dict[str, object], bars: list[object], now: datetime) -> datetime:
    raw = state.get("sync_from")
    if isinstance(raw, str):
        return datetime.fromisoformat(raw)
    if bars:
        return datetime.fromisoformat(str(bar_get(bars[-1], "t")))
    last = state.get("last_bar")
    if isinstance(last, str):
        return datetime.fromisoformat(last)
    return now - timedelta(days=1)


def _candle_from_bar(bar: object, moment: datetime) -> Candle:
    return Candle(
        time=moment,
        open=float(bar_get(bar, "o")),
        high=float(bar_get(bar, "h")),
        low=float(bar_get(bar, "l")),
        close=float(bar_get(bar, "c")),
        volume=float(bar_get(bar, "v")),
    )


def _adopt_position(
    book: _FillBook,
    broker,
    request: RunRequest,
    front: Instrument,
    lots: int,
    bars: list[dict[str, object]],
) -> None:
    """Принять позицию брокера и заново поставить стоп, если своего уже нет."""
    same_side = lots == 0 or book.held == 0 or (book.held > 0) == (lots > 0)
    book.secid = front.secid
    book.held = lots
    book.target = lots
    book.last_signed = 0
    if lots == 0:
        book.avg = 0.0
        book.stop_px = None
        book.stop_dist = None
        book.target = 0
        book.base = 0
        book.entry_i = None
        book.peak = 0
        book.opened = 0
        return
    price = broker.position_price(request.account_id, front.uid)
    if price is not None:
        book.avg = float(price)
    if book.avg and (book.stop_px is None or not same_side):
        _install_stop(book, request, bars)
    if book.base == 0 or not same_side:
        book.base = lots
    if book.entry_i is None or not same_side:
        book.entry_i = max(len(bars) - 1, 0)
    if book.peak < abs(lots):
        book.peak = abs(lots)
    if book.opened < abs(lots):
        book.opened = abs(lots)


def _install_stop(book: _FillBook, request: RunRequest, bars: list[dict[str, object]]) -> None:
    """Стоп от цены позиции: фиксированные рубли или медианы минутного диапазона."""
    params = request.params
    if params.stop_rub:
        book.stop_dist = params.stop_rub / 1000.0
    elif params.stop_mult:
        window = bars[-params.channel :] if params.channel > 0 else bars
        ranges = [float(bar_get(bar, "h")) - float(bar_get(bar, "l")) for bar in window]
        if not ranges:
            return
        ordered = sorted(ranges)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        book.stop_dist = params.stop_mult * median
    else:
        return
    book._sync_stop()


def _trim_bars(state: dict[str, object], book: _FillBook | None, params) -> bool:
    """Отрезать минутки старше окна. Индекс входа сдвигается, возраст сделки тот же."""
    if not state.get("ready"):
        return False
    changed = False
    if book is not None and len(book.trades) > TRADE_KEEP:
        book.trades = book.trades[-TRADE_KEEP:]
        changed = True
    bars = list(state.get("bars") or [])
    extra = len(bars) - bars_to_keep(params)
    if extra <= 0:
        return changed
    state["bars"] = bars[extra:]
    if book is None:
        return True
    if book.entry_i is not None:
        book.entry_i -= extra
    if book.cooldown_until:
        book.cooldown_until -= extra
    return True


def _save_trimmed(state: dict[str, object], store: StateStore, key: str, book: _FillBook, params) -> None:
    if _trim_bars(state, book, params):
        state["book"] = export_book(book)
    store.save(key, state)


def _halt(
    state: dict[str, object],
    store: StateStore,
    key: str,
    result: dict[str, object],
    reason: str,
    book: _FillBook | None,
    params,
) -> dict[str, object]:
    state["halted"] = reason
    _trim_bars(state, book, params)
    if book is not None:
        state["book"] = export_book(book)
    store.save(key, state)
    result["phase"] = "halted"
    result["halted"] = reason
    return result


def _bar(candle: Candle) -> dict[str, object]:
    return {
        "t": candle.time.astimezone(MSK).isoformat(),
        "o": candle.open,
        "h": candle.high,
        "l": candle.low,
        "c": candle.close,
        "v": candle.volume,
    }


def _merge_bars(existing: list[object], extra: list[object]) -> list[object]:
    if not extra:
        return existing
    if existing and all(str(bar_get(bar, "t")) > str(bar_get(existing[-1], "t")) for bar in extra):
        merged = list(existing)
        previous = str(bar_get(merged[-1], "t"))
        for bar in sorted(extra, key=lambda item: str(bar_get(item, "t"))):
            stamp = str(bar_get(bar, "t"))
            if stamp == previous:
                merged[-1] = bar
            else:
                merged.append(bar)
            previous = stamp
        return merged
    by_time = {str(bar_get(bar, "t")): bar for bar in existing}
    for bar in extra:
        by_time[str(bar_get(bar, "t"))] = bar
    return [by_time[key] for key in sorted(by_time)]


def _moscow(now: datetime | None) -> datetime:
    moment = now or datetime.now(MSK)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MSK)
    return moment.astimezone(MSK)
