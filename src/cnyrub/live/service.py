"""Один вызов функции — одна минута одного счёта.

История окна качается по одному московскому дню и в этот вызов заявка не
ставится. Первый ордер — на минуте, которая закрылась уже после того, как
окно собрано. Пропущенные минуты дописываются в канал и проверяются на стоп,
но шаг считается только по последней закрытой свече, и на биржу уходит
не больше одного рыночного ордера.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from cnyrub.engine import CLOCK_DAYS, _FillBook, export_book, load_book, reprice_fill, restore_book, step_minute
from cnyrub.live.broker import Candle, Instrument, choose_front
from cnyrub.live.config import RunRequest
from cnyrub.live.indicators import bar_levels
from cnyrub.live.state import StateStore, state_key

MSK = ZoneInfo("Europe/Moscow")
FLATTEN_AT = time(23, 40)
# Сессия деривативов длиннее основной: утро с 06:50 и вечер до 23:50.
SESSION_MINUTES = 18 * 60
HISTORY_WALKS = 90
CATCHUP_DAYS = 3


def history_goal(params) -> int:
    """Сколько минуток нужно, чтобы канал (и пять дней объёма короткого окна) ожил."""
    goal = params.channel
    if params.exit_channel:
        goal = max(goal, params.exit_channel)
    if params.clock_cap is not None:
        goal = max(goal, (CLOCK_DAYS + 1) * SESSION_MINUTES)
    return goal


def following_day(bar_time: datetime, last_day: date) -> date | None:
    """День следующей минуты для движка.

    До последнего дня передаём дату самой свечи: вход до trade_from закрыт,
    потому что эта дата раньше начала торговли. В последний день после 23:40
    остаток можно снять целиком, раньше — по десять контрактов в минуту.
    """
    local = bar_time.astimezone(MSK)
    if local.date() == last_day and local.time() >= FLATTEN_AT:
        return None
    return local.date()


def make_order_id(strategy: str, when: datetime) -> str:
    """Один и тот же id на эту минуту: повторный вызов не ставит вторую заявку."""
    stamp = when.astimezone(MSK).strftime("%Y%m%d%H%M")
    return f"{strategy}-{stamp}"[:36]


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

    front, trade_from = choose_front_safe(broker.cny_futures(), moment.date())
    result["secid"] = front.secid
    if _adopt_instrument(state, front, trade_from) == "halt":
        return _halt(state, store, key, result, "контракт сменился, пока позиция ещё открыта", None)

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
                state["last_bar"] = bars[-1]["t"]
        elif walked >= HISTORY_WALKS:
            state["book"] = export_book(book)
            return _halt(state, store, key, result, "не хватает минуток, чтобы собрать окно", book)
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "history"
        result["bars"] = len(bars)
        result["goal"] = goal
        result["ready"] = bool(state.get("ready"))
        return result

    broker_lots = int(broker.futures_position(request.account_id, front.uid))
    if request.reconcile:
        _adopt_position(book, broker, request.account_id, front, broker_lots)
        state["halted"] = None
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "reconciled"
        result["held"] = book.held
        result["halted"] = None
        return result
    if broker_lots != book.held:
        return _halt(state, store, key, result, f"позиция на счёте {broker_lots}, в книге {book.held}", book)

    if book.held == 0:
        book.equity = float(broker.equity(request.account_id))

    fresh = _fresh_candles(state, broker, front.uid, moment)
    if fresh is None:
        store.save(key, state)
        result["phase"] = "catchup"
        result["bars"] = len(state.get("bars") or [])
        result["held"] = book.held
        return result
    if not fresh:
        store.save(key, state)
        result["phase"] = "idle"
        result["bars"] = len(state.get("bars") or [])
        result["held"] = book.held
        result["target"] = book.target
        return result

    state["bars"] = _merge_bars(list(state.get("bars") or []), [_bar(candle) for candle in fresh])
    if len(fresh) > 1:
        for candle in fresh[:-1]:
            if book.stop_hit(candle.open, candle.high, candle.low):
                book.target = 0
                book.reason = "stop"
                break

    margin = float(broker.margin(front.uid))
    if margin <= 0:
        return _halt(state, store, key, result, "биржа не вернула гарантийное обеспечение", book)

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
    )
    stamp = _bar(last)["t"]
    signed = int(book.last_signed)
    if signed == 0:
        state["last_bar"] = stamp
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "signal" if book.target != book.held else "hold"
        result["held"] = book.held
        result["target"] = book.target
        result["bars"] = len(state["bars"])
        return result

    order_id = make_order_id(request.strategy, last.time)
    try:
        report = broker.market_order(request.account_id, front.uid, signed, order_id)
    except Exception:
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(state, store, key, result, "заявка не подтверждена", book)
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
        )

    price = last.close if report.price is None or report.price <= 0 else float(report.price)
    reprice_fill(book, before, last.close, price)
    if book.held == 0:
        try:
            book.equity = float(broker.equity(request.account_id))
        except Exception:
            pass
    state["last_bar"] = stamp
    state["book"] = export_book(book)
    store.save(key, state)
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
    return result


def choose_front_safe(instruments: list[Instrument], today: date) -> tuple[Instrument, date]:
    return choose_front(instruments, today)


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
        bars = _merge_bars(bars, extra)
        cursor = end
    state["bars"] = bars
    state["sync_from"] = cursor.isoformat()
    if cursor < now - timedelta(minutes=2):
        return None
    last_raw = state.get("last_bar")
    if not isinstance(last_raw, str):
        return []
    last_dt = datetime.fromisoformat(last_raw)
    pending: list[Candle] = []
    for bar in bars:
        moment = datetime.fromisoformat(str(bar["t"]))
        if moment > last_dt:
            pending.append(_candle_from_bar(bar, moment))
    return pending


def _sync_cursor(state: dict[str, object], bars: list[dict[str, object]], now: datetime) -> datetime:
    raw = state.get("sync_from")
    if isinstance(raw, str):
        return datetime.fromisoformat(raw)
    if bars:
        return datetime.fromisoformat(str(bars[-1]["t"]))
    last = state.get("last_bar")
    if isinstance(last, str):
        return datetime.fromisoformat(last)
    return now - timedelta(days=1)


def _candle_from_bar(bar: dict[str, object], moment: datetime) -> Candle:
    return Candle(
        time=moment,
        open=float(bar["o"]),
        high=float(bar["h"]),
        low=float(bar["l"]),
        close=float(bar["c"]),
        volume=float(bar["v"]),
    )


def _adopt_position(book: _FillBook, broker, account_id: str, front: Instrument, lots: int) -> None:
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
        return
    price = broker.position_price(account_id, front.uid)
    if price is not None:
        book.avg = float(price)


def _halt(
    state: dict[str, object],
    store: StateStore,
    key: str,
    result: dict[str, object],
    reason: str,
    book: _FillBook | None,
) -> dict[str, object]:
    state["halted"] = reason
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


def _merge_bars(existing: list[dict[str, object]], extra: list[dict[str, object]]) -> list[dict[str, object]]:
    by_time = {str(bar["t"]): bar for bar in existing}
    for bar in extra:
        by_time[str(bar["t"])] = bar
    return [by_time[key] for key in sorted(by_time)]


def _moscow(now: datetime | None) -> datetime:
    moment = now or datetime.now(MSK)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MSK)
    return moment.astimezone(MSK)
