"""Один вызов функции — одна минута одного счёта.

История окна качается по одному московскому дню и в этот вызов заявка на
фьючерс не ставится. Первый ордер на фьючерс — на минуте, которая закрылась
уже после того, как окно собрано. Пропущенные минуты дописываются в канал и
проверяются на стоп, шаг считается только по последней закрытой свече, и на
фьючерс уходит не больше одного рыночного ордера. В бакете остаётся окно
индикаторов и десять сессий сверху, более старые минутки стираются.
Свежий вход против старшего окна, которое уже два торговых дня в плюсе,
снимается до заявки. Своя позиция, стоп и выход остаются.
Моментум — восьмая стратегия на своём субсчёте: дневной знак, без запрета семи окон.

Сколько контрактов менять за минуту решает fill_per_minute. Пустое значение —
авто: не больше половины видимого стакана с той стороны, которую съест заявка.
0 ставит паузу: цель может смениться, но заявка не уходит. Положительное число —
потолок контрактов в минуту, стакан при этом не смотрим.
"""

from __future__ import annotations

import uuid
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
from cnyrub.live.broker import Candle, Instrument, book_lots, choose_front, half_book
from cnyrub.live.config import MOMENTUM, RunRequest
from cnyrub.live.indicators import bar_get, bar_levels
from cnyrub.live.momentum import HISTORY_DAYS, absorb, fold_candle, sign_for_today, target_lots
from cnyrub.live.state import StateStore, state_key
from cnyrub.live.veto import cancel_fresh_entry, publish_peer, veto_source

MSK = ZoneInfo("Europe/Moscow")
ZERO_FILL_LIMIT = 10
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
    В последний день после 23:40 следующая минута для движка уже не наступает:
    новую позицию он не открывает.
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
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cnyrub:{strategy}:{stamp}"))


def fill_price(reported: float | None, close: float) -> float:
    """Цена исполнения из ответа брокера; если масштаб не тот, берём close.

    Для фьючерса брокер может прислать цену в пунктах либо в рублях за контракт
    (в последнем случае в 1000 раз больше). Принимаем только значение в пределах
    10% от последней цены, иначе средняя цена в книге и стоп были бы испорчены.
    """
    if reported is None or reported <= 0:
        return close
    for candidate in (float(reported), float(reported) / 1000):
        if abs(candidate - close) <= 0.1 * close:
            return candidate
    return close


def move_side(
    book: _FillBook,
    *,
    day: date,
    last_day: date,
    opened: float,
    high: float,
    low: float,
) -> int:
    """Куда шагнет эта минута до нового решения: 1 покупка, −1 продажа, 0 стоит.

    Последний день и стоп обнуляют цель так же, как шаг движка перед сделкой.
    """
    held = book.held
    target = book.target
    if day == last_day and (held != 0 or target != 0):
        target = 0
    elif book.stop_hit(opened, high, low):
        target = 0
    delta = target - held
    if held > 0:
        delta = max(delta, -held)
    elif held < 0:
        delta = min(delta, -held)
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def pace_for(request: RunRequest, broker, uid: str, side: int) -> int:
    """Контрактов за эту минуту. Авто читает сторону стакана, которую съест заявка."""
    if request.fill_per_minute is not None:
        return request.fill_per_minute
    if side == 0:
        return 0
    payload = broker.order_book(uid)
    if not isinstance(payload, dict):
        raise RuntimeError("стакан пустой")
    return half_book(book_lots(payload, "asks" if side > 0 else "bids"))


def order_reason(book: _FillBook, before: dict[str, object], signed: int) -> str:
    """Короткая причина этого куска: пробой, добор, возврат или выход.

    before — книга до шага этой минуты. Первый кусок новой позиции берёт
    пробой, который записан там. Выход, назначенный уже после куска, сюда не входит.
    """
    before_held = int(before.get("held") or 0)
    if signed == 0:
        return ""
    if before_held == 0 or signed * before_held > 0:
        if before_held == 0:
            entry = str(before.get("entry") or "")
            if entry in {"up", "down"}:
                return entry
            return "up" if signed > 0 else "down"
        target = int(before.get("target") or 0)
        if before.get("scaled") and int(before.get("scale_level") or 0) == 0 and abs(target) > abs(before_held):
            return "scale_back"
        return "add"
    if book.held == 0 and book.trades:
        reason = str(book.trades[-1].get("reason") or "")
    else:
        reason = str(book.reason or "")
    if reason:
        return reason
    if book.scaled or before.get("scaled"):
        return "scale"
    return ""


def close_pnl(book: _FillBook, before: dict[str, object], signed: int) -> float | None:
    """Результат сокращения. На полном закрытии — вся сделка после комиссии.

    Пока позиция ещё открыта, это только контракты этой минуты: разница цены
    и комиссия их выхода. Комиссия входа остаётся в итоге, когда позиция дойдёт до нуля.
    """
    before_held = int(before.get("held") or 0)
    if signed == 0 or before_held == 0 or signed * before_held > 0:
        return None
    if book.held == 0 and book.trades:
        return float(book.trades[-1]["pnlcomm"])
    gross = float(book.gross) - float(before.get("gross") or 0)
    commission = float(book.commission) - float(before.get("commission") or 0)
    return gross - commission


def run_momentum(request: RunRequest, broker, store: StateStore, now: datetime | None = None) -> dict[str, object]:
    """Дневной моментум. Чужой запрет не читает и свой файл peers/ не пишет."""
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
    adopted = _adopt_instrument(state, front, trade_from)
    if adopted == "halt":
        return _halt(
            state,
            store,
            key,
            result,
            "контракт сменился, пока позиция ещё открыта",
            None,
            request.params,
            request.strategy,
            0,
        )
    if adopted == "reset":
        state["days"] = []
        state["today"] = None
        state["sized_day"] = None

    book = load_book(state.get("book") if isinstance(state.get("book"), dict) else None, front.secid, None)
    if not state.get("ready"):
        _pull_momentum_day(state, broker, front, moment.date())
        days = list(state.get("days") or [])
        if len(days) >= HISTORY_DAYS:
            state["ready"] = True
        elif int(state.get("history_walked") or 0) >= HISTORY_WALKS:
            state["book"] = export_book(book)
            return _halt(
                state,
                store,
                key,
                result,
                "не хватает дней, чтобы собрать моментум",
                book,
                request.params,
                request.strategy,
                0,
            )
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "history"
        result["bars"] = len(days)
        result["goal"] = HISTORY_DAYS
        result["ready"] = bool(state.get("ready"))
        return result

    broker_lots = int(broker.futures_position(request.account_id, front.uid))
    if request.reconcile:
        _adopt_position(book, broker, request, front, broker_lots, [])
        book.stop_dist = None
        book.stop_px = None
        state["halted"] = None
        state["zero_fills"] = 0
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
            request.strategy,
            0,
        )

    fresh = _momentum_fresh(state, broker, front.uid, moment)
    if fresh is None:
        store.save(key, state)
        result["phase"] = "catchup"
        result["held"] = book.held
        return result
    if not fresh:
        result["phase"] = "idle"
        result["held"] = book.held
        result["target"] = book.target
        return result

    days = [day for day in list(state.get("days") or []) if isinstance(day, dict)]
    today_bar = state.get("today") if isinstance(state.get("today"), dict) else None
    days, today_bar = absorb(days, today_bar, fresh)
    state["days"] = days
    state["today"] = today_bar
    last = fresh[-1]
    local = last.time.astimezone(MSK)
    if today_bar is not None and state.get("sized_day") != today_bar.get("d") and days:
        margin = _margin(state, broker, front.uid, moment.date(), refresh=True)
        if margin <= 0:
            return _halt(
                state,
                store,
                key,
                result,
                "биржа не вернула гарантийное обеспечение",
                book,
                request.params,
                request.strategy,
                0,
            )
        try:
            book.equity = float(broker.equity(request.account_id))
        except Exception:
            book.equity = book.equity
        book.stop_dist = None
        book.stop_px = None
        book.target = target_lots(sign_for_today(days), float(book.equity or 0.0), float(days[-1]["c"]), margin)
        if book.target == 0 and book.held:
            book.reason = "flat"
        state["sized_day"] = today_bar.get("d")

    before = export_book(book)
    side = move_side(book, day=local.date(), last_day=front.lsttrade, opened=last.open, high=last.high, low=last.low)
    try:
        pace = pace_for(request, broker, front.uid, side)
    except Exception:
        result["phase"] = "book"
        result["held"] = book.held
        result["target"] = book.target
        return result

    def advance(limit: int) -> None:
        step_minute(
            book,
            len(days),
            opened=last.open,
            high=last.high,
            low=last.low,
            close=last.close,
            volume=last.volume,
            day=local.date(),
            next_day=following_day(local, front.lsttrade),
            last_day=front.lsttrade,
            prior_high=float("nan"),
            prior_low=float("nan"),
            prior_vol=float("nan"),
            prior_range=float("nan"),
            exit_high=float("nan"),
            exit_low=float("nan"),
            clock_vol=float("nan"),
            entry_ready=0.0,
            clearance=0.0,
            cooldown=0,
            clock_volume=False,
            clock_cap=None,
            size_mode="flat",
            stop_mult=None,
            loss_bars=None,
            risk_fraction=None,
            margin=0.0,
            stop_rub=None,
            breakout_span=None,
            trade_from=trade_from,
            trail=False,
            fill_per_minute=limit,
        )

    advance(pace)
    stamp = last.time.astimezone(MSK).isoformat()
    signed = int(book.last_signed)
    if signed == 0:
        state["last_bar"] = stamp
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "paused" if request.fill_per_minute == 0 else ("signal" if book.target != book.held else "hold")
        result["held"] = book.held
        result["target"] = book.target
        result["bars"] = len(days)
        return result

    order_id = make_order_id(request.strategy, last.time)
    try:
        report = broker.market_order(request.account_id, front.uid, signed, order_id)
    except Exception:
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(state, store, key, result, "заявка не подтверждена", book, request.params, request.strategy, 0)
    wanted = abs(signed)
    if report.executed > wanted:
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(
            state,
            store,
            key,
            result,
            f"исполнено больше заявленного: {report.executed} из {wanted}",
            book,
            request.params,
            request.strategy,
            0,
        )
    if report.executed <= 0:
        restore_book(book, before)
        misses = int(state.get("zero_fills") or 0) + 1
        state["zero_fills"] = misses
        if misses >= ZERO_FILL_LIMIT:
            state["last_bar"] = stamp
            return _halt(
                state,
                store,
                key,
                result,
                f"заявка не исполнена {misses} раз подряд",
                book,
                request.params,
                request.strategy,
                0,
            )
        state["book"] = export_book(book)
        store.save(key, state)
        result["phase"] = "unfilled"
        result["unfilled"] = misses
        result["held"] = book.held
        result["target"] = book.target
        return result
    state["zero_fills"] = 0
    if report.executed < wanted:
        restore_book(book, before)
        advance(report.executed)
        signed = int(book.last_signed)
    price = fill_price(report.price, last.close)
    reprice_fill(book, before, last.close, price)
    state["last_bar"] = stamp
    state["book"] = export_book(book)
    store.save(key, state)
    result["phase"] = "order"
    result["held"] = book.held
    result["target"] = book.target
    result["bars"] = len(days)
    result["order"] = {
        "id": order_id,
        "signed": signed,
        "requested": wanted,
        "executed": report.executed,
        "price": price,
        "time": local.strftime("%Y-%m-%d %H:%M"),
        "reason": order_reason(book, before, signed),
    }
    pnl = close_pnl(book, before, signed)
    if pnl is not None:
        result["order"]["pnl"] = pnl
    return result


def _pull_momentum_day(state: dict[str, object], broker, front: Instrument, today: date) -> None:
    """Один прошедший день в дневную свечу. Сегодняшняя сессия остаётся незакрытой."""
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
        if start <= candle.time.astimezone(MSK) < end and _weekday(candle.time)
    ]
    candles.sort(key=lambda candle: candle.time)
    state["history_before"] = day.isoformat()
    state["history_walked"] = walked + 1
    if not candles:
        return
    folded: dict[str, object] | None = None
    for candle in candles:
        local = candle.time.astimezone(MSK)
        folded = fold_candle(folded, local, candle.open, candle.high, candle.low, candle.close, candle.volume)
    if folded is None:
        return
    if day < today:
        days = [item for item in list(state.get("days") or []) if isinstance(item, dict)]
        days.append(folded)
        days.sort(key=lambda item: str(item["d"]))
        state["days"] = days
    else:
        state["today"] = folded
    state["last_bar"] = candles[-1].time.astimezone(MSK).isoformat()


def _momentum_fresh(state: dict[str, object], broker, uid: str, now: datetime) -> list[Candle] | None:
    """Новые минутки после last_bar. Минутную историю в бакет не кладём."""
    cursor = _sync_cursor(state, [], now) - timedelta(minutes=2)
    downloaded: list[Candle] = []
    for _ in range(CATCHUP_DAYS):
        if cursor >= now - timedelta(minutes=1):
            break
        end = cursor + timedelta(days=1)
        if end > now:
            end = now
        if end <= cursor:
            break
        batch = broker.candles(uid, cursor, end)
        downloaded.extend(
            candle
            for candle in batch
            if cursor <= candle.time.astimezone(MSK) < end and _weekday(candle.time)
        )
        cursor = end
    if _sync_is_stale(state, cursor, now) or cursor < now - timedelta(minutes=2):
        state["sync_from"] = cursor.isoformat()
    if cursor < now - timedelta(minutes=2):
        return None
    last_raw = state.get("last_bar")
    if not isinstance(last_raw, str):
        return []
    last_dt = datetime.fromisoformat(last_raw)
    pending = [candle for candle in downloaded if candle.time.astimezone(MSK) > last_dt]
    pending.sort(key=lambda candle: candle.time)
    return pending


def run_minute(request: RunRequest, broker, store: StateStore, now: datetime | None = None) -> dict[str, object]:
    if request.strategy == MOMENTUM:
        return run_momentum(request, broker, store, now)
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
        return _halt(
            state,
            store,
            key,
            result,
            "контракт сменился, пока позиция ещё открыта",
            None,
            request.params,
            request.strategy,
            request.params.channel,
        )

    book = load_book(state.get("book") if isinstance(state.get("book"), dict) else None, front.secid, None)
    if not state.get("ready"):
        _pull_history_day(state, broker, front, moment.date(), request.params.weekends)
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
            return _halt(
                state,
                store,
                key,
                result,
                "не хватает минуток, чтобы собрать окно",
                book,
                request.params,
                request.strategy,
                request.params.channel,
            )
        _trim_bars(state, book, request.params)
        state["book"] = export_book(book)
        store.save(key, state)
        publish_peer(store, request.strategy, request.params.channel, book, list(state.get("bars") or []))
        result["phase"] = "history"
        result["bars"] = len(state.get("bars") or [])
        result["goal"] = goal
        result["ready"] = bool(state.get("ready"))
        return result

    broker_lots = int(broker.futures_position(request.account_id, front.uid))
    if request.reconcile:
        _adopt_position(book, broker, request, front, broker_lots, list(state.get("bars") or []))
        state["halted"] = None
        state["zero_fills"] = 0
        _trim_bars(state, book, request.params)
        state["book"] = export_book(book)
        store.save(key, state)
        publish_peer(store, request.strategy, request.params.channel, book, list(state.get("bars") or []))
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
            request.strategy,
            request.params.channel,
        )

    sync_before = state.get("sync_from")
    bars_before = len(state.get("bars") or [])
    fresh = _fresh_candles(state, broker, front.uid, moment, request.params.weekends)
    if fresh is None:
        _save_trimmed(state, store, key, book, request.params, request.strategy, request.params.channel)
        result["phase"] = "catchup"
        result["bars"] = len(state.get("bars") or [])
        result["held"] = book.held
        return result
    if not fresh:
        book_changed = _trim_bars(state, book, request.params)
        moved = state.get("sync_from") != sync_before or len(state.get("bars") or []) != bars_before
        if book_changed or moved:
            if book_changed:
                state["book"] = export_book(book)
            store.save(key, state)
            publish_peer(store, request.strategy, request.params.channel, book, list(state.get("bars") or []))
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
        return _halt(
            state,
            store,
            key,
            result,
            "биржа не вернула гарантийное обеспечение",
            book,
            request.params,
            request.strategy,
            request.params.channel,
        )

    last = fresh[-1]
    before = export_book(book)
    levels = bar_levels(list(state["bars"]), request.params)
    local = last.time.astimezone(MSK)
    side = move_side(
        book,
        day=local.date(),
        last_day=front.lsttrade,
        opened=last.open,
        high=last.high,
        low=last.low,
    )
    try:
        pace = pace_for(request, broker, front.uid, side)
    except Exception:
        result["phase"] = "book"
        result["held"] = book.held
        result["target"] = book.target
        return result
    def advance(limit: int) -> None:
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
            cooldown=request.params.cooldown,
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
            fill_per_minute=limit,
            drift=levels["drift"],
            path=levels["path"],
            surge_vol=levels["surge_vol"],
            eff_low=request.params.eff_low,
            eff_high=request.params.eff_high,
            surge_cap=request.params.surge_cap,
            leverage=request.params.leverage,
            min_stop=request.params.min_stop,
            skip_flat=request.params.skip_flat,
        )

    advance(pace)
    blocked_by = ""
    if book.last_signed == 0 and book.held == 0 and book.target != 0:
        blocked_by = veto_source(store, request.strategy, request.params.channel, book)
        if blocked_by:
            cancel_fresh_entry(book)
    stamp = _bar(last)["t"]
    signed = int(book.last_signed)
    if signed == 0:
        state["last_bar"] = stamp
        _trim_bars(state, book, request.params)
        state["book"] = export_book(book)
        store.save(key, state)
        publish_peer(store, request.strategy, request.params.channel, book, list(state.get("bars") or []))
        if blocked_by:
            result["phase"] = "veto"
            result["veto"] = blocked_by
        elif request.fill_per_minute == 0:
            result["phase"] = "paused"
        else:
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
        return _halt(
            state,
            store,
            key,
            result,
            "заявка не подтверждена",
            book,
            request.params,
            request.strategy,
            request.params.channel,
        )
    wanted = abs(signed)
    if report.executed > wanted:
        restore_book(book, before)
        state["last_bar"] = stamp
        return _halt(
            state,
            store,
            key,
            result,
            f"исполнено больше заявленного: {report.executed} из {wanted}",
            book,
            request.params,
            request.strategy,
            request.params.channel,
        )
    if report.executed <= 0:
        restore_book(book, before)
        misses = int(state.get("zero_fills") or 0) + 1
        state["zero_fills"] = misses
        if misses >= ZERO_FILL_LIMIT:
            state["last_bar"] = stamp
            return _halt(
                state,
                store,
                key,
                result,
                f"заявка не исполнена {misses} раз подряд",
                book,
                request.params,
                request.strategy,
                request.params.channel,
            )
        # last_bar остаётся прежним: следующая минута заново решит, нужна ли заявка.
        _trim_bars(state, book, request.params)
        state["book"] = export_book(book)
        store.save(key, state)
        publish_peer(store, request.strategy, request.params.channel, book, list(state.get("bars") or []))
        result["phase"] = "unfilled"
        result["unfilled"] = misses
        result["held"] = book.held
        result["target"] = book.target
        result["bars"] = len(state["bars"])
        return result
    state["zero_fills"] = 0
    if report.executed < wanted:
        # Принимаем исполненную часть: книга повторяет минуту с этим потолком, остаток
        # цель дозаберёт на следующих минутах.
        restore_book(book, before)
        advance(report.executed)
        signed = int(book.last_signed)

    price = fill_price(report.price, last.close)
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
        "requested": wanted,
        "executed": report.executed,
        "price": price,
        "time": local.strftime("%Y-%m-%d %H:%M"),
        "reason": order_reason(book, before, signed),
    }
    pnl = close_pnl(book, before, signed)
    if pnl is not None:
        result["order"]["pnl"] = pnl
    state["book"] = export_book(book)
    store.save(key, state)
    publish_peer(store, request.strategy, request.params.channel, book, list(state.get("bars") or []))
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


def _pull_history_day(
    state: dict[str, object], broker, front: Instrument, today: date, weekends: bool = False
) -> None:
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
        if start <= candle.time.astimezone(MSK) < end and (weekends or _weekday(candle.time))
    ]
    state["bars"] = _merge_bars(list(state.get("bars") or []), [_bar(candle) for candle in candles])
    state["history_before"] = day.isoformat()
    state["history_walked"] = walked + 1


def _weekday(moment: datetime) -> bool:
    """Суббота и воскресенье: сессия выходного дня тонкая, окно по ней не считаем."""
    return moment.astimezone(MSK).weekday() < 5


def _fresh_candles(
    state: dict[str, object], broker, uid: str, now: datetime, weekends: bool = False
) -> list[Candle] | None:
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
            if cursor <= candle.time.astimezone(MSK) < end and (weekends or _weekday(candle.time))
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


def _save_trimmed(
    state: dict[str, object],
    store: StateStore,
    key: str,
    book: _FillBook,
    params,
    strategy: str,
    channel: int,
) -> None:
    if _trim_bars(state, book, params):
        state["book"] = export_book(book)
    store.save(key, state)
    publish_peer(store, strategy, channel, book, list(state.get("bars") or []))


def _halt(
    state: dict[str, object],
    store: StateStore,
    key: str,
    result: dict[str, object],
    reason: str,
    book: _FillBook | None,
    params,
    strategy: str,
    channel: int,
) -> dict[str, object]:
    if not state.get("halted"):
        result["halt_new"] = True
    state["halted"] = reason
    _trim_bars(state, book, params)
    if book is not None:
        state["book"] = export_book(book)
    store.save(key, state)
    publish_peer(store, strategy, channel, book, list(state.get("bars") or []))
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
