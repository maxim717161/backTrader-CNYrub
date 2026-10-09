"""Живой контур без сети: разбор событий, история, заявка, стоп и сверка."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest import WINDOWS, channel_view
from cnyrub.engine import _FillBook, export_book, reprice_fill, step_minute
from cnyrub.live.broker import (
    FillReport,
    Instrument,
    book_lots,
    choose_front,
    half_book,
    margin_rub,
    api_error_text,
    parse_candle,
    parse_fill,
    parse_instrument,
    quotation,
)
from cnyrub.live.config import PRESETS, parse_event
from cnyrub.live.handler import handle, read_lockbox_token
from cnyrub.live.indicators import bar_levels
from cnyrub.live.service import (
    following_day,
    bars_to_keep,
    history_goal,
    close_pnl,
    make_order_id,
    order_reason,
    run_minute,
)
from cnyrub.live.telegram import notify_trade, trade_text
from cnyrub.live.state import MemoryStore, state_key

MSK = ZoneInfo("Europe/Moscow")
ACCOUNT = "acc-short"


def test_presets_match_the_researched_windows():
    short, long, thirty, fortyfive, sixty = WINDOWS
    assert PRESETS["short"].channel == short.channel == 525
    assert PRESETS["short"].exit_channel == short.exit_channel
    assert PRESETS["short"].stop_mult is None and short.stop_mult is None
    assert PRESETS["short"].clock_cap == short.clock_cap == 5
    assert PRESETS["short"].loss_bars == short.loss_bars == 1450
    assert PRESETS["short"].risk_fraction == short.risk_fraction == 0.10
    assert PRESETS["short"].stop_rub == short.stop_rub == 285
    assert PRESETS["short"].scale_step is None and short.scale_step is None
    assert PRESETS["short"].breakout_span is None
    assert PRESETS["long"].channel == long.channel == 12420
    assert PRESETS["long"].exit_channel == 0
    assert PRESETS["long"].stop_mult == long.stop_mult == 22
    assert PRESETS["long"].breakout_span == long.breakout_span == 12
    assert PRESETS["long"].scale_step == long.scale_step == 100
    assert PRESETS["long"].scale_back == long.scale_back == 50
    assert PRESETS["long"].scale_floor == long.scale_floor == 0.5
    assert PRESETS["long"].clock_cap is None
    assert not hasattr(PRESETS["short"], "margin")
    assert PRESETS["thirty"].channel == thirty.channel == 30
    assert PRESETS["thirty"].exit_channel == 0
    assert PRESETS["thirty"].stop_mult == thirty.stop_mult == 8
    assert PRESETS["thirty"].clock_cap == 5
    assert PRESETS["thirty"].eff_low == thirty.eff_low == 0.15
    assert PRESETS["thirty"].eff_high == thirty.eff_high == 0.5
    assert PRESETS["thirty"].surge_cap == thirty.surge_cap == 3
    assert PRESETS["thirty"].leverage == thirty.leverage == 4
    assert thirty.cash == 100_000
    assert PRESETS["fortyfive"].channel == fortyfive.channel == 45
    assert PRESETS["fortyfive"].exit_channel == 0
    assert PRESETS["fortyfive"].stop_mult == fortyfive.stop_mult == 8
    assert PRESETS["fortyfive"].clock_cap == 5
    assert PRESETS["fortyfive"].eff_low is None and fortyfive.eff_low is None
    assert PRESETS["fortyfive"].surge_cap is None and fortyfive.surge_cap is None
    assert PRESETS["fortyfive"].leverage == fortyfive.leverage == 4
    assert fortyfive.cash == 100_000
    assert PRESETS["sixty"].channel == sixty.channel == 60
    assert PRESETS["sixty"].exit_channel == 0
    assert PRESETS["sixty"].stop_mult == sixty.stop_mult == 8
    assert PRESETS["sixty"].clock_cap == 5
    assert PRESETS["sixty"].eff_low is None and sixty.eff_low is None
    assert PRESETS["sixty"].surge_cap is None and sixty.surge_cap is None
    assert PRESETS["sixty"].leverage == sixty.leverage == 5
    assert sixty.cash == 100_000
    assert parse_event(
        {"strategy": "sixty", "account_id": "1", "token": "t"}
    ).fill_per_minute is None
    assert parse_event(
        {"strategy": "sixty", "account_id": "1", "token": "t", "fill_per_minute": "авто"}
    ).fill_per_minute is None
    assert parse_event(
        {"strategy": "sixty", "account_id": "1", "token": "t", "fill_per_minute": 0}
    ).fill_per_minute == 0
    assert parse_event(
        {"strategy": "sixty", "account_id": "1", "token": "t", "fill_per_minute": 25}
    ).fill_per_minute == 25


def test_levels_match_channel_view():
    index = pd.date_range("2026-09-01 10:00", periods=48, freq="min", tz="Europe/Moscow")
    frame = pd.DataFrame(
        {
            "open": [10 + i * 0.01 for i in range(len(index))],
            "high": [10.2 + (i % 5) * 0.01 for i in range(len(index))],
            "low": [9.8 + (i % 4) * 0.02 for i in range(len(index))],
            "close": [10.05 + i * 0.01 for i in range(len(index))],
            "volume": [100 + (i % 7) * 3 for i in range(len(index))],
        },
        index=index,
    )
    # Пять предыдущих дней той же минуты: часы 10:00 повторяются отдельно.
    clock = pd.date_range("2026-09-01 10:00", periods=8, freq="D", tz="Europe/Moscow")
    extra = pd.DataFrame(
        {
            "open": [11] * len(clock),
            "high": [11.4] * len(clock),
            "low": [10.6] * len(clock),
            "close": [11.1] * len(clock),
            "volume": [50 + day * 10 for day in range(len(clock))],
        },
        index=clock,
    )
    frame = pd.concat([frame, extra]).sort_index()
    view = channel_view(frame, 5, 3)
    bars = [
        {
            "t": moment.isoformat(),
            "o": float(row.open),
            "h": float(row.high),
            "l": float(row.low),
            "c": float(row.close),
            "v": float(row.volume),
        }
        for moment, row in frame.iterrows()
    ]
    params = PRESETS["short"]
    params = type(params)(
        channel=5,
        exit_channel=3,
        stop_mult=params.stop_mult,
        clock_cap=params.clock_cap,
        size_mode=params.size_mode,
        loss_bars=params.loss_bars,
        risk_fraction=params.risk_fraction,
        stop_rub=params.stop_rub,
        breakout_span=params.breakout_span,
        scale_step=params.scale_step,
        scale_back=params.scale_back,
        scale_floor=params.scale_floor,
    )
    for end in (6, 12, 20, len(bars)):
        levels = bar_levels(bars[:end], params, clock_days=5)
        row = view.iloc[end - 1]
        for name in (
            "prior_high",
            "prior_low",
            "prior_vol",
            "prior_range",
            "exit_high",
            "exit_low",
            "clock_vol",
            "surge_vol",
            "drift",
            "path",
        ):
            got = levels[name]
            expected = float(row[name])
            if pd.isna(expected):
                assert pd.isna(got)
            else:
                assert got == pytest.approx(expected)
        assert levels["entry_ready"] == float(row["entry_ready"])


def test_surge_and_straightness_match_channel_view():
    index = pd.date_range("2026-01-05 10:00", periods=400, freq="min", tz="Europe/Moscow")
    frame = pd.DataFrame(
        {
            "open": [10.0] * len(index),
            "high": [10.2] * len(index),
            "low": [9.8] * len(index),
            "close": [10.0 + i * 0.001 for i in range(len(index))],
            "volume": [10 + (i % 17) * 3 + (i % 2) for i in range(len(index))],
        },
        index=index,
    )
    view = channel_view(frame, 30, 0)
    bars = [
        {
            "t": moment.isoformat(),
            "o": float(row.open),
            "h": float(row.high),
            "l": float(row.low),
            "c": float(row.close),
            "v": float(row.volume),
        }
        for moment, row in frame.iterrows()
    ]
    params = PRESETS["thirty"]
    for end in (60, 61, 80, 300, 301, 400):
        levels = bar_levels(bars[:end], params)
        row = view.iloc[end - 1]
        for name in ("surge_vol", "drift", "path", "prior_high", "prior_vol"):
            got = levels[name]
            expected = float(row[name])
            if pd.isna(expected):
                assert pd.isna(got)
            else:
                assert got == pytest.approx(expected)
    assert history_goal(params) == (5 + 1) * 18 * 60


def test_parse_timer_envelope_and_direct_json():
    direct = parse_event(
        {"strategy": "long", "account_id": "42", "token": "secret-token", "reconcile": "false"}
    )
    assert direct.strategy == "long"
    assert direct.account_id == "42"
    assert direct.token == "secret-token"
    assert direct.reconcile is False
    assert direct.params.channel == 12420
    assert direct.fill_per_minute is None

    timer = parse_event(
        {
            "messages": [
                {
                    "details": {
                        "payload": '{"strategy":"short","account_id":"7","secret_id":"lockbox","fill_per_minute":2}'
                    }
                }
            ]
        }
    )
    assert timer.strategy == "short"
    assert timer.secret_id == "lockbox"
    assert timer.token is None
    assert timer.fill_per_minute == 2
    assert timer.params.channel == 525


def test_parse_raw_string_bom_and_empty_body():
    raw = '{"strategy":"sixty","account_id":"9","token":"t"}'
    parsed = parse_event(raw)
    assert parsed.strategy == "sixty"
    assert parsed.account_id == "9"
    assert parse_event("\ufeff" + raw).strategy == "sixty"
    assert parse_event(raw.encode("utf-8")).strategy == "sixty"

    with pytest.raises(ValueError, match="пустое"):
        parse_event("")
    with pytest.raises(ValueError, match="пустое"):
        parse_event("   \n")
    with pytest.raises(ValueError, match="не JSON"):
        parse_event("not-json")


def test_parse_https_invoke_envelope():
    body = '{"strategy":"fortyfive","account_id":"3","secret_id":"box"}'
    parsed = parse_event(
        {
            "httpMethod": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": body,
            "isBase64Encoded": False,
        }
    )
    assert parsed.strategy == "fortyfive"
    assert parsed.secret_id == "box"

    encoded = base64.b64encode(body.encode()).decode()
    parsed_b64 = parse_event({"httpMethod": "POST", "headers": {}, "body": encoded, "isBase64Encoded": True})
    assert parsed_b64.account_id == "3"

    with pytest.raises(ValueError, match="пустое"):
        parse_event({"httpMethod": "POST", "headers": {}, "body": "", "isBase64Encoded": False})


def test_parse_requires_account_and_a_secret():
    with pytest.raises(ValueError):
        parse_event({"strategy": "short", "token": "t"})
    with pytest.raises(ValueError):
        parse_event({"strategy": "short", "account_id": "1"})
    with pytest.raises(ValueError):
        parse_event({"strategy": "short", "account_id": "1", "token": "t", "fill_per_minute": -1})
    with pytest.raises(ValueError):
        parse_event({"strategy": "short", "account_id": "1", "token": "t", "fill_per_minute": "много"})


def test_lockbox_reads_token_entry_without_returning_it_in_the_url():
    seen: list[str] = []

    def get(url: str, headers: dict[str, str]) -> dict:
        seen.append(url)
        if "computeMetadata" in url:
            assert headers["Metadata-Flavor"] == "Google"
            return {"access_token": "iam"}
        assert headers["Authorization"] == "Bearer iam"
        return {"entries": [{"key": "token", "textValue": "real-token"}]}

    assert read_lockbox_token("secret-1", get) == "real-token"
    assert seen[1].endswith("/secrets/secret-1/payload")


def test_quotation_margin_candle_and_front_contract():
    assert quotation({"units": "11", "nano": 250000000}) == pytest.approx(11.25)
    assert margin_rub(
        {"initialMarginOnBuy": {"units": "1400", "nano": 0}, "initialMarginOnSell": {"units": "1500", "nano": 0}}
    ) == 1500
    candle = parse_candle(
        {
            "time": "2026-09-28T07:00:00Z",
            "open": {"units": "11", "nano": 0},
            "high": {"units": "12", "nano": 0},
            "low": {"units": "10", "nano": 0},
            "close": {"units": "11", "nano": 500000000},
            "volume": "40",
            "isComplete": True,
        }
    )
    assert candle is not None
    assert candle.close == pytest.approx(11.5)
    assert candle.time.astimezone(MSK).hour == 10
    assert parse_candle({"time": "2026-09-28T07:01:00Z", "isComplete": False, "open": {}, "high": {}, "low": {}, "close": {}}) is None
    assert parse_fill(
        {"orderId": "abc", "lotsRequested": "10", "lotsExecuted": "10", "executedOrderPrice": {"units": "11", "nano": 0}},
        "fallback",
    ) == FillReport("abc", 10, 10, 11.0)
    instrument = parse_instrument(
        {
            "ticker": "CRZ6",
            "uid": "uid-1",
            "figi": "FUT",
            "classCode": "SPBFUT",
            "lot": 1000,
            "basicAsset": "CNYRUB",
            "firstTradeDate": "2026-06-16T00:00:00Z",
            "lastTradeDate": "2026-12-15T00:00:00Z",
        }
    )
    assert instrument is not None
    assert instrument.lsttrade == date(2026, 12, 15)
    assert instrument.lot == 1000
    broker_lot = parse_instrument(
        {**_row(), "lot": 1, "basicAssetSize": {"units": "1000", "nano": 0}}
    )
    assert broker_lot is not None and broker_lot.lot == 1000 and broker_lot.secid == "CRZ6"
    assert parse_instrument({**_row(), "lot": 1, "basicAsset": "CNYRUB"}) is not None
    assert parse_instrument({**_row(), "ticker": "CNYRUBF"}) is None
    assert parse_instrument({**_row(), "lot": 1, "basicAsset": "UCNY"}) is None
    assert parse_instrument({**_row(), "lot": 10}) is None
    previous = Instrument("CRU6", "uid-0", "", date(2026, 3, 17), date(2026, 6, 15), 1000)
    front, trade_from = choose_front([previous, instrument], date(2026, 9, 28))
    assert front.secid == "CRZ6"
    assert trade_from == date(2026, 6, 16)


def _row() -> dict:
    return {
        "ticker": "CRZ6",
        "uid": "uid-1",
        "figi": "FUT",
        "classCode": "SPBFUT",
        "lot": 1000,
        "basicAsset": "CNYRUB",
        "firstTradeDate": "2026-06-16T00:00:00Z",
        "lastTradeDate": "2026-12-15T00:00:00Z",
    }


class FakeBroker:
    def __init__(self) -> None:
        self.instrument = Instrument("CRZ6", "uid-1", "FUT", date(2026, 6, 16), date(2026, 12, 15), 1000)
        self._candles: list = []
        self.margin_value = 5_000.0
        self.lots = 0
        self.avg_price: float | None = None
        self.equity_value = 100_000.0
        self.orders: list[dict] = []
        self.fill_price: float | None = None
        self.executed: int | None = None
        self.fail = False
        self.futures_calls = 0
        self.book_calls = 0
        self.book_error = False
        # 20 контрактов на стороне: авто берёт половину, то есть 10.
        self.bids = 20
        self.asks = 20

    def cny_futures(self):
        self.futures_calls += 1
        return [self.instrument]

    def candles(self, uid, start, end):
        assert uid == self.instrument.uid
        start_m = start.astimezone(MSK)
        end_m = end.astimezone(MSK)
        return [candle for candle in self._candles if start_m <= candle.time.astimezone(MSK) < end_m]

    def margin(self, uid):
        return self.margin_value

    def futures_position(self, account_id, uid):
        return self.lots

    def position_price(self, account_id, uid):
        return self.avg_price

    def equity(self, account_id):
        return self.equity_value

    def order_book(self, uid):
        self.book_calls += 1
        if self.book_error:
            raise RuntimeError("стакан недоступен")
        assert uid == self.instrument.uid
        return {
            "bids": [{"quantity": str(self.bids)}],
            "asks": [{"quantity": str(self.asks)}],
        }

    def market_order(self, account_id, uid, signed, order_id):
        self.orders.append({"signed": signed, "order_id": order_id, "account": account_id, "uid": uid})
        if self.fail:
            raise RuntimeError("timeout")
        executed = abs(signed) if self.executed is None else self.executed
        if executed == abs(signed):
            self.lots += signed
        elif executed:
            self.lots += int(signed / abs(signed) * executed)
        return FillReport(order_id, abs(signed), executed, self.fill_price)


def _candle(moment: datetime, close: float, *, high: float | None = None, low: float | None = None, volume: float = 100):
    from cnyrub.live.broker import Candle

    high = close if high is None else high
    low = close if low is None else low
    return Candle(moment, close, high, low, close, volume)


def _request(**overrides):
    payload = {"strategy": "short", "account_id": ACCOUNT, "token": "t", "channel": 5, "exit_channel": 5, "clock_cap": None}
    payload.update(overrides)
    return parse_event(payload)


def _quiet_bars(start: datetime, count: int, price: float = 10.0) -> list[dict]:
    bars = []
    for i in range(count):
        moment = start + timedelta(minutes=i)
        bars.append({"t": moment.isoformat(), "o": price, "h": price + 0.05, "l": price - 0.05, "c": price, "v": 100})
    return bars


def _ready(store: MemoryStore, bars: list[dict], book: dict | None = None, strategy: str = "short") -> None:
    store.save(
        state_key(strategy, ACCOUNT),
        {
            "version": 1,
            "strategy": strategy,
            "account_id": ACCOUNT,
            "instrument": {
                "secid": "CRZ6",
                "uid": "uid-1",
                "figi": "FUT",
                "frsttrade": "2026-06-16",
                "lsttrade": "2026-12-15",
                "trade_from": "2026-06-16",
            },
            "bars": bars,
            "ready": True,
            "last_bar": bars[-1]["t"],
            "history_before": "2026-09-01",
            "history_walked": 1,
            "halted": None,
            "book": book,
        },
    )


def test_history_loads_one_day_and_does_not_order_until_the_next_call():
    broker = FakeBroker()
    store = MemoryStore()
    day = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    previous = datetime(2026, 9, 27, 10, 0, tzinfo=MSK)
    broker._candles = [_candle(previous + timedelta(minutes=i), 10.0) for i in range(2)]
    broker._candles += [_candle(day + timedelta(minutes=i), 10.0) for i in range(2)]
    request = _request(channel=4, exit_channel=4)
    now = datetime(2026, 9, 28, 12, 0, tzinfo=MSK)
    first = run_minute(request, broker, store, now=now)
    assert first["phase"] == "history"
    assert first["ready"] is False
    assert first["order"] is None
    assert broker.orders == []
    second = run_minute(request, broker, store, now=now)
    assert second["phase"] == "history"
    assert second["ready"] is True
    assert second["bars"] == 4
    assert broker.orders == []
    broker._candles.append(_candle(datetime(2026, 9, 28, 10, 2, tzinfo=MSK), 10.4, high=10.4, low=10.3))
    third = run_minute(request, broker, store, now=datetime(2026, 9, 28, 12, 5, tzinfo=MSK))
    assert third["order"] is None
    assert third["phase"] == "signal"
    assert third["target"] != 0
    assert broker.orders == []


def test_every_window_keeps_ten_sessions_past_its_own_goal():
    extra = 10 * 18 * 60
    for params in PRESETS.values():
        assert bars_to_keep(params) == history_goal(params) + extra
    assert bars_to_keep(PRESETS["long"]) > bars_to_keep(PRESETS["sixty"])
    assert bars_to_keep(PRESETS["short"]) == bars_to_keep(PRESETS["sixty"])


def test_stored_minutes_stop_at_the_window():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 1, 10, 0, tzinfo=MSK)
    count = 11_000
    book = export_book(_FillBook("CRZ6", 100_000.0))
    book["entry_i"] = 100
    _ready(store, _quiet_bars(start, count), book)
    fresh = start + timedelta(minutes=count)
    broker._candles.append(_candle(fresh, 10.0))
    result = run_minute(_request(), broker, store, now=fresh + timedelta(minutes=1))
    saved = store.load(state_key("short", ACCOUNT))
    keep = 5 + 10 * 18 * 60
    assert result["bars"] == keep
    assert len(saved["bars"]) == keep
    assert saved["bars"][-1][0] == fresh.isoformat()
    assert saved["bars"][0][0] == (start + timedelta(minutes=count + 1 - keep)).isoformat()
    assert saved["bars"][-1][1:5] == [10.0, 10.0, 10.0, 10.0]
    dropped = count + 1 - keep
    assert saved["book"]["entry_i"] == 100 - dropped
    assert (keep - 1) - saved["book"]["entry_i"] == count - 100


def test_a_quiet_repeat_does_not_rewrite_the_bucket():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    now = start + timedelta(minutes=6)
    first = run_minute(_request(), broker, store, now=now)
    assert first["phase"] == "idle"
    writes = store.saves
    second = run_minute(_request(), broker, store, now=now)
    assert second["phase"] == "idle"
    assert second["order"] is None
    assert store.saves == writes
    assert broker.futures_calls == 1


def test_api_error_text_prefers_the_exchange_message():
    assert api_error_text(400, '{"message":"instrument not available for trading"}') == (
        "instrument not available for trading"
    )
    assert api_error_text(400, "") == "HTTP 400"
    assert book_lots({"asks": [{"quantity": "3"}, {"quantity": "7"}]}, "asks") == 10
    assert book_lots({"asks": []}, "bids") == 0
    assert half_book(10) == 5
    assert half_book(1) == 0
    assert half_book(0) == 0


def test_a_multi_day_gap_continues_on_the_next_call_and_does_not_order():
    broker = FakeBroker()
    store = MemoryStore()
    monday = datetime(2026, 9, 21, 18, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(monday - timedelta(minutes=4), 5))
    friday = datetime(2026, 9, 25, 12, 0, tzinfo=MSK)
    broker._candles.append(_candle(friday, 10.0))
    now = datetime(2026, 9, 25, 12, 5, tzinfo=MSK)
    request = _request()
    first = run_minute(request, broker, store, now=now)
    assert first["phase"] == "catchup"
    assert first["order"] is None
    assert broker.orders == []
    saved = store.load(state_key("short", ACCOUNT))
    assert datetime.fromisoformat(saved["sync_from"]) > monday
    second = run_minute(request, broker, store, now=now)
    assert second["phase"] == "hold"
    assert second["order"] is None
    assert broker.orders == []
    assert any(bar[0].startswith("2026-09-25T12:00") for bar in store.load(state_key("short", ACCOUNT))["bars"])


def test_breakout_orders_at_most_ten_and_reuses_the_minute_id():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    request = _request()
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    signal = run_minute(request, broker, store, now=start + timedelta(minutes=6))
    assert signal["phase"] == "signal"
    assert signal["order"] is None
    assert broker.orders == []
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    broker.fill_price = 10.8
    filled = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert filled["phase"] == "order"
    order = filled["order"]
    assert order["signed"] == 10
    assert order["executed"] == 10
    assert order["id"] == "short-202609281006"
    assert len(order["id"]) <= 36
    assert order["price"] == 10.8
    assert order["time"] == "2026-09-28 10:06"
    assert order["reason"] == "up"
    assert "pnl" not in order
    saved = store.load(state_key("short", ACCOUNT))
    assert saved["book"]["held"] == 10
    assert saved["book"]["avg"] == pytest.approx(10.8)
    assert saved["book"]["target"] == 19
    assert broker.lots == 10


def test_smaller_pace_is_the_order_size():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    request = _request(fill_per_minute=2)
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(request, broker, store, now=start + timedelta(minutes=6))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    filled = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert filled["order"]["signed"] == 2
    assert store.load(state_key("short", ACCOUNT))["book"]["held"] == 2


def test_auto_takes_half_the_book_and_a_number_is_a_ceiling():
    broker = FakeBroker()
    broker.asks = 5
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(_request(), broker, store, now=start + timedelta(minutes=6))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    thin = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert thin["order"]["signed"] == 2
    assert broker.book_calls == 1

    wide = FakeBroker()
    wide.asks = 100
    wide_store = MemoryStore()
    _ready(wide_store, _quiet_bars(start, 5))
    wide._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(_request(), wide, wide_store, now=start + timedelta(minutes=6))
    wide._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    filled = run_minute(_request(), wide, wide_store, now=start + timedelta(minutes=7))
    assert filled["order"]["signed"] == 19

    capped = FakeBroker()
    capped.asks = 100
    capped.book_error = True
    cap_store = MemoryStore()
    _ready(cap_store, _quiet_bars(start, 5))
    capped._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(_request(fill_per_minute=4), capped, cap_store, now=start + timedelta(minutes=6))
    capped._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    limited = run_minute(_request(fill_per_minute=4), capped, cap_store, now=start + timedelta(minutes=7))
    assert limited["order"]["signed"] == 4
    assert capped.book_calls == 0


def test_pause_keeps_the_signal_and_sends_no_order():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    request = _request(fill_per_minute=0)
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    signal = run_minute(request, broker, store, now=start + timedelta(minutes=6))
    assert signal["phase"] == "paused"
    assert signal["target"] != 0
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    paused = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert paused["phase"] == "paused"
    assert paused["order"] is None
    assert paused["held"] == 0
    assert broker.orders == []
    assert broker.book_calls == 0


def test_unread_book_does_not_trade_and_does_not_halt():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(_request(), broker, store, now=start + timedelta(minutes=6))
    broker.book_error = True
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    missed = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert missed["phase"] == "book"
    assert missed["halted"] is None
    assert missed["order"] is None
    assert broker.orders == []
    assert store.load(state_key("short", ACCOUNT))["book"]["held"] == 0
    broker.book_error = False
    retried = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert retried["order"]["signed"] == 10


def test_position_mismatch_halts_and_reconcile_adopts_the_broker():
    broker = FakeBroker()
    broker.lots = 4
    broker.avg_price = 11.5
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    request = _request()
    halted = run_minute(request, broker, store, now=start + timedelta(minutes=6))
    assert halted["phase"] == "halted"
    assert "4" in halted["halted"]
    assert broker.orders == []
    again = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert again["phase"] == "halted"
    assert broker.orders == []
    adopted = run_minute(_request(reconcile=True), broker, store, now=start + timedelta(minutes=8))
    assert adopted["phase"] == "reconciled"
    assert adopted["held"] == 4
    assert adopted["halted"] is None
    book = store.load(state_key("short", ACCOUNT))["book"]
    assert book["held"] == 4
    assert book["target"] == 4
    assert book["avg"] == pytest.approx(11.5)
    assert book["stop_dist"] == pytest.approx(0.285)
    assert book["stop_px"] == pytest.approx(11.5 - 0.285)
    assert book["entry_i"] == 4
    assert book["base"] == 4
    assert broker.orders == []


def test_partial_fill_restores_the_book_and_halts():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    request = _request()
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(request, broker, store, now=start + timedelta(minutes=6))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    broker.executed = 3
    halted = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert halted["phase"] == "halted"
    assert "3" in halted["halted"]
    book = store.load(state_key("short", ACCOUNT))["book"]
    assert book["held"] == 0
    assert book["target"] != 0
    assert len(broker.orders) == 1


def test_missed_bar_stop_sends_one_reduce_and_keeps_open_equity():
    broker = FakeBroker()
    broker.lots = 12
    broker.equity_value = 999_999.0
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    book = _FillBook("CRZ6", 50_000.0)
    book.held = 12
    book.target = 12
    book.avg = 10.0
    book.stop_dist = 0.2
    book.stop_px = 9.8
    book.base = 12
    book.entry_i = 1
    _ready(store, _quiet_bars(start, 5), export_book(book))
    broker._candles.append(_candle(start + timedelta(minutes=5), 9.4, high=9.6, low=9.0))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.0, high=10.1, low=9.9))
    result = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert result["phase"] == "order"
    assert result["order"]["signed"] == -10
    assert result["order"]["reason"] == "stop"
    assert result["order"]["pnl"] == pytest.approx(-10)
    assert len(broker.orders) == 1
    saved = store.load(state_key("short", ACCOUNT))["book"]
    assert saved["held"] == 2
    assert saved["equity"] == pytest.approx(50_000.0)
    assert saved["reason"] == "stop"


def test_failed_order_does_not_keep_the_fill():
    broker = FakeBroker()
    broker.fail = True
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(_request(), broker, store, now=start + timedelta(minutes=6))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    halted = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert halted["phase"] == "halted"
    assert store.load(state_key("short", ACCOUNT))["book"]["held"] == 0


def test_two_accounts_keep_separate_state():
    assert state_key("short", "one") != state_key("long", "two")


def test_closing_a_future_reports_the_trade_result():
    broker = FakeBroker()
    broker.lots = 10
    broker.margin_value = 5_000.0
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    book = _FillBook("CRZ6", 100_000.0)
    book.held = 10
    book.target = 10
    book.avg = 10.0
    book.stop_dist = 0.2
    book.stop_px = 9.8
    book.base = 10
    book.entry_i = 1
    _ready(store, _quiet_bars(start, 5), export_book(book))
    broker._candles.append(_candle(start + timedelta(minutes=5), 9.4, high=9.6, low=9.0))
    result = run_minute(_request(), broker, store, now=start + timedelta(minutes=6))
    assert result["order"]["signed"] == -10
    assert result["order"]["reason"] == "stop"
    assert result["order"]["pnl"] == pytest.approx(-6010)
    assert broker.lots == 0
    assert [item["uid"] for item in broker.orders] == [broker.instrument.uid]


def test_last_evening_bar_does_not_open_into_expiry():
    broker = FakeBroker()
    broker.instrument = Instrument("CRZ6", "uid-1", "FUT", date(2026, 6, 16), date(2026, 9, 29), 1000)
    store = MemoryStore()
    start = datetime(2026, 9, 28, 23, 40, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    broker._candles.append(_candle(datetime(2026, 9, 28, 23, 49, tzinfo=MSK), 10.4, high=10.45, low=10.3, volume=200))
    result = run_minute(_request(), broker, store, now=datetime(2026, 9, 28, 23, 50, tzinfo=MSK))
    assert result["order"] is None
    assert result["target"] == 0
    assert broker.orders == []

    earlier = FakeBroker()
    earlier.instrument = broker.instrument
    earlier_store = MemoryStore()
    _ready(earlier_store, _quiet_bars(start, 5))
    earlier._candles.append(_candle(datetime(2026, 9, 28, 23, 48, tzinfo=MSK), 10.4, high=10.45, low=10.3, volume=200))
    opened = run_minute(_request(), earlier, earlier_store, now=datetime(2026, 9, 28, 23, 49, tzinfo=MSK))
    assert opened["phase"] == "signal"
    assert opened["target"] != 0


def test_following_day_flattens_only_at_the_end_of_expiry():
    last = date(2026, 12, 15)
    assert following_day(datetime(2026, 12, 15, 23, 39, tzinfo=MSK), last) == last
    assert following_day(datetime(2026, 12, 15, 23, 40, tzinfo=MSK), last) is None
    assert following_day(datetime(2026, 12, 15, 23, 49, tzinfo=MSK), last) is None
    assert following_day(datetime(2026, 12, 14, 23, 48, tzinfo=MSK), last) == date(2026, 12, 14)
    # 23:49 — последняя свеча вечера, следующая минута уже день экспирации.
    assert following_day(datetime(2026, 12, 14, 23, 49, tzinfo=MSK), last) == last
    # Пятница 11 декабря: следующая сессия — понедельник 14-го, не суббота.
    monday = date(2026, 12, 14)
    assert following_day(datetime(2026, 12, 11, 23, 49, tzinfo=MSK), last) == monday
    assert following_day(datetime(2026, 12, 11, 23, 49, tzinfo=MSK), monday) == monday


def test_reprice_open_add_reduce_and_close():
    book = _FillBook("CRZ6", 100_000.0)
    book.target = 10
    before = export_book(book)
    book.move(10.0, 1, 10)
    reprice_fill(book, before, 10.0, 10.5)
    assert book.held == 10
    assert book.avg == pytest.approx(10.5)
    assert book.best == pytest.approx(10.5)

    before = export_book(book)
    book.target = 20
    book.move(11.0, 2, 10)
    reprice_fill(book, before, 11.0, 12.0)
    assert book.avg == pytest.approx((10.5 * 10 + 12 * 10) / 20)

    before = export_book(book)
    book.target = 10
    book.move(13.0, 3, 10)
    gross_at_close = book.gross
    reprice_fill(book, before, 13.0, 14.0)
    assert book.held == 10
    assert book.gross == pytest.approx(gross_at_close + (14.0 - 13.0) * 10 * 1000)

    before = export_book(book)
    book.target = 0
    book.move(13.0, 4, 10)
    priced_at_close = book.equity
    pnl_at_close = float(book.trades[-1]["pnl"])
    reprice_fill(book, before, 13.0, 15.0)
    assert book.held == 0
    assert book.equity == pytest.approx(priced_at_close + 20_000)
    assert book.trades[-1]["pnl"] == pytest.approx(pnl_at_close + 20_000)


def test_step_minute_pace_override_does_not_change_the_default():
    def run(pace):
        book = _FillBook("CRZ6", 100_000.0)
        book.target = 25
        step_minute(
            book,
            1,
            opened=10,
            high=10,
            low=10,
            close=10,
            volume=100,
            day=date(2026, 9, 28),
            next_day=date(2026, 9, 28),
            last_day=date(2026, 12, 15),
            prior_high=float("nan"),
            prior_low=float("nan"),
            prior_vol=float("nan"),
            prior_range=float("nan"),
            exit_high=float("nan"),
            exit_low=float("nan"),
            clock_vol=float("nan"),
            entry_ready=0,
            clearance=0,
            cooldown=0,
            clock_volume=False,
            clock_cap=None,
            size_mode="flat",
            stop_mult=None,
            loss_bars=None,
            risk_fraction=None,
            margin=1000,
            stop_rub=None,
            breakout_span=None,
            trade_from=date(2026, 6, 16),
            trail=False,
            fill_per_minute=pace,
        )
        return book.held

    assert run(None) == 10
    assert run(2) == 2
    assert run(0) == 0

    def run_last(pace):
        book = _FillBook("CRZ6", 100_000.0)
        book.held = 25
        book.target = 25
        book.entry_i = 0
        step_minute(
            book,
            1,
            opened=10,
            high=10,
            low=10,
            close=10,
            volume=100,
            day=date(2026, 12, 15),
            next_day=None,
            last_day=date(2026, 12, 15),
            prior_high=float("nan"),
            prior_low=float("nan"),
            prior_vol=float("nan"),
            prior_range=float("nan"),
            exit_high=float("nan"),
            exit_low=float("nan"),
            clock_vol=float("nan"),
            entry_ready=0,
            clearance=0,
            cooldown=0,
            clock_volume=False,
            clock_cap=None,
            size_mode="flat",
            stop_mult=None,
            loss_bars=None,
            risk_fraction=None,
            margin=1000,
            stop_rub=None,
            breakout_span=None,
            trade_from=date(2026, 6, 16),
            trail=False,
            fill_per_minute=pace,
        )
        return book.held

    assert run_last(None) == 0
    assert run_last(4) == 21
    assert run_last(0) == 25


def test_handler_logs_the_same_json_it_returns(capsys):
    result = handle(
        {"strategy": "short", "account_id": ACCOUNT, "token": "secret-token", "channel": 3, "exit_channel": 0, "clock_cap": None},
        store=MemoryStore(),
        now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK),
        broker_factory=lambda token: FakeBroker(),
    )
    logged = capsys.readouterr().out
    assert "secret-token" not in logged
    assert json.loads(logged) == result


def test_handler_uses_the_token_only_to_build_the_broker():
    seen: list[str] = []

    class Quiet(FakeBroker):
        pass

    def factory(token: str):
        seen.append(token)
        broker = Quiet()
        broker._candles = []
        return broker

    result = handle(
        {"strategy": "short", "account_id": ACCOUNT, "secret_id": "box", "channel": 3, "exit_channel": 0, "clock_cap": None},
        store=MemoryStore(),
        now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK),
        secret_reader=lambda secret_id: "from-lockbox",
        broker_factory=factory,
    )
    assert seen == ["from-lockbox"]
    assert "token" not in result
    assert "from-lockbox" not in str(result)
    assert result["phase"] == "history"


def test_handler_tells_telegram_about_the_fill_and_survives_a_notifier_error():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    event = {
        "strategy": "short",
        "account_id": ACCOUNT,
        "token": "t",
        "channel": 5,
        "exit_channel": 5,
        "clock_cap": None,
    }
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    signal = handle(event, store=store, now=start + timedelta(minutes=6), broker_factory=lambda token: broker)
    assert signal["order"] is None
    assert "telegram" not in signal
    sent: list[dict[str, object]] = []
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    broker.fill_price = 10.8
    filled = handle(
        event,
        store=store,
        now=start + timedelta(minutes=7),
        broker_factory=lambda token: broker,
        notifier=sent.append,
    )
    assert filled["phase"] == "order"
    assert sent == [filled]
    assert trade_text(filled) == (
        "стратегия short\nCRZ6 2026-09-28 10:06\nпокупка 10 по 10.8, пробой вверх\nпозиция 10, цель 19"
    )

    def broken(result: dict[str, object]) -> None:
        raise RuntimeError("bot 123456:secret https://api.telegram.org/bot123456:secret/sendMessage")

    again = handle(
        event,
        store=store,
        now=start + timedelta(minutes=7),
        broker_factory=lambda token: broker,
        notifier=broken,
    )
    assert again["telegram"] == "не отправлено"
    assert "secret" not in str(again)
    assert "api.telegram.org" not in str(again)


def test_order_reason_separates_the_breakout_from_the_next_chunk_and_the_return():
    book = _FillBook("CRZ6", 100_000.0)
    book.entry = "up"
    book.target = 20
    opened = export_book(book)
    assert order_reason(book, opened, 10) == "up"
    book.held = 10
    adding = export_book(book)
    assert order_reason(book, adding, 10) == "add"
    book.scaled = True
    book.scale_level = 0
    book.held = 10
    book.target = 20
    restored = export_book(book)
    assert order_reason(book, restored, 10) == "scale_back"
    book.held = 0
    book.trades = [{"reason": "stop"}]
    cover = export_book(book)
    cover["held"] = -10
    assert order_reason(book, cover, 10) == "stop"


def test_close_pnl_is_the_chunk_until_the_position_is_flat():
    book = _FillBook("CRZ6", 100_000.0)
    book.held = 10
    book.avg = 10.0
    book.commission = 10.0
    before = export_book(book)
    book.held = 6
    book.gross = 4_000.0
    book.commission = 14.0
    assert close_pnl(book, before, -4) == pytest.approx(3_996)
    flat = _FillBook("CRZ6", 100_000.0)
    flat.trades = [{"pnlcomm": -6_010.0}]
    assert close_pnl(flat, before, -10) == pytest.approx(-6_010)
    assert close_pnl(book, before, 4) is None


def test_order_id_is_stable_for_the_minute():
    moment = datetime(2026, 9, 28, 10, 6, tzinfo=MSK)
    assert make_order_id("long", moment) == make_order_id("long", moment)
    assert make_order_id("long", moment) == "long-202609281006"


def test_yandex_zip_contains_only_the_cloud_function(tmp_path):
    import zipfile

    from function.pack import FILES, ZIP_PATH, build

    fresh = tmp_path / "fresh.zip"
    extract = tmp_path / "extract"
    build(fresh)
    with zipfile.ZipFile(fresh) as built, zipfile.ZipFile(ZIP_PATH) as stored:
        assert built.namelist() == stored.namelist() == list(FILES)
        for name in FILES:
            assert built.read(name) == stored.read(name)
            assert "backtest" not in name and "pandas" not in name
        text = stored.read("requirements.txt").decode("utf-8")
        assert "boto3" in text
        assert "backtrader" not in text and "pandas" not in text
        stored.extractall(extract)
    code = (
        "import function.index, sys; "
        "assert callable(function.index.handler); "
        "assert 'backtrader' not in sys.modules; "
        "assert 'pandas' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd=extract)


def test_live_package_does_not_import_backtrader_or_pandas():
    code = (
        "import cnyrub.live.handler, cnyrub.live.service, cnyrub.engine; "
        "import sys; "
        "assert 'backtrader' not in sys.modules; "
        "assert 'pandas' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd="/workspace")
