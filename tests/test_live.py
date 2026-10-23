"""Живой контур без сети: разбор событий, история, заявка, стоп и сверка."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest import WINDOWS, channel_view
from cnyrub.engine import COMMISSION, _FillBook, export_book, reprice_fill, step_minute
from cnyrub.live.broker import (
    TinkoffClient,
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
    fill_price,
    make_order_id,
    order_reason,
    run_minute,
)
from cnyrub.live.vk import status_text, trade_text
from cnyrub.live.state import MemoryStore, state_key
from cnyrub.live.veto import held_trading_days, peer_key, publish_peer, veto_source

MSK = ZoneInfo("Europe/Moscow")
ACCOUNT = "acc-short"


def test_presets_match_the_researched_windows():
    short, long, thirty, fortyfive, sixty, m1760, m2160 = WINDOWS
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
    assert PRESETS["m1760"].channel == m1760.channel == 1760
    assert PRESETS["m1760"].exit_channel == 0
    assert PRESETS["m1760"].stop_mult == m1760.stop_mult == 22
    assert PRESETS["m1760"].clock_cap is None and m1760.clock_cap is None
    assert PRESETS["m1760"].surge_cap == m1760.surge_cap == 5
    assert PRESETS["m1760"].leverage == m1760.leverage == 5
    assert PRESETS["m1760"].cooldown == m1760.cooldown == 60
    assert PRESETS["m1760"].min_stop == m1760.min_stop == pytest.approx(0.008)
    assert PRESETS["m1760"].skip_flat is True and m1760.skip_flat is True
    assert m1760.cash == 100_000
    assert PRESETS["m2160"].channel == m2160.channel == 2160
    assert PRESETS["m2160"].exit_channel == 0
    assert PRESETS["m2160"].stop_mult == m2160.stop_mult == 22
    assert PRESETS["m2160"].surge_cap == m2160.surge_cap == 8
    assert PRESETS["m2160"].leverage == m2160.leverage == 4
    assert PRESETS["m2160"].cooldown == m2160.cooldown == 60
    assert PRESETS["m2160"].min_stop == pytest.approx(0.008)
    assert PRESETS["m2160"].skip_flat is True and m2160.skip_flat is True
    assert m2160.cash == 100_000
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
    assert direct.probe is False
    probed = parse_event({"strategy": "thirty", "account_id": "42", "secret_id": "box", "probe": True})
    assert probed.probe is True
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


def test_parse_requires_an_account_but_not_a_token():
    with pytest.raises(ValueError):
        parse_event({"strategy": "short", "token": "t"})
    bare = parse_event({"strategy": "short", "account_id": "1"})
    assert bare.token is None and bare.secret_id is None
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
    day = datetime(2026, 9, 29, 10, 0, tzinfo=MSK)
    previous = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    broker._candles = [_candle(previous + timedelta(minutes=i), 10.0) for i in range(2)]
    broker._candles += [_candle(day + timedelta(minutes=i), 10.0) for i in range(2)]
    request = _request(channel=4, exit_channel=4)
    now = datetime(2026, 9, 29, 12, 0, tzinfo=MSK)
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
    broker._candles.append(_candle(datetime(2026, 9, 29, 10, 2, tzinfo=MSK), 10.4, high=10.4, low=10.3))
    third = run_minute(request, broker, store, now=datetime(2026, 9, 29, 12, 5, tzinfo=MSK))
    assert third["order"] is None
    assert third["phase"] == "signal"
    assert third["target"] != 0
    assert broker.orders == []


def test_weekend_minutes_are_skipped_unless_asked():
    friday = datetime(2026, 10, 16, 10, 0, tzinfo=MSK)
    saturday = datetime(2026, 10, 17, 10, 0, tzinfo=MSK)
    now = datetime(2026, 10, 17, 12, 0, tzinfo=MSK)

    def run(**overrides):
        broker = FakeBroker()
        store = MemoryStore()
        _ready(store, _quiet_bars(friday, 5))
        broker._candles.append(_candle(saturday, 10.4, high=10.45, low=10.3))
        result = run_minute(_request(**overrides), broker, store, now=now)
        return result, store.load(state_key("short", ACCOUNT))

    quiet, saved = run()
    assert quiet["phase"] == "idle"
    assert len(saved["bars"]) == 5
    busy, saved = run(weekends=True)
    assert busy["phase"] == "signal"
    assert len(saved["bars"]) == 6


def test_thin_market_guards_are_on_for_the_stop_windows():
    for name in ("thirty", "fortyfive", "sixty"):
        params = PRESETS[name]
        assert params.min_stop == pytest.approx(0.008)
        assert params.cooldown == 15
        assert params.skip_flat is True
        assert params.weekends is False
    assert PRESETS["long"].min_stop == pytest.approx(0.008)
    assert PRESETS["long"].skip_flat is True
    assert PRESETS["long"].cooldown == 0
    assert PRESETS["short"].min_stop is None and PRESETS["short"].cooldown == 0
    for name in ("m1760", "m2160"):
        params = PRESETS[name]
        assert params.min_stop == pytest.approx(0.008)
        assert params.cooldown == 60
        assert params.skip_flat is True
        assert params.weekends is False
    request = parse_event(
        {
            "strategy": "sixty",
            "account_id": ACCOUNT,
            "token": "t",
            "min_stop": 0.012,
            "cooldown": 0,
            "skip_flat": False,
            "weekends": True,
        }
    )
    assert request.params.min_stop == pytest.approx(0.012)
    assert request.params.cooldown == 0
    assert request.params.skip_flat is False
    assert request.params.weekends is True


def test_every_window_keeps_ten_sessions_past_its_own_goal():
    extra = 10 * 18 * 60
    for params in PRESETS.values():
        assert bars_to_keep(params) == history_goal(params) + extra
    assert bars_to_keep(PRESETS["long"]) > bars_to_keep(PRESETS["short"])
    assert bars_to_keep(PRESETS["m2160"]) == 2_160 + extra
    assert bars_to_keep(PRESETS["m1760"]) == 1_760 + extra
    assert bars_to_keep(PRESETS["m2160"]) > bars_to_keep(PRESETS["m1760"])
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
    assert order["id"] == make_order_id("short", start + timedelta(minutes=6))
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


def _breakout_ready():
    broker = FakeBroker()
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 5))
    request = _request()
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    run_minute(request, broker, store, now=start + timedelta(minutes=6))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    return broker, store, request, start


def test_partial_fill_is_accepted_and_the_rest_is_taken_next_minute():
    broker, store, request, start = _breakout_ready()
    broker.executed = 3
    filled = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert filled["phase"] == "order"
    assert filled["halted"] is None
    assert "halt_new" not in filled
    assert filled["order"]["requested"] == 10
    assert filled["order"]["executed"] == 3
    assert filled["order"]["signed"] == 3
    assert filled["held"] == 3
    assert filled["target"] > 3
    book = store.load(state_key("short", ACCOUNT))["book"]
    assert book["held"] == 3 == broker.lots
    assert book["commission"] == pytest.approx(3 * COMMISSION)

    broker.executed = None
    broker._candles.append(_candle(start + timedelta(minutes=7), 10.5, high=10.55, low=10.4))
    more = run_minute(request, broker, store, now=start + timedelta(minutes=8))
    assert more["phase"] == "order"
    assert more["held"] == broker.lots
    assert more["order"]["executed"] == more["order"]["requested"]


def test_partial_close_keeps_the_remaining_position_and_its_stop():
    broker = FakeBroker()
    broker.lots = 10
    broker.executed = 4
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
    book.opened = 10
    book.peak = 10
    book.commission = 10 * COMMISSION
    _ready(store, _quiet_bars(start, 5), export_book(book))
    broker._candles.append(_candle(start + timedelta(minutes=5), 9.4, high=9.6, low=9.0))
    result = run_minute(_request(), broker, store, now=start + timedelta(minutes=6))
    assert result["phase"] == "order"
    assert result["order"]["requested"] == 10
    assert result["order"]["executed"] == 4
    assert result["held"] == 6 == broker.lots
    assert result["order"]["pnl"] == pytest.approx(-0.6 * 4 * 1000 - 4 * COMMISSION)
    saved = store.load(state_key("short", ACCOUNT))["book"]
    assert saved["held"] == 6
    assert saved["target"] == 0
    assert saved["stop_px"] == pytest.approx(9.8)

    broker.executed = None
    broker._candles.append(_candle(start + timedelta(minutes=6), 9.4, high=9.5, low=9.3))
    last = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert last["held"] == 0 == broker.lots
    assert last["order"]["pnl"] is not None


def test_zero_fill_does_not_halt_and_the_next_minute_decides_again():
    broker, store, request, start = _breakout_ready()
    broker.executed = 0
    first = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert first["phase"] == "unfilled"
    assert first["unfilled"] == 1
    assert first["halted"] is None
    assert "halt_new" not in first
    saved = store.load(state_key("short", ACCOUNT))
    assert saved["zero_fills"] == 1
    assert saved["halted"] is None
    assert saved["book"]["held"] == 0
    assert saved["last_bar"] != _bar_stamp(start + timedelta(minutes=6))
    assert len(broker.orders) == 1

    broker.executed = None
    broker._candles.append(_candle(start + timedelta(minutes=7), 10.6, high=10.65, low=10.5))
    filled = run_minute(request, broker, store, now=start + timedelta(minutes=8))
    assert filled["phase"] == "order"
    assert filled["order"]["executed"] == filled["order"]["requested"]
    assert store.load(state_key("short", ACCOUNT))["zero_fills"] == 0


def _bar_stamp(moment: datetime) -> str:
    return moment.isoformat()


def test_ten_zero_fills_in_a_row_halt_the_strategy():
    from cnyrub.live.service import ZERO_FILL_LIMIT

    assert ZERO_FILL_LIMIT == 10
    broker, store, request, start = _breakout_ready()
    broker.executed = 0
    for n in range(1, 10):
        result = run_minute(request, broker, store, now=start + timedelta(minutes=6 + n))
        assert result["phase"] == "unfilled", n
        assert result["unfilled"] == n
        broker._candles.append(_candle(start + timedelta(minutes=6 + n), 10.5 + n * 0.1, high=10.6 + n * 0.1, low=10.4 + n * 0.1))
    halted = run_minute(request, broker, store, now=start + timedelta(minutes=16))
    assert halted["phase"] == "halted"
    assert halted["halted"] == "заявка не исполнена 10 раз подряд"
    assert halted["halt_new"] is True
    assert len(broker.orders) == 10
    again = run_minute(request, broker, store, now=start + timedelta(minutes=17))
    assert again["phase"] == "halted"
    assert "halt_new" not in again
    assert len(broker.orders) == 10

    broker.executed = None
    broker.lots = 0
    reconciled = run_minute(parse_event({**_event_for(request), "reconcile": True}), broker, store, now=start + timedelta(minutes=18))
    assert reconciled["phase"] == "reconciled"
    assert store.load(state_key("short", ACCOUNT))["zero_fills"] == 0


def _event_for(request) -> dict:
    return {"strategy": request.strategy, "account_id": request.account_id, "token": "t", "channel": 5, "exit_channel": 5, "clock_cap": None}


def test_a_filled_order_resets_the_zero_fill_counter():
    broker, store, request, start = _breakout_ready()
    broker.executed = 0
    run_minute(request, broker, store, now=start + timedelta(minutes=7))
    run_minute(request, broker, store, now=start + timedelta(minutes=8))
    assert store.load(state_key("short", ACCOUNT))["zero_fills"] == 2
    broker.executed = None
    broker._candles.append(_candle(start + timedelta(minutes=7), 10.6, high=10.65, low=10.5))
    run_minute(request, broker, store, now=start + timedelta(minutes=9))
    assert store.load(state_key("short", ACCOUNT))["zero_fills"] == 0


def _failing_event(**extra):
    return {"strategy": "short", "account_id": ACCOUNT, "token": "secret-token", "channel": 3, "exit_channel": 0, "clock_cap": None, **extra}


def test_a_failed_run_writes_to_vk_once_a_day_and_a_restart_resets_it(capsys):
    class Broken(FakeBroker):
        def cny_futures(self):
            raise RuntimeError("API недоступно, секрет secret-token")

        def candles(self, uid, start, end):
            raise RuntimeError("API недоступно, секрет secret-token")

    store = MemoryStore()
    sent: list[dict] = []
    noon = datetime(2026, 9, 28, 12, 0, tzinfo=MSK)

    def failing(event, when):
        with pytest.raises(RuntimeError):
            handle(event, store=store, now=when, broker_factory=lambda token: Broken(), notifier=sent.append)

    failing(_failing_event(), noon)
    failing(_failing_event(), noon + timedelta(minutes=1))
    assert len(sent) == 1
    assert sent[0]["phase"] == "error" and sent[0]["error_new"] is True
    assert "secret-token" not in json.dumps(sent[0])
    logged = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(logged) == 2
    assert "secret-token" not in json.dumps(logged)

    failing(_failing_event(), noon + timedelta(days=1))
    assert len(sent) == 2

    handle(_failing_event(reconcile=True), store=store, now=noon + timedelta(days=1), broker_factory=lambda token: FakeBroker(), notifier=sent.append)
    sent.clear()
    failing(_failing_event(), noon + timedelta(days=1, minutes=5))
    assert len(sent) == 1


def test_a_hung_run_is_cut_off_and_names_the_stage(monkeypatch, capsys):
    import time

    class Hanging(FakeBroker):
        def cny_futures(self):
            time.sleep(30)

    monkeypatch.setenv("RUN_DEADLINE", "1")
    sent: list[dict] = []
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="этап: минута стратегии"):
        handle(
            _failing_event(),
            store=MemoryStore(),
            now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK),
            broker_factory=lambda token: Hanging(),
            notifier=sent.append,
        )
    assert time.monotonic() - started < 5
    assert len(sent) == 1
    assert "этап: минута стратегии" in sent[0]["error"]
    assert "secret-token" not in capsys.readouterr().out


def test_a_normal_run_reports_its_duration(capsys):
    result = handle(
        _failing_event(),
        store=MemoryStore(),
        now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK),
        broker_factory=lambda token: FakeBroker(),
    )
    assert isinstance(result["sec"], float)
    assert json.loads(capsys.readouterr().out)["sec"] == result["sec"]


def test_error_alert_survives_a_dead_bucket():
    from cnyrub.live import handler as handler_module

    class DeadStore:
        def load(self, key):
            raise RuntimeError("bucket down")

        def save(self, key, document):
            raise RuntimeError("bucket down")

    handler_module._ALERTED.clear()
    sent: list[dict] = []
    noon = datetime(2026, 9, 28, 12, 0, tzinfo=MSK)
    for minute in (0, 1):
        with pytest.raises(RuntimeError):
            handle(_failing_event(), store=DeadStore(), now=noon + timedelta(minutes=minute), broker_factory=lambda token: FakeBroker(), notifier=sent.append)
    assert len(sent) == 1


def test_overfill_halts():
    broker, store, request, start = _breakout_ready()
    broker.executed = 11
    halted = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert halted["phase"] == "halted"
    assert "больше заявленного" in halted["halted"]


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
    assert result["order"]["pnl"] == pytest.approx(-10 * COMMISSION)
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
    assert result["order"]["pnl"] == pytest.approx(-6000 - 10 * COMMISSION)
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
        book.target = 250
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

    assert run(None) == 100
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


def test_probe_sends_the_saved_status_and_does_not_trade(capsys):
    store = MemoryStore()
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    _ready(store, _quiet_bars(start, 4), {"held": -3, "target": -3}, strategy="thirty")
    saved = store.load(state_key("thirty", ACCOUNT))
    saved["halted"] = "позиция на счёте 0, в книге -3"
    store.save(state_key("thirty", ACCOUNT), saved)
    sent: list[dict[str, object]] = []

    def factory(token: str):
        raise AssertionError(token)

    result = handle(
        {"strategy": "thirty", "account_id": ACCOUNT, "token": "secret-token", "probe": True},
        store=store,
        broker_factory=factory,
        status_notifier=sent.append,
    )
    assert result["phase"] == "probe"
    assert result["secid"] == "CRZ6"
    assert result["bars"] == 4
    assert result["held"] == -3
    assert result["target"] == -3
    assert result["halted"] == "позиция на счёте 0, в книге -3"
    assert result["order"] is None
    assert sent == [result]
    assert "secret-token" not in capsys.readouterr().out
    assert store.saves == 2


def test_probe_still_answers_when_the_book_cannot_be_read(capsys):
    class Broken:
        def load(self, key: str) -> dict[str, object] | None:
            raise TimeoutError(key)

        def save(self, key: str, document: dict[str, object]) -> None:
            raise AssertionError(key)

    sent: list[dict[str, object]] = []

    def factory(token: str):
        raise AssertionError(token)

    result = handle(
        {"strategy": "thirty", "account_id": ACCOUNT, "token": "secret-token", "probe": True},
        store=Broken(),
        broker_factory=factory,
        status_notifier=sent.append,
    )
    assert result["phase"] == "probe"
    assert result["state"] == "не прочитано"
    assert result["order"] is None
    assert sent == [result]
    assert "книга не прочитана" in status_text(result)
    assert "secret-token" not in capsys.readouterr().out


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


def test_handler_logs_a_failed_run_and_reraises(capsys):
    class Broken(FakeBroker):
        def cny_futures(self):
            raise RuntimeError("API недоступно, секрет secret-token")

    with pytest.raises(RuntimeError):
        handle(
            {"strategy": "short", "account_id": ACCOUNT, "token": "secret-token", "channel": 3, "exit_channel": 0, "clock_cap": None},
            store=MemoryStore(),
            now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK),
            broker_factory=lambda token: Broken(),
        )
    logged = json.loads(capsys.readouterr().out)
    assert logged["phase"] == "error"
    assert "secret-token" not in json.dumps(logged)


def test_client_stops_when_the_time_budget_is_spent():
    import cnyrub.live.broker as broker_module

    client = TinkoffClient("t")
    client._deadline = broker_module.time.monotonic() - 1
    with pytest.raises(RuntimeError, match="лимит времени"):
        client._call("OrdersService", "GetOrders", {})


def test_timer_payload_without_a_token_uses_the_function_secret(monkeypatch, capsys):
    seen: list[str] = []

    def factory(token: str):
        seen.append(token)
        return FakeBroker()

    event = {"strategy": "short", "account_id": ACCOUNT, "channel": 3, "exit_channel": 0, "clock_cap": None}
    monkeypatch.setenv("TOKEN", "  env-secret  ")
    result = handle(event, store=MemoryStore(), now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK), broker_factory=factory)
    assert seen == ["env-secret"]
    assert "env-secret" not in json.dumps(result) and "env-secret" not in capsys.readouterr().out

    seen.clear()
    handle({**event, "token": "field-token"}, store=MemoryStore(), now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK), broker_factory=factory)
    assert seen == ["field-token"]


def test_a_run_without_any_token_source_fails_clearly(monkeypatch):
    monkeypatch.delenv("TOKEN", raising=False)
    event = {"strategy": "short", "account_id": ACCOUNT}
    with pytest.raises(ValueError, match="TOKEN"):
        handle(event, store=MemoryStore(), now=datetime(2026, 9, 28, 12, 0, tzinfo=MSK), notifier=lambda result: None)


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


def test_handler_tells_max_about_the_fill_and_survives_a_notifier_error():
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
    assert "vk" not in signal
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
        raise RuntimeError("bot 123456:secret https://api.vk.com/messages")

    again = handle(
        event,
        store=store,
        now=start + timedelta(minutes=7),
        broker_factory=lambda token: broker,
        notifier=broken,
    )
    assert again["vk"] == "не отправлено"
    assert "secret" not in str(again)
    assert "api.vk.com" not in str(again)


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
    assert make_order_id("long", moment) != make_order_id("short", moment)
    assert make_order_id("long", moment) != make_order_id("long", moment + timedelta(minutes=1))
    assert str(uuid.UUID(make_order_id("long", moment))) == make_order_id("long", moment)


def test_fill_price_accepts_points_and_rejects_wrong_scale():
    assert fill_price(10.8, 10.75) == 10.8
    assert fill_price(10_800.0, 10.75) == pytest.approx(10.8)
    assert fill_price(1.0, 10.75) == 10.75
    assert fill_price(None, 10.75) == 10.75
    assert fill_price(0.0, 10.75) == 10.75


def _peer(name: str, *, channel: int, held: int, days: int, pnl: float) -> dict[str, object]:
    return {
        "strategy": name,
        "channel": channel,
        "held": held,
        "target": held,
        "held_days": days,
        "pnl": pnl,
    }


def test_held_days_count_dates_after_the_entry_and_add_one_when_it_fell_out():
    bars = [
        {"t": "2026-09-28T10:00:00+03:00", "c": 10},
        {"t": "2026-09-28T10:01:00+03:00", "c": 10},
        {"t": "2026-09-29T10:00:00+03:00", "c": 10},
        {"t": "2026-09-30T10:00:00+03:00", "c": 10},
    ]
    assert held_trading_days(bars, 0) == 2
    assert held_trading_days(bars, 2) == 1
    assert held_trading_days(bars, -1) == 3
    assert held_trading_days(bars[:2], -1) == 1
    assert held_trading_days([], 0) == 0


def test_only_a_longer_green_position_blocks_the_opposite_entry():
    store = MemoryStore()
    book = _FillBook("CRZ6", 100_000.0)
    book.target = 10
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=2, pnl=1500))
    store.save(peer_key("thirty"), _peer("thirty", channel=30, held=-8, days=2, pnl=1500))
    assert veto_source(store, "short", 525, book) == "long"
    book.target = -10
    assert veto_source(store, "short", 525, book) == ""
    book.held = -1
    book.target = 10
    assert veto_source(store, "short", 525, book) == ""
    book.held = 0
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=1, pnl=1500))
    assert veto_source(store, "short", 525, book) == ""
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=2, pnl=0))
    assert veto_source(store, "short", 525, book) == ""
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=2, pnl=1500))
    store.save(peer_key("fortyfive"), {"strategy": "other", "channel": 45, "held": -8, "target": -8, "held_days": 2, "pnl": 1})
    assert veto_source(store, "short", 525, book) == "long"

    class Broken:
        def load(self, key: str):
            raise RuntimeError(key)

    assert veto_source(Broken(), "short", 525, book) == ""
    store.save(peer_key("m1760"), _peer("m1760", channel=1760, held=-8, days=2, pnl=1500))
    store.save(peer_key("m2160"), _peer("m2160", channel=2160, held=-8, days=2, pnl=1500))
    book.target = 10
    assert veto_source(store, "sixty", 60, book) == "long, m1760, m2160"
    assert veto_source(store, "m1760", 1760, book) == "long, m2160"
    assert veto_source(store, "m2160", 2160, book) == "long"
    assert veto_source(store, "long", 12420, book) == ""


def test_a_senior_short_in_profit_blocks_a_fresh_long_and_keeps_a_short():
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    broker = FakeBroker()
    store = MemoryStore()
    _ready(store, _quiet_bars(start, 5))
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=2, pnl=1500))
    request = _request()
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    blocked = run_minute(request, broker, store, now=start + timedelta(minutes=6))
    assert blocked["phase"] == "veto"
    assert blocked["veto"] == "long"
    assert blocked["order"] is None
    assert blocked["target"] == 0
    assert broker.orders == []
    assert store.load(state_key("short", ACCOUNT))["book"]["target"] == 0
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    still = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert still["phase"] == "veto"
    assert broker.orders == []

    same = FakeBroker()
    same_store = MemoryStore()
    _ready(same_store, _quiet_bars(start, 5))
    same_store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=2, pnl=1500))
    same._candles.append(_candle(start + timedelta(minutes=5), 9.8, high=9.9, low=9.7))
    opened = run_minute(request, same, same_store, now=start + timedelta(minutes=6))
    assert opened["phase"] == "signal"
    assert opened["target"] < 0
    assert same.orders == []


def test_one_day_or_a_flat_peer_does_not_block_the_breakout():
    start = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    broker = FakeBroker()
    store = MemoryStore()
    _ready(store, _quiet_bars(start, 5))
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=1, pnl=1500))
    request = _request()
    broker._candles.append(_candle(start + timedelta(minutes=5), 10.4, high=10.45, low=10.3))
    signal = run_minute(request, broker, store, now=start + timedelta(minutes=6))
    assert signal["phase"] == "signal"
    assert signal["target"] > 0
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.5, high=10.55, low=10.4))
    filled = run_minute(request, broker, store, now=start + timedelta(minutes=7))
    assert filled["phase"] == "order"
    assert filled["order"]["signed"] == 10


def test_a_stop_still_orders_while_a_senior_peer_is_in_profit():
    broker = FakeBroker()
    broker.lots = 12
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
    store.save(peer_key("long"), _peer("long", channel=12420, held=-8, days=3, pnl=4000))
    broker._candles.append(_candle(start + timedelta(minutes=5), 9.4, high=9.6, low=9.0))
    broker._candles.append(_candle(start + timedelta(minutes=6), 10.0, high=10.1, low=9.9))
    result = run_minute(_request(), broker, store, now=start + timedelta(minutes=7))
    assert result["phase"] == "order"
    assert result["order"]["signed"] == -10
    assert result["order"]["reason"] == "stop"


def test_a_fill_publishes_the_peer_and_a_flat_book_clears_it():
    store = MemoryStore()
    book = _FillBook("CRZ6", 100_000.0)
    book.held = -4
    book.target = -4
    book.avg = 10.0
    book.entry_i = 0
    bars = [
        {"t": "2026-09-28T10:00:00+03:00", "c": 10.0},
        {"t": "2026-09-29T10:00:00+03:00", "c": 9.8},
        {"t": "2026-09-30T10:00:00+03:00", "c": 9.5},
    ]
    publish_peer(store, "long", 12420, book, bars)
    saved = store.load(peer_key("long"))
    assert saved["held"] == -4
    assert saved["target"] == -4
    assert saved["held_days"] == 2
    assert saved["channel"] == 12420
    assert saved["pnl"] == pytest.approx((9.5 - 10.0) * -4 * 1000)
    book.held = 0
    publish_peer(store, "long", 12420, book, bars)
    cleared = store.load(peer_key("long"))
    assert cleared["held"] == 0
    assert cleared["pnl"] == 0
    publish_peer(store, "missing", 10, book, bars)
    assert store.load(peer_key("missing")) is None
    publish_peer(store, "long", 12420, None, bars)


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
        readme = stored.read("README.txt").decode("utf-8")
        for word in ("reconcile", "probe", "fill_per_minute", "secret_id", "halted", "VK_TOKEN", "veto", "peers/"):
            assert word in readme
        for name in PRESETS:
            assert name in readme
        assert "momentum" in readme
        stored.extractall(extract)
    code = (
        "import function.index, sys; "
        "assert callable(function.index.handler); "
        "assert 'backtrader' not in sys.modules; "
        "assert 'pandas' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd=extract)


def test_momentum_sign_matches_the_chosen_account_where_the_window_fits():
    """Живой знак совпадает с выбранным счётом, пока окну хватает пяти дней до конца ряда."""
    import numpy as np

    pytest.importorskip("research.cr_momentum")
    from research.cr_momentum import Series
    from research.cr_momentum_params import chosen_account_position
    from cnyrub.live.momentum import HISTORY_DAYS, HOLD, LOOKBACK, sign_for_today

    assert HISTORY_DAYS == LOOKBACK + 1

    count = 90
    up = (0.004, -0.001, 0.003, -0.002, 0.005)
    down = (-0.004, 0.001, -0.003, 0.002, -0.005)
    level = np.zeros(count)
    level[0] = 10.0
    for index in range(1, count):
        factor = 1.0 + (up if index < 60 else down)[(index - 1) % 5]
        if index == 33:
            factor = 1.08
        level[index] = level[index - 1] * factor
    volume = np.full(count, 100.0)
    volume[40] = 5_000.0
    day_range = np.full(count, 0.002)
    day_range[25] = 0.2
    ret = np.zeros(count)
    ret[1:] = np.diff(level)
    series = Series(
        days=np.arange(count),
        contract=np.array(["CR"] * count),
        ret=ret,
        roll=np.zeros(count, dtype=bool),
        level=level,
    )
    position = chosen_account_position(series, volume, day_range)
    days = [
        {
            "d": f"2026-01-{index + 1:02d}",
            "o": float(level[index]),
            "h": float(level[index] * (1 + day_range[index] / 2)),
            "l": float(level[index] * (1 - day_range[index] / 2)),
            "c": float(level[index]),
            "v": float(volume[index]),
        }
        for index in range(count)
    ]
    matched = 0
    for today in range(count):
        decision = LOOKBACK + ((today - (LOOKBACK + 1)) // HOLD) * HOLD if today >= LOOKBACK + 1 else None
        if decision is not None and decision + HOLD >= count:
            continue
        assert sign_for_today(days[:today]) == int(position[today])
        matched += 1
    assert matched == 86
    assert int(position[21]) == 1
    assert int(position[27]) == 0
    assert int(position[33]) == 1
    assert int(position[34]) == 0
    assert int(position[35]) == 1
    assert int(position[42]) == 0
    assert int(position[71]) == -1
    tail = count - 1
    assert int(position[tail]) == 0
    assert sign_for_today(days[:tail]) == -1


def _momentum_days(count: int, end: date) -> list[dict[str, float | str]]:
    cursor = end
    found: list[date] = []
    while len(found) < count:
        if cursor.weekday() < 5:
            found.append(cursor)
        cursor -= timedelta(days=1)
    found.reverse()
    days: list[dict[str, float | str]] = []
    for index, day in enumerate(found):
        close = 10.05 if index == 40 else 10.0
        days.append({"d": day.isoformat(), "o": close, "h": close + 0.01, "l": close - 0.01, "c": close, "v": 100.0})
    return days


def test_momentum_orders_on_its_own_account_and_does_not_publish_a_peer():
    broker = FakeBroker()
    broker.margin_value = 100.0
    store = MemoryStore()
    account = "acc-momentum"
    days = _momentum_days(45, date(2026, 9, 29))
    store.save(
        state_key("momentum", account),
        {
            "version": 1,
            "strategy": "momentum",
            "account_id": account,
            "instrument": {
                "secid": "CRZ6",
                "uid": "uid-1",
                "figi": "FUT",
                "frsttrade": "2026-06-16",
                "lsttrade": "2026-12-15",
                "trade_from": "2026-06-16",
            },
            "days": days,
            "today": None,
            "bars": [],
            "ready": True,
            "last_bar": "2026-09-29T23:00:00+03:00",
            "sized_day": None,
            "history_before": "2026-09-29",
            "history_walked": 45,
            "halted": None,
            "book": None,
        },
    )
    moment = datetime(2026, 9, 30, 10, 0, tzinfo=MSK)
    broker._candles = [_candle(moment, 10.0)]
    request = _request(strategy="momentum", account_id=account, channel=None)
    first = run_minute(request, broker, store, now=datetime(2026, 9, 30, 10, 5, tzinfo=MSK))
    assert first["phase"] == "order"
    assert first["target"] == 30
    assert first["held"] == 10
    assert first["order"]["signed"] == 10
    assert broker.orders == [{"signed": 10, "order_id": first["order"]["id"], "account": account, "uid": "uid-1"}]
    assert peer_key("momentum") not in store.data
    assert not any(key.startswith("peers/") for key in store.data)
    broker._candles.append(_candle(datetime(2026, 9, 30, 10, 5, tzinfo=MSK), 10.0))
    second = run_minute(request, broker, store, now=datetime(2026, 9, 30, 10, 6, tzinfo=MSK))
    assert second["phase"] == "order"
    assert second["target"] == 30
    assert second["held"] == 20
    assert second["order"]["signed"] == 10
    assert peer_key("momentum") not in store.data


def test_momentum_is_outside_the_seven_window_exchange():
    request = parse_event({"strategy": "momentum", "account_id": "acc-momentum", "token": "t"})
    assert request.strategy == "momentum"
    assert request.params.leverage == 3.0
    assert request.params.channel == 0
    assert "momentum" not in PRESETS
    assert set(PRESETS) == {"short", "long", "thirty", "fortyfive", "sixty", "m1760", "m2160"}
    store = MemoryStore()
    book = _FillBook("CRZ6", None)
    book.held = -3
    book.target = -3
    book.avg = 10.0
    publish_peer(store, "momentum", 0, book, [{"t": "2026-09-30T10:00:00+03:00", "c": 10.0}])
    assert store.load(peer_key("momentum")) is None
    fresh = _FillBook("CRZ6", None)
    fresh.target = 2
    store.save(
        peer_key("momentum"),
        {
            "strategy": "momentum",
            "channel": 10**9,
            "held": -8,
            "target": -8,
            "held_days": 9,
            "pnl": 50_000.0,
        },
    )
    for name, channel in (("short", 525), ("m1760", 1760), ("m2160", 2160), ("long", 12_420)):
        assert veto_source(store, name, channel, fresh) == ""


def test_momentum_history_does_not_order_or_write_a_peer():
    broker = FakeBroker()
    store = MemoryStore()
    yesterday = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)
    today = datetime(2026, 9, 29, 10, 0, tzinfo=MSK)
    broker._candles = [_candle(yesterday + timedelta(minutes=i), 10.0) for i in range(3)]
    broker._candles += [_candle(today + timedelta(minutes=i), 10.0) for i in range(2)]
    request = _request(strategy="momentum", account_id="acc-momentum", channel=None)
    now = datetime(2026, 9, 29, 12, 0, tzinfo=MSK)
    first = run_minute(request, broker, store, now=now)
    assert first["phase"] == "history"
    assert first["order"] is None
    assert broker.orders == []
    opened = store.load(state_key("momentum", "acc-momentum"))
    assert opened["days"] == []
    assert opened["today"]["d"] == "2026-09-29"
    second = run_minute(request, broker, store, now=now)
    assert second["phase"] == "history"
    assert second["order"] is None
    assert broker.orders == []
    assert not any(key.startswith("peers/") for key in store.data)
    saved = store.load(state_key("momentum", "acc-momentum"))
    assert [day["d"] for day in saved["days"]] == ["2026-09-28"]
    assert saved["ready"] is False


def test_live_package_does_not_import_backtrader_or_pandas():
    code = (
        "import cnyrub.live.handler, cnyrub.live.service, cnyrub.engine; "
        "import sys; "
        "assert 'backtrader' not in sys.modules; "
        "assert 'pandas' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd="/workspace")
