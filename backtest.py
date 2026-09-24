"""Проверка трендового канала на минутках каждого квартального CNY/RUB.

Склеенный ряд не используется: у каждого контракта свой прогон. История
длиннее фронтального окна на месяц до экспирации предыдущего выпуска.
Позиция за границу контракта не переносится. В последний день новая сделка
не открывается, уже открытая закрывается на первом открытии этого дня.

Сигнал — дневное закрытие за пределами предыдущих 20 дневных экстремумов
этого же контракта. Вход на первом открытии следующего дня, и только если
объём дня сигнала не ниже медианы предыдущих 20 дней. Выход — обратный
пробой 10 дней или стоп в 2,5 медианы дневного диапазона.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime
from pathlib import Path

import backtrader as bt
import pandas as pd

CHANNEL = 20
EXIT_CHANNEL = 10
STOP_MULT = 2.5
MULTIPLIER = 1000.0
COMMISSION = 1.0
MARGIN = 20_000.0
START_CASH = 1_000_000.0
BARS_DIR = Path("data/bars")


class MinuteDonchian(bt.Strategy):
    params = dict(
        secid="",
        last_day="",
        channel=CHANNEL,
        exit_channel=EXIT_CHANNEL,
        stop_mult=STOP_MULT,
    )

    def __init__(self) -> None:
        self.entry_order = None
        self.protective = None
        self.stop_dist = None
        self.exit_reason = ""
        self.trades: list[dict[str, object]] = []
        self.values: list[tuple[object, float]] = []
        self.days: list[dict[str, object]] = []
        self.current: dict[str, object] | None = None

    def next_open(self) -> None:
        today = self.data.datetime.date(0)
        if self.current is None or self.current["date"] == today:
            return
        self.days.append(self.current)
        if today == self._last_day():
            self._exit("expiry")
            return
        if self.position:
            self._exit_on_channel()
            return
        if self.entry_order is not None:
            return
        self._enter_on_breakout()

    def next(self) -> None:
        self.values.append((self.data.datetime.datetime(0), float(self.broker.getvalue())))
        self._accumulate()

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Canceled, order.Margin, order.Rejected, order.Expired):
            if self.entry_order is not None and order.ref == self.entry_order.ref:
                self.entry_order = None
            return
        if order.status != order.Completed:
            return
        if self.protective is not None and order.ref == self.protective.ref:
            self.exit_reason = "stop"
            self.protective = None
            return
        if self.entry_order is None or order.ref != self.entry_order.ref:
            return
        self._arm_stop(order)
        self.entry_order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if not trade.isclosed:
            return
        self.trades.append(
            {
                "secid": self.p.secid,
                "direction": "long" if trade.long else "short",
                "pnl": float(trade.pnl),
                "pnlcomm": float(trade.pnlcomm),
                "bars": int(trade.barlen),
                "reason": self.exit_reason,
            }
        )
        self.exit_reason = ""

    def session_count(self) -> int:
        return len(self.days) + (1 if self.current is not None else 0)

    def _last_day(self) -> date:
        value = self.p.last_day
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    def _accumulate(self) -> None:
        today = self.data.datetime.date(0)
        opened = float(self.data.open[0])
        high = float(self.data.high[0])
        low = float(self.data.low[0])
        close = float(self.data.close[0])
        volume = float(self.data.volume[0])
        if self.current is None or self.current["date"] != today:
            self.current = {
                "date": today,
                "open": opened,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
            return
        self.current["high"] = max(float(self.current["high"]), high)
        self.current["low"] = min(float(self.current["low"]), low)
        self.current["close"] = close
        self.current["volume"] = float(self.current["volume"]) + volume

    def _prior(self, length: int) -> list[dict[str, object]] | None:
        if len(self.days) < length + 1:
            return None
        return self.days[-(length + 1) : -1]

    def _exit_on_channel(self) -> None:
        prior = self._prior(self.p.exit_channel)
        if prior is None:
            return
        signal = self.days[-1]
        if float(self.position.size) > 0 and float(signal["close"]) < min(float(day["low"]) for day in prior):
            self._exit("channel")
        elif float(self.position.size) < 0 and float(signal["close"]) > max(float(day["high"]) for day in prior):
            self._exit("channel")

    def _enter_on_breakout(self) -> None:
        prior = self._prior(self.p.channel)
        if prior is None:
            return
        signal = self.days[-1]
        if not _volume_ok(signal, prior):
            return
        ranges = [float(day["high"]) - float(day["low"]) for day in prior]
        self.stop_dist = self.p.stop_mult * statistics.median(ranges)
        close = float(signal["close"])
        if close > max(float(day["high"]) for day in prior):
            self.entry_order = self.buy()
        elif close < min(float(day["low"]) for day in prior):
            self.entry_order = self.sell()

    def _exit(self, reason: str) -> None:
        if not self.position:
            self._drop_protective()
            return
        self.exit_reason = reason
        self._drop_protective()
        self.close()

    def _drop_protective(self) -> None:
        if self.protective is not None:
            self.cancel(self.protective)
            self.protective = None

    def _arm_stop(self, entry: bt.Order) -> None:
        fill = float(entry.executed.price)
        dist = float(self.stop_dist)
        if entry.isbuy():
            stop_px = fill - dist
            protective = self.sell(exectype=bt.Order.Stop, price=stop_px, size=entry.executed.size)
        else:
            stop_px = fill + dist
            protective = self.buy(exectype=bt.Order.Stop, price=stop_px, size=entry.executed.size)
        # Заявка, отправленная из notify_order, иначе ожила бы только со следующего бара.
        broker = self.broker
        broker.submitted.remove(protective)
        protective.accept()
        broker._try_exec_stop(
            protective,
            self.data.open[0],
            self.data.high[0],
            self.data.low[0],
            stop_px,
            self.data.close[0],
        )
        if protective.alive():
            broker.pending.append(protective)
            self.protective = protective


def _volume_ok(signal: dict[str, object], prior: list[dict[str, object]]) -> bool:
    typical = statistics.median(float(day["volume"]) for day in prior)
    if typical <= 0:
        return True
    return float(signal["volume"]) >= typical


def _as_feed(frame: pd.DataFrame) -> pd.DataFrame:
    feed = frame.copy()
    if not isinstance(feed.index, pd.DatetimeIndex):
        feed["datetime"] = pd.to_datetime(feed["datetime"])
        feed = feed.set_index("datetime")
    feed = feed.sort_index()
    return feed.loc[:, ["open", "high", "low", "close", "volume"]]


def load_minutes(bars_dir: Path = BARS_DIR) -> dict[str, pd.DataFrame]:
    """Минутки каждого контракта отдельно, в порядке экспирации."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(bars_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["datetime", "open", "high", "low", "close", "volume"])
        frames[path.stem] = _as_feed(frame)
    ordered = sorted(frames, key=lambda secid: frames[secid].index[-1])
    return {secid: frames[secid] for secid in ordered}


def run_contract(secid: str, frame: pd.DataFrame) -> MinuteDonchian:
    feed = _as_feed(frame)
    last_day = pd.Timestamp(feed.index[-1]).date().isoformat()
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.addstrategy(MinuteDonchian, secid=secid, last_day=last_day)
    cerebro.adddata(bt.feeds.PandasData(dataname=feed))
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(
        commission=COMMISSION,
        commtype=bt.CommInfoBase.COMM_FIXED,
        mult=MULTIPLIER,
        margin=MARGIN,
        stocklike=False,
    )
    return cerebro.run()[0]


def session_days(frame: pd.DataFrame) -> pd.DataFrame:
    feed = _as_feed(frame)
    work = feed.reset_index()
    work["day"] = work["datetime"].dt.floor("D")
    return work.groupby("day", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )


def buy_and_hold(frame: pd.DataFrame) -> float:
    """Лонг с первого возможного входа стратегии до открытия последнего дня."""
    daily = session_days(frame)
    if len(daily) < CHANNEL + 3:
        return 0.0
    entry = float(daily["open"].iloc[CHANNEL + 1])
    exit_ = float(daily["open"].iloc[-1])
    return (exit_ - entry) * MULTIPLIER - 2 * COMMISSION


def run_all(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    results = []
    for secid, frame in frames.items():
        strategy = run_contract(secid, frame)
        gross = sum(float(trade["pnl"]) for trade in strategy.trades)
        net = sum(float(trade["pnlcomm"]) for trade in strategy.trades)
        results.append(
            {
                "secid": secid,
                "days": strategy.session_count(),
                "trades": strategy.trades,
                "gross": gross,
                "net": net,
                "hold": buy_and_hold(frame),
                "values": strategy.values,
            }
        )
    trades = [trade for item in results for trade in item["trades"]]
    return {"contracts": results, "trades": trades}


def _profit_factor(trades: list[dict[str, object]]) -> float | None:
    wins = sum(float(trade["pnlcomm"]) for trade in trades if float(trade["pnlcomm"]) > 0)
    losses = sum(float(trade["pnlcomm"]) for trade in trades if float(trade["pnlcomm"]) < 0)
    if losses == 0:
        return None
    return wins / abs(losses)


def _max_drawdown(values: list[tuple[object, float]]) -> float:
    peak = START_CASH
    worst = 0.0
    for _moment, value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def report(summary: dict[str, object]) -> str:
    trades = summary["trades"]
    assert isinstance(trades, list)
    contracts = summary["contracts"]
    assert isinstance(contracts, list)
    net = sum(float(trade["pnlcomm"]) for trade in trades)
    gross = sum(float(trade["pnl"]) for trade in trades)
    hold = sum(float(item["hold"]) for item in contracts)
    longs = [trade for trade in trades if trade["direction"] == "long"]
    shorts = [trade for trade in trades if trade["direction"] == "short"]
    wins = [trade for trade in trades if float(trade["pnlcomm"]) > 0]
    factor = _profit_factor(trades)
    factor_text = "—" if factor is None else f"{factor:.2f}"
    worst = min((_max_drawdown(item["values"]) for item in contracts), default=0.0)
    lines = [
        "Трендовый канал на минутках каждого контракта, 1 контракт, лот 1000 юаней.",
        "Вход, если объём дня сигнала не ниже медианы предыдущих 20 дней.",
        f"Комиссия {COMMISSION:.0f} руб. за сторону. В последний день позиция закрывается.",
        "Окна пересекаются на месяц: сумма результатов — не один счёт.",
        f"Сделок: {len(trades)}. Прибыльных: {len(wins)}.",
        f"Лонгов: {len(longs)}, результат {sum(float(trade['pnlcomm']) for trade in longs):,.0f} руб.",
        f"Шортов: {len(shorts)}, результат {sum(float(trade['pnlcomm']) for trade in shorts):,.0f} руб.",
        f"Без комиссии: {gross:,.0f} руб. После комиссии: {net:,.0f} руб.",
        f"Фактор прибыли: {factor_text}. Худшая просадка одного контракта: {worst:,.0f} руб.",
        f"Просто лонг на тех же историях: {hold:,.0f} руб.",
        "",
        f"{'SECID':<8} {'ДНЕЙ':>5} {'СДЕЛОК':>7} {'СТРАТЕГИЯ':>12} {'ЛОНГ':>12}",
    ]
    for item in contracts:
        lines.append(
            f"{item['secid']:<8} {item['days']:>5} {len(item['trades']):>7} "
            f"{item['net']:>12,.0f} {item['hold']:>12,.0f}"
        )
    reasons: dict[str, int] = {}
    for trade in trades:
        reason = str(trade["reason"] or "—")
        reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        lines.append("")
        lines.append("Выходы: " + ", ".join(f"{name} {count}" for name, count in sorted(reasons.items())))
    if trades:
        lines.append("")
        lines.append("Сделки:")
        ranked = sorted(trades, key=lambda trade: float(trade["pnlcomm"]), reverse=True)
        for trade in ranked:
            lines.append(
                f"  {trade['secid']:<6} {trade['direction']:<5} {float(trade['pnlcomm']):>10,.0f} руб.  "
                f"{trade['reason'] or '—':<8} {int(trade['bars'])} мин."
            )
    return "\n".join(lines)


def main() -> int:
    summary = run_all(load_minutes())
    print(report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
