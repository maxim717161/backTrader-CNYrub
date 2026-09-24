"""Проверка трендового канала на фронтальных окнах CNY/RUB.

Позиция живёт только внутри одного контракта. В последний день окна новая
сделка не открывается, уже открытая закрывается на открытии этого дня.
Сигнал — закрытие за пределами предыдущих 20 дневных максимумов или минимумов
этого же контракта. Выход — обратный пробой 10 дней или стоп в 2,5 медианы
дневного диапазона. CRM2 не торгуется.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import backtrader as bt
import pandas as pd

CHANNEL = 20
EXIT_CHANNEL = 10
STOP_MULT = 2.5
VOL_FRAC = 0.1
MULTIPLIER = 1000.0
COMMISSION = 1.0
MARGIN = 20_000.0
START_CASH = 1_000_000.0
SKIP_SECIDS = {"CRM2"}
DATA_PATH = Path("data/continuous/cny_front_1m.parquet")


class DailyData(bt.feeds.PandasData):
    lines = ("flatten_on_close", "is_last")
    params = (
        ("flatten_on_close", -1),
        ("is_last", -1),
    )


class FrontDonchian(bt.Strategy):
    params = dict(
        secid="",
        channel=CHANNEL,
        exit_channel=EXIT_CHANNEL,
        stop_mult=STOP_MULT,
        vol_frac=VOL_FRAC,
    )

    def __init__(self) -> None:
        self.entry_order = None
        self.protective = None
        self.stop_dist = None
        self.exit_reason = ""
        self.trades: list[dict[str, object]] = []
        self.values: list[tuple[object, float]] = []

    def next(self) -> None:
        self.values.append((self.data.datetime.date(0), float(self.broker.getvalue())))
        if self.data.flatten_on_close[0]:
            self._exit("expiry")
            return
        if self.data.is_last[0] or len(self) <= self.p.channel:
            return
        if self.position:
            self._exit_on_channel()
            return
        if self.entry_order is not None:
            return
        self._enter_on_breakout()

    def notify_order(self, order: bt.Order) -> None:
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

    def _exit_on_channel(self) -> None:
        if len(self) <= self.p.exit_channel:
            return
        lows = _window(self.data.low, self.p.exit_channel)
        highs = _window(self.data.high, self.p.exit_channel)
        if self.position.size > 0 and self.data.close[0] < min(lows):
            self._exit("channel")
        elif self.position.size < 0 and self.data.close[0] > max(highs):
            self._exit("channel")

    def _enter_on_breakout(self) -> None:
        if not _volume_ok(self.data, self.p.channel, self.p.vol_frac):
            return
        highs = _window(self.data.high, self.p.channel)
        lows = _window(self.data.low, self.p.channel)
        ranges = [high - low for high, low in zip(highs, lows)]
        self.stop_dist = self.p.stop_mult * statistics.median(ranges)
        close = self.data.close[0]
        if close > max(highs):
            self.entry_order = self.buy()
        elif close < min(lows):
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
        # Заявка, отправленная из notify_order, иначе ожила бы только со следующего дня.
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


def _window(line: bt.LineSeries, length: int) -> list[float]:
    return [float(line[-i]) for i in range(1, length + 1)]


def _volume_ok(data: bt.LineSeries, length: int, fraction: float) -> bool:
    volumes = _window(data.volume, length)
    typical = statistics.median(volumes)
    if typical <= 0:
        return True
    return float(data.volume[0]) >= fraction * typical


def load_daily(path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Дневные свечи по каждому фронтальному окну, в порядке склейки."""
    bars = pd.read_parquet(path, columns=["datetime", "open", "high", "low", "close", "volume", "secid"])
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["day"] = bars["datetime"].dt.floor("D")
    grouped = bars.groupby(["secid", "day"], sort=False)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    frames: dict[str, pd.DataFrame] = {}
    for secid in bars["secid"].drop_duplicates():
        frame = daily.loc[secid].copy()
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "datetime"
        frame["flatten_on_close"] = 0.0
        frame["is_last"] = 0.0
        frame.iloc[-1, frame.columns.get_loc("is_last")] = 1.0
        if len(frame) >= 2:
            frame.iloc[-2, frame.columns.get_loc("flatten_on_close")] = 1.0
        frames[str(secid)] = frame
    return frames


def run_contract(secid: str, frame: pd.DataFrame) -> FrontDonchian:
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(FrontDonchian, secid=secid)
    cerebro.adddata(DailyData(dataname=frame))
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(
        commission=COMMISSION,
        commtype=bt.CommInfoBase.COMM_FIXED,
        mult=MULTIPLIER,
        margin=MARGIN,
        stocklike=False,
    )
    return cerebro.run()[0]


def buy_and_hold(frame: pd.DataFrame) -> float:
    """Лонг с первого возможного входа стратегии до открытия последнего дня."""
    if len(frame) < CHANNEL + 3:
        return 0.0
    entry = float(frame["open"].iloc[CHANNEL + 1])
    exit_ = float(frame["open"].iloc[-1])
    return (exit_ - entry) * MULTIPLIER - 2 * COMMISSION


def run_all(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    traded = {secid: frame for secid, frame in frames.items() if secid not in SKIP_SECIDS}
    results = []
    for secid, frame in traded.items():
        strategy = run_contract(secid, frame)
        gross = sum(float(trade["pnl"]) for trade in strategy.trades)
        net = sum(float(trade["pnlcomm"]) for trade in strategy.trades)
        results.append(
            {
                "secid": secid,
                "days": len(frame),
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


def _max_drawdown(results: list[dict[str, object]]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for item in results:
        previous = START_CASH
        for _day, value in item["values"]:
            equity += value - previous
            previous = value
            peak = max(peak, equity)
            worst = min(worst, equity - peak)
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
    lines = [
        "Трендовый канал, 1 контракт, лот 1000 юаней.",
        f"Комиссия {COMMISSION:.0f} руб. за сторону. CRM2 пропущен.",
        f"Сделок: {len(trades)}. Прибыльных: {len(wins)}.",
        f"Лонгов: {len(longs)}, результат {sum(float(trade['pnlcomm']) for trade in longs):,.0f} руб.",
        f"Шортов: {len(shorts)}, результат {sum(float(trade['pnlcomm']) for trade in shorts):,.0f} руб.",
        f"Без комиссии: {gross:,.0f} руб. После комиссии: {net:,.0f} руб.",
        f"Фактор прибыли: {factor_text}. Просадка: {_max_drawdown(contracts):,.0f} руб.",
        f"Просто лонг на тех же окнах: {hold:,.0f} руб.",
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
    lines.append("")
    lines.append("Выходы: " + ", ".join(f"{name} {count}" for name, count in sorted(reasons.items())))
    return "\n".join(lines)


def main() -> int:
    summary = run_all(load_daily())
    print(report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
