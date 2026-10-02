"""Точка входа Yandex Cloud Functions.

Готовый архив для Облака: yandex/cnyrub-function.zip.
В нём точка входа function.index.handler, код живого счёта и requirements.txt.
Исследования, минуток и тестов там нет. Зависимость одна: boto3.
backtrader в функцию не входит. Пересобрать архив: python -m function.pack.

  yc serverless function version create \\
    --function-name cnyrub \\
    --runtime python312 \\
    --entrypoint function.index.handler \\
    --memory 256m \\
    --execution-timeout 60s \\
    --source-path yandex/cnyrub-function.zip \\
    --environment STATE_BUCKET=<бакет> \\
    --environment AWS_ACCESS_KEY_ID=<ключ> \\
    --environment AWS_SECRET_ACCESS_KEY=<секрет>

Пять таймеров на минуту, пока идёт сессия деривативов, повторы выключены.
В payload таймера JSON: strategy (short, long, thirty, fortyfive или sixty),
account_id, secret_id. Свободные рубли всех этих окон покупают LQDT
(или TMON в поле cash_ticker; в приложении тот же фонд подписан TMON@). В рублях остаётся залог на сделку этой минуты:
fill_per_minute лотов, по умолчанию 10, и ещё половина этого залога.
Фонд продаётся только если этих рублей не хватает на увеличение позиции.
После сокращения фьючерса свободные рубли сверх этого залога снова покупают фонд.
Тестовый вызов шлёт тот же JSON, но с полем token вместо secret_id.
Оба пути торгуют реальный счёт. Залог читается из API, история окна
догружается сама по одному дню и в этот вызов заявка не ставится.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from cnyrub.live.handler import handle


def handler(event, context):
    return handle(event, context)
