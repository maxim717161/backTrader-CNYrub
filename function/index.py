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
    --environment AWS_SECRET_ACCESS_KEY=<секрет> \\
    --environment TELEGRAM_BOT_TOKEN=<токен бота> \\
    --environment TELEGRAM_CHAT_ID=<чат>

Пять таймеров на минуту, пока идёт сессия деривативов, повторы выключены.
В payload таймера JSON: strategy (short, long, thirty, fortyfive или sixty),
account_id, secret_id. fill_per_minute по умолчанию авто: за минуту берётся
не больше половины видимого стакана с той стороны, которую съест заявка.
0 — пауза, заявка не ставится. Положительное число — потолок контрактов
в минуту, стакан при этом не читается.
Тестовый вызов шлёт тот же JSON, но с полем token вместо secret_id.
Оба пути торгуют реальный счёт. Залог читается из API, история окна
догружается сама по одному дню и в этот вызов заявка не ставится.
Исполненная заявка на фьючерс уходит в Telegram, если заданы
TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID. Первая строка — имя стратегии,
дальше контракт, время, сторона, цена и короткая причина: пробой вверх,
пробой вниз, добор, возврат, откат, стоп, канал, время или экспирация.
У сокращения есть финансовый результат: пока позиция открыта — по контрактам
этой минуты, когда она дошла до нуля — итог всей сделки после комиссии.
Сбой Telegram сделку не отменяет.
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
