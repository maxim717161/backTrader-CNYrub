"""Точка входа Yandex Cloud Functions.

Архив — корень репозитория: каталоги function и src и файл requirements.txt.
Туда не кладут .venv, data и tests. Точка входа: function.index.handler.
Зависимости ставятся из requirements.txt (boto3). backtrader в функцию не входит.

  yc serverless function version create \\
    --function-name cnyrub \\
    --runtime python312 \\
    --entrypoint function.index.handler \\
    --memory 256m \\
    --execution-timeout 60s \\
    --source-path . \\
    --environment STATE_BUCKET=<бакет> \\
    --environment AWS_ACCESS_KEY_ID=<ключ> \\
    --environment AWS_SECRET_ACCESS_KEY=<секрет>

Три таймера на минуту, пока идёт сессия деривативов, повторы выключены.
В payload таймера JSON: strategy (short, long или thirty), account_id, secret_id.
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
