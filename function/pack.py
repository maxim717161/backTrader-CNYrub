"""Собрать архив Cloud Function: только то, что исполняется в Яндекс Облаке.

В архиве точка входа function.index.handler, код живого счёта и requirements.txt.
Исследование, история минуток и тесты туда не входят. backtrader тоже.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "yandex" / "cnyrub-function.zip"

# Корень архива совпадает с корнем функции. requirements.txt лежит сверху:
# Облако ставит из него зависимости само.
FILES = (
    "requirements.txt",
    "function/index.py",
    "src/cnyrub/__init__.py",
    "src/cnyrub/contracts.py",
    "src/cnyrub/engine.py",
    "src/cnyrub/live/__init__.py",
    "src/cnyrub/live/broker.py",
    "src/cnyrub/live/config.py",
    "src/cnyrub/live/handler.py",
    "src/cnyrub/live/indicators.py",
    "src/cnyrub/live/service.py",
    "src/cnyrub/live/state.py",
)


def build(dest: Path | None = None) -> Path:
    """Записать архив. Повторная сборка из тех же файлов даёт те же байты."""
    target = ZIP_PATH if dest is None else dest
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            info = zipfile.ZipInfo(filename=name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (ROOT / name).read_bytes())
    return target


if __name__ == "__main__":
    print(build())
