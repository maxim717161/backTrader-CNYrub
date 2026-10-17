"""Где сейчас находится запуск. Нужно, чтобы при зависании назвать этап."""

from __future__ import annotations

stage = "старт"


def mark(text: str) -> None:
    global stage
    stage = text
