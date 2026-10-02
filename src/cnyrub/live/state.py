"""Состояние стратегии: один JSON на счёт в Object Storage или в памяти теста."""

from __future__ import annotations

import json
import os
from typing import Protocol


def compact_document(document: dict[str, object]) -> dict[str, object]:
    """Свечи в бакете — списки из шести полей, без повторения имён на каждой минуте."""
    bars = document.get("bars")
    if not isinstance(bars, list) or not any(isinstance(bar, dict) for bar in bars):
        return document
    slim = dict(document)
    slim["bars"] = [_compact_bar(bar) for bar in bars]
    return slim


def _compact_bar(bar: object) -> list[object]:
    if isinstance(bar, dict):
        return [bar["t"], bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]]
    return list(bar)


def state_key(strategy: str, account_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in account_id)
    return f"state/{strategy}-{safe}.json"


class StateStore(Protocol):
    def load(self, key: str) -> dict[str, object] | None: ...

    def save(self, key: str, document: dict[str, object]) -> None: ...


class MemoryStore:
    """Копия через JSON, чтобы тест не делил словарь с вызывающим кодом."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, object]] = {}
        self.saves = 0

    def load(self, key: str) -> dict[str, object] | None:
        document = self.data.get(key)
        if document is None:
            return None
        return json.loads(json.dumps(document))

    def save(self, key: str, document: dict[str, object]) -> None:
        self.saves += 1
        self.data[key] = json.loads(json.dumps(compact_document(document), separators=(",", ":")))


class ObjectStore:
    """Бакет Yandex Object Storage. Ключи читаются из окружения в момент вызова."""

    def __init__(self, bucket: str, endpoint: str = "https://storage.yandexcloud.net") -> None:
        self.bucket = bucket
        self.endpoint = endpoint
        self._client = None

    def load(self, key: str) -> dict[str, object] | None:
        from botocore.exceptions import ClientError

        try:
            response = self._s3().get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        body = response["Body"].read().decode("utf-8")
        document = json.loads(body)
        if not isinstance(document, dict):
            raise ValueError("состояние в бакете — не объект")
        return document

    def save(self, key: str, document: dict[str, object]) -> None:
        body = json.dumps(
            compact_document(document),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._s3().put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")

    def _s3(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name="ru-central1",
            )
        return self._client


def object_store_from_env() -> ObjectStore:
    bucket = os.environ.get("STATE_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("нужна переменная STATE_BUCKET")
    return ObjectStore(bucket)
