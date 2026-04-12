from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class MinioSettings:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False

    @classmethod
    def from_env(cls) -> "MinioSettings":
        access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER") or "minioadmin"
        secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or "minioadmin"

        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=access_key,
            secret_key=secret_key,
            bucket=os.getenv("MINIO_BUCKET", "credit-risk"),
            secure=_as_bool(os.getenv("MINIO_SECURE"), default=False),
        )
