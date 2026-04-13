from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from minio import Minio
from minio.error import S3Error

from credit_risk.config import MinioSettings


logger = logging.getLogger(__name__)


class MinioStorage:
    def __init__(self, settings: MinioSettings | None = None) -> None:
        self.settings = settings or MinioSettings.from_env()
        self.client = Minio(
            endpoint=self.settings.endpoint,
            access_key=self.settings.access_key,
            secret_key=self.settings.secret_key,
            secure=self.settings.secure,
        )

    @property
    def bucket(self) -> str:
        return self.settings.bucket

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            logger.info("Creating missing MinIO bucket '%s'.", self.bucket)
            self.client.make_bucket(self.bucket)

    def object_exists(self, *, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchBucket", "NoSuchKey", "NoSuchObject"}:
                return False
            raise

    def put_file(
        self,
        *,
        object_name: str,
        file_path: str | Path,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.ensure_bucket()
        path = Path(file_path)
        logger.info("Uploading file '%s' to MinIO object '%s'.", path, object_name)
        self.client.fput_object(
            self.bucket,
            object_name,
            str(path),
            content_type=content_type,
            metadata=metadata,
        )

    def put_bytes(
        self,
        *,
        object_name: str,
        payload: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.ensure_bucket()
        data = BytesIO(payload)
        logger.info("Uploading bytes payload to MinIO object '%s'.", object_name)
        self.client.put_object(
            self.bucket,
            object_name,
            data,
            length=len(payload),
            content_type=content_type,
            metadata=metadata,
        )

    def get_bytes(self, *, object_name: str) -> bytes:
        logger.info("Reading MinIO object '%s'.", object_name)
        response = self.client.get_object(self.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def put_json(self, *, object_name: str, payload: dict[str, Any]) -> None:
        self.put_bytes(
            object_name=object_name,
            payload=json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8"),
            content_type="application/json",
        )

    def get_json(self, *, object_name: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(object_name=object_name).decode("utf-8"))

    def list_objects(self, *, prefix: str, recursive: bool = True) -> list[str]:
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=recursive)
        return [item.object_name for item in objects]
    
    def exists(self, *, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                return False
            raise
