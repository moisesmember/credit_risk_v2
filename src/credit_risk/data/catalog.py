from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from io import BytesIO
import logging
from pathlib import Path

import pandas as pd

from credit_risk.storage.minio_client import MinioStorage
from credit_risk.storage.paths import (
    dataset_object_name,
    latest_dataset_manifest_name,
    month_key,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetVersion:
    dataset_name: str
    filename: str
    stage: str
    month: str
    object_name: str
    uploaded_at: str


class DatasetCatalog:
    def __init__(self, storage: MinioStorage | None = None) -> None:
        self.storage = storage or MinioStorage()

    def register_existing_dataset(
        self,
        *,
        dataset_name: str,
        filename: str,
        stage: str = "raw",
        reference_date: date | datetime | None = None,
    ) -> DatasetVersion:
        version = self._build_version(
            dataset_name=dataset_name,
            filename=filename,
            stage=stage,
            reference_date=reference_date,
        )
        manifest_name = latest_dataset_manifest_name(dataset_name=dataset_name, stage=stage)

        if self.storage.object_exists(object_name=manifest_name):
            current_manifest = self.storage.get_json(object_name=manifest_name)
            if current_manifest.get("object_name") == version.object_name:
                return DatasetVersion(**current_manifest)

        logger.info(
            "Registering existing dataset object '%s' as the latest version for dataset '%s'.",
            version.object_name,
            dataset_name,
        )
        self._write_latest_manifest(version)
        return version

    def publish_local_dataset(
        self,
        *,
        local_path: str | Path,
        dataset_name: str,
        stage: str = "raw",
        reference_date: date | datetime | None = None,
        content_type: str = "text/csv",
        overwrite: bool = False,
    ) -> DatasetVersion:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        version = self._build_version(
            dataset_name=dataset_name,
            filename=path.name,
            stage=stage,
            reference_date=reference_date,
        )
        month = version.month
        object_name = version.object_name
        metadata = {"dataset_name": dataset_name, "stage": stage, "month": month}

        if self.storage.object_exists(object_name=object_name) and not overwrite:
            logger.info(
                "Skipping upload for dataset '%s' because object '%s' already exists.",
                dataset_name,
                object_name,
            )
            return self.register_existing_dataset(
                dataset_name=dataset_name,
                filename=path.name,
                stage=stage,
                reference_date=reference_date,
            )

        self.storage.put_file(
            object_name=object_name,
            file_path=path,
            content_type=content_type,
            metadata=metadata,
        )
        self._write_latest_manifest(version)
        logger.info(
            "Published dataset '%s' to MinIO object '%s' for month '%s'.",
            dataset_name,
            object_name,
            month,
        )
        return version

    def read_csv(
        self,
        *,
        dataset_name: str,
        stage: str = "raw",
        month: str | None = None,
        **read_csv_kwargs: object,
    ) -> pd.DataFrame:
        version = self.resolve_version(dataset_name=dataset_name, stage=stage, month=month)
        payload = self.storage.get_bytes(object_name=version.object_name)
        return pd.read_csv(BytesIO(payload), **read_csv_kwargs)

    def resolve_version(
        self,
        *,
        dataset_name: str,
        stage: str = "raw",
        month: str | None = None,
    ) -> DatasetVersion:
        if month is None:
            return self.get_latest_version(dataset_name=dataset_name, stage=stage)

        prefix = f"datasets/{stage}/{dataset_name}/{month}/"
        objects = sorted(self.storage.list_objects(prefix=prefix))
        if not objects:
            raise FileNotFoundError(
                f"No dataset found in bucket '{self.storage.bucket}' with prefix '{prefix}'."
            )

        object_name = objects[-1]
        return DatasetVersion(
            dataset_name=dataset_name,
            filename=Path(object_name).name,
            stage=stage,
            month=month,
            object_name=object_name,
            uploaded_at="unknown",
        )

    def get_latest_version(self, *, dataset_name: str, stage: str = "raw") -> DatasetVersion:
        manifest_name = latest_dataset_manifest_name(dataset_name=dataset_name, stage=stage)
        logger.info(
            "Resolving latest version for dataset '%s' from manifest '%s'.",
            dataset_name,
            manifest_name,
        )
        payload = self.storage.get_json(object_name=manifest_name)
        return DatasetVersion(**payload)

    def _write_latest_manifest(self, version: DatasetVersion) -> None:
        manifest_name = latest_dataset_manifest_name(
            dataset_name=version.dataset_name,
            stage=version.stage,
        )
        self.storage.put_json(object_name=manifest_name, payload=asdict(version))

    def _build_version(
        self,
        *,
        dataset_name: str,
        filename: str,
        stage: str,
        reference_date: date | datetime | None,
    ) -> DatasetVersion:
        month = month_key(reference_date)
        object_name = dataset_object_name(
            dataset_name=dataset_name,
            filename=filename,
            month=month,
            stage=stage,
        )
        return DatasetVersion(
            dataset_name=dataset_name,
            filename=filename,
            stage=stage,
            month=month,
            object_name=object_name,
            uploaded_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
