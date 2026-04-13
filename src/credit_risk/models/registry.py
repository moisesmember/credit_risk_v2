from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
import logging

from credit_risk.storage.minio_client import MinioStorage
from credit_risk.storage.paths import (
    latest_model_manifest_name,
    model_object_name,
    month_key,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArtifactVersion:
    model_name: str
    artifact_name: str
    stage: str
    month: str
    object_name: str
    uploaded_at: str
    serializer: str = "pickle"


class ModelRegistry:
    def __init__(self, storage: MinioStorage | None = None) -> None:
        self.storage = storage or MinioStorage()

    def save_pickle(
        self,
        *,
        model: object,
        model_name: str,
        artifact_name: str = "model.pkl",
        stage: str = "trained",
        reference_date: date | datetime | None = None,
    ) -> ModelArtifactVersion:
        month = month_key(reference_date)
        object_name = model_object_name(
            model_name=model_name,
            artifact_name=artifact_name,
            month=month,
            stage=stage,
        )
        payload = pickle.dumps(model)
        self.storage.put_bytes(
            object_name=object_name,
            payload=payload,
            content_type="application/octet-stream",
            metadata={"model_name": model_name, "stage": stage, "month": month},
        )

        version = ModelArtifactVersion(
            model_name=model_name,
            artifact_name=artifact_name,
            stage=stage,
            month=month,
            object_name=object_name,
            uploaded_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        self._write_latest_manifest(version)
        logger.info(
            "Saved model '%s' to MinIO object '%s' for month '%s'.",
            model_name,
            object_name,
            month,
        )
        return version

    def load_pickle(
        self,
        *,
        model_name: str,
        stage: str = "trained",
        month: str | None = None,
    ) -> object:
        version = self.resolve_version(model_name=model_name, stage=stage, month=month)
        payload = self.storage.get_bytes(object_name=version.object_name)

        if payload[:1] == b"{":
            raise ValueError(
                f"Object '{version.object_name}' appears to be JSON/text, not a pickle artifact."
            )

        return pickle.loads(payload)

    # def resolve_version(
    #     self,
    #     *,
    #     model_name: str,
    #     stage: str = "trained",
    #     month: str | None = None,
    # ) -> ModelArtifactVersion:
    #     if month is None:
    #         manifest_name = latest_model_manifest_name(model_name=model_name, stage=stage)
    #         payload = self.storage.get_json(object_name=manifest_name)
    #         return ModelArtifactVersion(**payload)

    #     prefix = f"models/{stage}/{model_name}/{month}/"
    #     objects = sorted(self.storage.list_objects(prefix=prefix))
    #     if not objects:
    #         raise FileNotFoundError(
    #             f"No model found in bucket '{self.storage.bucket}' with prefix '{prefix}'."
    #         )

    #     object_name = objects[-1]
    #     return ModelArtifactVersion(
    #         model_name=model_name,
    #         artifact_name=object_name.rsplit("/", 1)[-1],
    #         stage=stage,
    #         month=month,
    #         object_name=object_name,
    #         uploaded_at="unknown",
    #     )
    def resolve_version(
        self,
        *,
        model_name: str,
        stage: str = "trained",
        month: str | None = None,
        artifact_name: str = "model.pkl",
    ) -> ModelArtifactVersion:
        if month is None:
            manifest_name = latest_model_manifest_name(model_name=model_name, stage=stage)
            payload = self.storage.get_json(object_name=manifest_name)
            return ModelArtifactVersion(**payload)

        object_name = model_object_name(
            model_name=model_name,
            artifact_name=artifact_name,
            month=month,
            stage=stage,
        )

        if not self.storage.exists(object_name=object_name):
            raise FileNotFoundError(
                f"Model artifact not found in bucket '{self.storage.bucket}': '{object_name}'."
            )

        return ModelArtifactVersion(
            model_name=model_name,
            artifact_name=artifact_name,
            stage=stage,
            month=month,
            object_name=object_name,
            uploaded_at="unknown",
        )

    def _write_latest_manifest(self, version: ModelArtifactVersion) -> None:
        manifest_name = latest_model_manifest_name(
            model_name=version.model_name,
            stage=version.stage,
        )
        self.storage.put_json(object_name=manifest_name, payload=asdict(version))
