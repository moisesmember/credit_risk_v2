from __future__ import annotations

from datetime import date, datetime


def month_key(reference_date: date | datetime | None = None) -> str:
    current = reference_date or datetime.utcnow()
    return f"{current.year:04d}-{current.month:02d}"


def dataset_object_name(*, dataset_name: str, filename: str, month: str, stage: str = "raw") -> str:
    return f"datasets/{stage}/{dataset_name}/{month}/{filename}"


def latest_dataset_manifest_name(*, dataset_name: str, stage: str = "raw") -> str:
    return f"_catalog/datasets/{stage}/{dataset_name}/latest.json"


def model_object_name(
    *,
    model_name: str,
    artifact_name: str,
    month: str,
    stage: str = "trained",
) -> str:
    return f"models/{stage}/{model_name}/{month}/{artifact_name}"


def latest_model_manifest_name(*, model_name: str, stage: str = "trained") -> str:
    return f"_catalog/models/{stage}/{model_name}/latest.json"
