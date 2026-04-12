from __future__ import annotations

from pydantic import BaseModel, Field


class IngestDatasetRequest(BaseModel):
    local_path: str = Field(
        default="data/credit_risk_dataset.csv",
        description="Local staging path for the CSV. If it does not exist, the service downloads it from Kaggle.",
    )
    month: str | None = Field(
        default=None,
        description="Reference month in YYYY-MM format. Defaults to the current month.",
        examples=["2026-04"],
    )
    overwrite: bool = Field(
        default=False,
        description="When true, overwrites the monthly object in MinIO even if it already exists.",
    )


class IngestDatasetResponse(BaseModel):
    dataset_name: str
    month: str
    object_name: str
    uploaded_at: str
    message: str


class HealthResponse(BaseModel):
    status: str
