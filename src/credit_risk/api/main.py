from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from credit_risk.api.schemas import HealthResponse, IngestDatasetRequest, IngestDatasetResponse
from credit_risk.datasets.credit_risk import publish_credit_risk_dataset
from credit_risk.logging_utils import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Credit Risk Data API",
    version="0.1.0",
    description=(
        "API for credit risk dataset ingestion and MinIO-backed storage. "
        "Swagger UI is available at /docs."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/datasets/credit-risk/ingest",
    response_model=IngestDatasetResponse,
    tags=["datasets"],
    summary="Ingest credit risk dataset into MinIO",
)
def ingest_credit_risk_dataset(payload: IngestDatasetRequest) -> IngestDatasetResponse:
    reference_date = None
    if payload.month:
        reference_date = datetime.strptime(f"{payload.month}-01", "%Y-%m-%d")

    logger.info(
        "Received ingestion request for local_path='%s', month='%s', overwrite=%s.",
        payload.local_path,
        payload.month,
        payload.overwrite,
    )
    version = publish_credit_risk_dataset(
        local_path=Path(payload.local_path),
        reference_date=reference_date,
        overwrite=payload.overwrite,
    )

    return IngestDatasetResponse(
        dataset_name=version.dataset_name,
        month=version.month,
        object_name=version.object_name,
        uploaded_at=version.uploaded_at,
        message="Dataset ingested to MinIO.",
    )


def run() -> None:
    uvicorn.run("credit_risk.api.main:app", host="0.0.0.0", port=8000, reload=False)
