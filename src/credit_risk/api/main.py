from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from credit_risk.api.schemas import (
    BenchmarkCandidateResponse,
    BusinessDecisionResponse,
    CreditApplicationRequest,
    HealthResponse,
    IngestDatasetRequest,
    IngestDatasetResponse,
    MetricsResponse,
    PredictionResponse,
    TrainModelRequest,
    TrainModelResponse,
    TunedModelResponse,
)
from credit_risk.datasets.credit_risk import publish_credit_risk_dataset
from credit_risk.logging_utils import configure_logging
from credit_risk.serving.service import CreditRiskPredictionService
from credit_risk.training.service import CreditRiskTrainer


configure_logging()
logger = logging.getLogger(__name__)
trainer = CreditRiskTrainer()
prediction_service = CreditRiskPredictionService()


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


@app.post(
    "/models/credit-risk/train",
    response_model=TrainModelResponse,
    tags=["models"],
    summary="Train and store the top-3 quorum model in MinIO",
)
def train_credit_risk_model(payload: TrainModelRequest) -> TrainModelResponse:
    reference_date = None
    if payload.model_month:
        reference_date = datetime.strptime(f"{payload.model_month}-01", "%Y-%m-%d")

    result = trainer.train(
        dataset_month=payload.dataset_month,
        model_name=payload.model_name,
        minimum_accuracy=payload.minimum_accuracy,
        optuna_trials=payload.optuna_trials,
        threshold=payload.threshold,
        reference_date=reference_date,
    )
    return TrainModelResponse(
        model_name=result.model_name,
        object_name=result.stored_model.object_name,
        model_month=result.stored_model.month,
        top3_models=result.top3_models,
        minimum_accuracy=result.minimum_accuracy,
        test_metrics=MetricsResponse(**result.test_metrics.__dict__),
        implementation_metrics=MetricsResponse(**result.implementation_metrics.__dict__),
        benchmark_ranking=[BenchmarkCandidateResponse(**item.__dict__) for item in result.benchmark_ranking],
        tuned_ranking=[TunedModelResponse(**item.__dict__) for item in result.tuned_ranking],
        message="Top-3 quorum model trained and stored in MinIO.",
    )


@app.post(
    "/models/credit-risk/predict",
    response_model=PredictionResponse,
    tags=["models"],
    summary="Predict default risk from business input",
)
def predict_credit_risk(payload: CreditApplicationRequest) -> PredictionResponse:
    feature_payload = payload.model_dump()
    model_name = feature_payload.pop("model_name")
    model_month = feature_payload.pop("model_month")
    result = prediction_service.predict(
        payload=feature_payload,
        model_name=model_name,
        month=model_month,
    )
    return PredictionResponse(
        model_name=result.model_name,
        model_month=result.model_month,
        default_probability=result.default_probability,
        predicted_label=result.predicted_label,
        threshold=result.threshold,
        decision=BusinessDecisionResponse(
            risk_band=result.decision.risk_band,
            recommendation=result.decision.recommendation,
            reasons=result.decision.reasons,
        ),
    )


def run() -> None:
    uvicorn.run("credit_risk.api.main:app", host="0.0.0.0", port=8000, reload=False)
