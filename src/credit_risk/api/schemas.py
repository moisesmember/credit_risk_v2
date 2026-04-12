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


class TrainModelRequest(BaseModel):
    dataset_month: str | None = Field(
        default=None,
        description="Dataset month in YYYY-MM. If omitted, uses the latest dataset manifest from MinIO.",
    )
    model_name: str = Field(
        default="credit_risk_quorum_top3",
        description="Versioned model name to be stored in MinIO.",
    )
    minimum_accuracy: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum implementation accuracy required before storing the model.",
    )
    optuna_trials: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Number of Optuna trials per top-3 candidate.",
    )
    threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Decision threshold for classification.",
    )
    model_month: str | None = Field(
        default=None,
        description="Storage month in YYYY-MM for the versioned model artifact.",
    )


class MetricsResponse(BaseModel):
    split: str
    accuracy: float
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    predicted_event_rate: float
    observed_event_rate: float


class BenchmarkCandidateResponse(BaseModel):
    model: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_average_precision_mean: float
    cv_f1_mean: float
    test_accuracy: float
    test_roc_auc: float
    implementation_accuracy: float
    implementation_roc_auc: float


class TunedModelResponse(BaseModel):
    model: str
    optuna_best_cv_roc_auc: float
    test_accuracy: float
    test_roc_auc: float
    implementation_accuracy: float
    implementation_roc_auc: float
    best_params: dict


class TrainModelResponse(BaseModel):
    model_name: str
    object_name: str
    model_month: str
    top3_models: list[str]
    minimum_accuracy: float
    test_metrics: MetricsResponse
    implementation_metrics: MetricsResponse
    benchmark_ranking: list[BenchmarkCandidateResponse]
    tuned_ranking: list[TunedModelResponse]
    message: str


class CreditApplicationRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float | None = None
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float | None = None
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float
    model_name: str = Field(default="credit_risk_quorum_top3")
    model_month: str | None = Field(default=None)


class BusinessDecisionResponse(BaseModel):
    risk_band: str
    recommendation: str
    reasons: list[str]


class PredictionResponse(BaseModel):
    model_name: str
    model_month: str
    default_probability: float
    predicted_label: int
    threshold: float
    decision: BusinessDecisionResponse
