from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from credit_risk.models.registry import ModelRegistry
from credit_risk.serving.business import BusinessDecision, derive_business_decision
from credit_risk.training.bundle import CreditRiskQuorumModel


@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    model_month: str
    default_probability: float
    predicted_label: int
    threshold: float
    policy_name: str
    decision: BusinessDecision


class CreditRiskPredictionService:
    def __init__(self, *, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    def load_model(self, *, model_name: str, month: str | None = None) -> CreditRiskQuorumModel:
        model = self.model_registry.load_pickle(model_name=model_name, month=month)
        if not isinstance(model, CreditRiskQuorumModel):
            raise TypeError("Loaded artifact is not a CreditRiskQuorumModel.")
        return model

    def predict(
        self,
        *,
        payload: dict[str, Any],
        model_name: str,
        month: str | None = None,
        policy_name: str = "balanceado",
    ) -> PredictionResult:
        model = self.load_model(model_name=model_name, month=month)
        features = pd.DataFrame([payload])
        probability = float(model.predict_default_probability(features)[0])
        label = int(probability >= model.threshold)
        decision = derive_business_decision(payload, probability, policy_name=policy_name)
        return PredictionResult(
            model_name=model.model_name,
            model_month=model.model_month,
            default_probability=probability,
            predicted_label=label,
            threshold=model.threshold,
            policy_name=decision.policy_name,
            decision=decision,
        )
