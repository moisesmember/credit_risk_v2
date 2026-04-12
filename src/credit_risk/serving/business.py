from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionPolicy:
    name: str
    low_risk_max_probability: float
    medium_risk_max_probability: float


@dataclass(frozen=True)
class BusinessDecision:
    policy_name: str
    risk_band: str
    recommendation: str
    reasons: list[str]


DECISION_POLICIES: dict[str, DecisionPolicy] = {
    "conservador": DecisionPolicy(
        name="conservador",
        low_risk_max_probability=0.12,
        medium_risk_max_probability=0.35,
    ),
    "balanceado": DecisionPolicy(
        name="balanceado",
        low_risk_max_probability=0.20,
        medium_risk_max_probability=0.50,
    ),
    "ousado": DecisionPolicy(
        name="ousado",
        low_risk_max_probability=0.30,
        medium_risk_max_probability=0.65,
    ),
}


def get_decision_policy(policy_name: str = "balanceado") -> DecisionPolicy:
    normalized = policy_name.strip().lower()
    if normalized not in DECISION_POLICIES:
        supported = ", ".join(sorted(DECISION_POLICIES))
        raise ValueError(f"Unsupported decision policy '{policy_name}'. Supported values: {supported}.")
    return DECISION_POLICIES[normalized]


def derive_business_decision(
    payload: dict[str, Any],
    probability: float,
    *,
    policy_name: str = "balanceado",
) -> BusinessDecision:
    policy = get_decision_policy(policy_name)
    reasons: list[str] = []
    if payload.get("cb_person_default_on_file") == "Y":
        reasons.append("historico_de_inadimplencia")
    if payload.get("loan_percent_income") is not None and float(payload["loan_percent_income"]) >= 0.35:
        reasons.append("alto_comprometimento_da_renda")
    if payload.get("loan_int_rate") is not None and float(payload["loan_int_rate"]) >= 15.0:
        reasons.append("taxa_de_juros_elevada")
    if payload.get("person_income") is not None and payload.get("loan_amnt") is not None:
        if float(payload["loan_amnt"]) / max(float(payload["person_income"]), 1.0) >= 0.40:
            reasons.append("valor_do_emprestimo_alto_para_a_renda")

    if probability < policy.low_risk_max_probability:
        return BusinessDecision(policy.name, "baixo", "aprovar", reasons or ["risco_controlado"])
    if probability < policy.medium_risk_max_probability:
        return BusinessDecision(policy.name, "medio", "revisao_manual", reasons or ["risco_intermediario"])
    return BusinessDecision(policy.name, "alto", "rejeitar", reasons or ["risco_elevado"])
