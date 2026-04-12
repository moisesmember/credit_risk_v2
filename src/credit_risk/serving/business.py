from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BusinessDecision:
    risk_band: str
    recommendation: str
    reasons: list[str]


def derive_business_decision(payload: dict[str, Any], probability: float) -> BusinessDecision:
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

    if probability < 0.20:
        return BusinessDecision("baixo", "aprovar", reasons or ["risco_controlado"])
    if probability < 0.50:
        return BusinessDecision("medio", "revisao_manual", reasons or ["risco_intermediario"])
    return BusinessDecision("alto", "rejeitar", reasons or ["risco_elevado"])
