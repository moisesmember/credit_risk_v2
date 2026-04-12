from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from credit_risk.serving.business import derive_business_decision


class BusinessPolicyTests(unittest.TestCase):
    def test_low_risk_application_returns_approve(self) -> None:
        decision = derive_business_decision(
            {
                "cb_person_default_on_file": "N",
                "loan_percent_income": 0.10,
                "loan_int_rate": 8.0,
                "person_income": 80000,
                "loan_amnt": 5000,
            },
            0.12,
        )

        self.assertEqual(decision.risk_band, "baixo")
        self.assertEqual(decision.recommendation, "aprovar")

    def test_high_risk_application_returns_reject_with_reasons(self) -> None:
        decision = derive_business_decision(
            {
                "cb_person_default_on_file": "Y",
                "loan_percent_income": 0.45,
                "loan_int_rate": 19.0,
                "person_income": 20000,
                "loan_amnt": 12000,
            },
            0.76,
        )

        self.assertEqual(decision.risk_band, "alto")
        self.assertEqual(decision.recommendation, "rejeitar")
        self.assertIn("historico_de_inadimplencia", decision.reasons)
