from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from _bootstrap import install_test_stubs

install_test_stubs()

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
class ApiTests(unittest.TestCase):
    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in the current environment")
    def setUp(self) -> None:
        from fastapi.testclient import TestClient
        from credit_risk.api.main import app

        self.app = app
        self.client = TestClient(app)

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in the current environment")
    def test_healthcheck(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in the current environment")
    @patch("credit_risk.api.main.publish_credit_risk_dataset")
    def test_ingest_route_returns_published_version(self, publish_mock) -> None:
        publish_mock.return_value = type(
            "DatasetVersion",
            (),
            {
                "dataset_name": "credit_risk_dataset",
                "month": "2026-04",
                "object_name": "datasets/raw/credit_risk_dataset/2026-04/credit_risk_dataset.csv",
                "uploaded_at": "2026-04-12T00:00:00Z",
            },
        )()

        response = self.client.post(
            "/datasets/credit-risk/ingest",
            json={
                "local_path": "data/credit_risk_dataset.csv",
                "month": "2026-04",
                "overwrite": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "dataset_name": "credit_risk_dataset",
                "month": "2026-04",
                "object_name": "datasets/raw/credit_risk_dataset/2026-04/credit_risk_dataset.csv",
                "uploaded_at": "2026-04-12T00:00:00Z",
                "message": "Dataset ingested to MinIO.",
            },
        )

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in the current environment")
    @patch("credit_risk.api.main.trainer.train")
    def test_train_route_returns_training_summary(self, train_mock) -> None:
        train_mock.return_value = type(
            "TrainingRunResult",
            (),
            {
                "model_name": "credit_risk_quorum_top3",
                "stored_model": type("StoredModel", (), {"object_name": "models/trained/credit_risk_quorum_top3/2026-04/model.pkl", "month": "2026-04"})(),
                "top3_models": ["hist_gradient_boosting", "random_forest", "extra_trees"],
                "minimum_accuracy": 0.80,
                "test_metrics": type("Metrics", (), {"__dict__": {
                    "split": "test", "accuracy": 0.84, "roc_auc": 0.91, "average_precision": 0.85,
                    "precision": 0.74, "recall": 0.69, "f1": 0.71, "predicted_event_rate": 0.20, "observed_event_rate": 0.22
                }})(),
                "implementation_metrics": type("Metrics", (), {"__dict__": {
                    "split": "implementation", "accuracy": 0.83, "roc_auc": 0.90, "average_precision": 0.84,
                    "precision": 0.72, "recall": 0.68, "f1": 0.70, "predicted_event_rate": 0.21, "observed_event_rate": 0.22
                }})(),
                "benchmark_ranking": [],
                "tuned_ranking": [],
            },
        )()

        response = self.client.post(
            "/models/credit-risk/train",
            json={"dataset_month": "2026-04", "optuna_trials": 2, "minimum_accuracy": 0.80},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "credit_risk_quorum_top3")
        self.assertEqual(response.json()["model_month"], "2026-04")

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in the current environment")
    @patch("credit_risk.api.main.prediction_service.predict")
    def test_predict_route_returns_business_prediction(self, predict_mock) -> None:
        predict_mock.return_value = type(
            "PredictionResult",
            (),
            {
                "model_name": "credit_risk_quorum_top3",
                "model_month": "2026-04",
                "default_probability": 0.63,
                "predicted_label": 1,
                "threshold": 0.5,
                "decision": type(
                    "BusinessDecision",
                    (),
                    {"risk_band": "alto", "recommendation": "rejeitar", "reasons": ["historico_de_inadimplencia"]},
                )(),
            },
        )()

        response = self.client.post(
            "/models/credit-risk/predict",
            json={
                "person_age": 32,
                "person_income": 25000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3,
                "loan_intent": "PERSONAL",
                "loan_grade": "C",
                "loan_amnt": 12000,
                "loan_int_rate": 17.5,
                "loan_percent_income": 0.42,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 6,
                "model_name": "credit_risk_quorum_top3",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"]["recommendation"], "rejeitar")
