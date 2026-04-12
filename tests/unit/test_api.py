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
