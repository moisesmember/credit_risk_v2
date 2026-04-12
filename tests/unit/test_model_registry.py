from __future__ import annotations

import pickle
import unittest
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from _bootstrap import install_test_stubs

install_test_stubs()

from credit_risk.models.registry import ModelRegistry
from credit_risk.storage.paths import latest_model_manifest_name


class FakeStorage:
    def __init__(self) -> None:
        self.bucket = "credit-risk"
        self.bytes_store: dict[str, bytes] = {}
        self.json_store: dict[str, dict[str, object]] = {}
        self.put_bytes_calls: list[dict[str, object]] = []

    def put_bytes(
        self,
        *,
        object_name: str,
        payload: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.put_bytes_calls.append(
            {
                "object_name": object_name,
                "content_type": content_type,
                "metadata": metadata,
            }
        )
        self.bytes_store[object_name] = payload

    def get_bytes(self, *, object_name: str) -> bytes:
        return self.bytes_store[object_name]

    def put_json(self, *, object_name: str, payload: dict[str, object]) -> None:
        self.json_store[object_name] = payload

    def get_json(self, *, object_name: str) -> dict[str, object]:
        return self.json_store[object_name]

    def list_objects(self, *, prefix: str, recursive: bool = True) -> list[str]:
        return sorted(name for name in self.bytes_store if name.startswith(prefix))


class ModelRegistryTests(unittest.TestCase):
    def test_save_pickle_persists_model_and_manifest(self) -> None:
        storage = FakeStorage()
        registry = ModelRegistry(storage=storage)
        model = {"threshold": 0.42}

        version = registry.save_pickle(
            model=model,
            model_name="credit_risk_rf",
            reference_date=datetime(2026, 4, 1),
        )

        manifest_name = latest_model_manifest_name(model_name="credit_risk_rf", stage="trained")

        self.assertEqual(len(storage.put_bytes_calls), 1)
        self.assertIn("models/trained/credit_risk_rf/2026-04/model.pkl", version.object_name)
        self.assertEqual(storage.json_store[manifest_name]["object_name"], version.object_name)
        self.assertEqual(pickle.loads(storage.bytes_store[version.object_name]), model)

    def test_load_pickle_uses_latest_manifest(self) -> None:
        storage = FakeStorage()
        registry = ModelRegistry(storage=storage)

        object_name = "models/trained/credit_risk_rf/2026-04/model.pkl"
        storage.bytes_store[object_name] = pickle.dumps({"score": 0.99})
        storage.json_store[
            latest_model_manifest_name(model_name="credit_risk_rf", stage="trained")
        ] = {
            "model_name": "credit_risk_rf",
            "artifact_name": "model.pkl",
            "stage": "trained",
            "month": "2026-04",
            "object_name": object_name,
            "uploaded_at": "2026-04-12T00:00:00Z",
            "serializer": "pickle",
        }

        model = registry.load_pickle(model_name="credit_risk_rf")

        self.assertEqual(model, {"score": 0.99})
