from __future__ import annotations

import tempfile
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

from credit_risk.data.catalog import DatasetCatalog
from credit_risk.storage.paths import latest_dataset_manifest_name


class FakeStorage:
    def __init__(self, existing_objects: set[str] | None = None) -> None:
        self.bucket = "credit-risk"
        self.existing_objects = set(existing_objects or set())
        self.put_file_calls: list[dict[str, object]] = []
        self.put_json_calls: list[dict[str, object]] = []
        self.json_store: dict[str, dict[str, object]] = {}

    def object_exists(self, *, object_name: str) -> bool:
        return object_name in self.existing_objects

    def put_file(self, **kwargs: object) -> None:
        self.put_file_calls.append(kwargs)
        self.existing_objects.add(str(kwargs["object_name"]))

    def put_json(self, *, object_name: str, payload: dict[str, object]) -> None:
        self.put_json_calls.append({"object_name": object_name, "payload": payload})
        self.json_store[object_name] = payload
        self.existing_objects.add(object_name)

    def get_json(self, *, object_name: str) -> dict[str, object]:
        return self.json_store[object_name]

    def list_objects(self, *, prefix: str, recursive: bool = True) -> list[str]:
        return sorted(name for name in self.existing_objects if name.startswith(prefix))

    def get_bytes(self, *, object_name: str) -> bytes:
        raise NotImplementedError


class DatasetCatalogTests(unittest.TestCase):
    def test_publish_local_dataset_uploads_and_writes_manifest(self) -> None:
        storage = FakeStorage()
        catalog = DatasetCatalog(storage=storage)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "credit_risk_dataset.csv"
            dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")

            version = catalog.publish_local_dataset(
                local_path=dataset_path,
                dataset_name="credit_risk_dataset",
                reference_date=datetime(2026, 4, 1),
            )

        self.assertEqual(len(storage.put_file_calls), 1)
        self.assertEqual(len(storage.put_json_calls), 1)
        self.assertEqual(version.filename, "credit_risk_dataset.csv")
        self.assertEqual(version.object_name, "datasets/raw/credit_risk_dataset/2026-04/credit_risk_dataset.csv")

    def test_publish_local_dataset_skips_upload_when_object_already_exists(self) -> None:
        month = "2026-04"
        existing_object = f"datasets/raw/credit_risk_dataset/{month}/credit_risk_dataset.csv"
        manifest_name = latest_dataset_manifest_name(
            dataset_name="credit_risk_dataset",
            stage="raw",
        )
        storage = FakeStorage(existing_objects={existing_object, manifest_name})
        storage.json_store[manifest_name] = {
            "dataset_name": "credit_risk_dataset",
            "filename": "credit_risk_dataset.csv",
            "stage": "raw",
            "month": month,
            "object_name": existing_object,
            "uploaded_at": "2026-04-12T00:00:00Z",
        }
        catalog = DatasetCatalog(storage=storage)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "credit_risk_dataset.csv"
            dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")

            version = catalog.publish_local_dataset(
                local_path=dataset_path,
                dataset_name="credit_risk_dataset",
                reference_date=datetime(2026, 4, 1),
            )

        self.assertEqual(version.object_name, existing_object)
        self.assertEqual(version.uploaded_at, "2026-04-12T00:00:00Z")
        self.assertEqual(storage.put_file_calls, [])
        self.assertEqual(storage.put_json_calls, [])

    def test_publish_local_dataset_refreshes_stale_manifest_without_reuploading_file(self) -> None:
        month = "2026-04"
        existing_object = f"datasets/raw/credit_risk_dataset/{month}/credit_risk_dataset.csv"
        manifest_name = latest_dataset_manifest_name(
            dataset_name="credit_risk_dataset",
            stage="raw",
        )
        storage = FakeStorage(existing_objects={existing_object, manifest_name})
        storage.json_store[manifest_name] = {
            "dataset_name": "credit_risk_dataset",
            "filename": "credit_risk_dataset.csv",
            "stage": "raw",
            "month": "2026-03",
            "object_name": "datasets/raw/credit_risk_dataset/2026-03/credit_risk_dataset.csv",
            "uploaded_at": "2026-03-12T00:00:00Z",
        }
        catalog = DatasetCatalog(storage=storage)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "credit_risk_dataset.csv"
            dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")

            version = catalog.publish_local_dataset(
                local_path=dataset_path,
                dataset_name="credit_risk_dataset",
                reference_date=datetime(2026, 4, 1),
            )

        self.assertEqual(version.object_name, existing_object)
        self.assertEqual(storage.put_file_calls, [])
        self.assertEqual(len(storage.put_json_calls), 1)
