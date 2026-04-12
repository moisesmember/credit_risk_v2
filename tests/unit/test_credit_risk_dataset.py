from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from _bootstrap import install_test_stubs

install_test_stubs()

from credit_risk.data.catalog import DatasetCatalog
from credit_risk.datasets.credit_risk import ensure_local_credit_risk_dataset, publish_credit_risk_dataset
from credit_risk.storage.paths import latest_dataset_manifest_name


class FakeStorage:
    def __init__(self, existing_objects: set[str] | None = None) -> None:
        self.bucket = "credit-risk"
        self.existing_objects = set(existing_objects or set())
        self.json_store: dict[str, dict[str, object]] = {}
        self.put_json_calls: list[dict[str, object]] = []

    def object_exists(self, *, object_name: str) -> bool:
        return object_name in self.existing_objects

    def put_json(self, *, object_name: str, payload: dict[str, object]) -> None:
        self.json_store[object_name] = payload
        self.put_json_calls.append({"object_name": object_name, "payload": payload})
        self.existing_objects.add(object_name)

    def get_json(self, *, object_name: str) -> dict[str, object]:
        return self.json_store[object_name]


class CreditRiskDatasetTests(unittest.TestCase):
    @patch("credit_risk.datasets.credit_risk.kagglehub.dataset_download")
    def test_existing_local_dataset_is_reused_without_kaggle_download(self, download_mock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "credit_risk_dataset.csv"
            local_path.write_text("a,b\n1,2\n", encoding="utf-8")

            resolved = ensure_local_credit_risk_dataset(local_path=local_path)

        self.assertEqual(resolved, local_path)
        download_mock.assert_not_called()

    @patch("credit_risk.datasets.credit_risk.kagglehub.dataset_download")
    def test_missing_local_dataset_is_downloaded_from_kaggle(self, download_mock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            kaggle_dir = root / "kaggle-download"
            kaggle_dir.mkdir()
            source_csv = kaggle_dir / "credit_risk_dataset.csv"
            source_csv.write_text("a,b\n1,2\n", encoding="utf-8")

            local_path = root / "data" / "credit_risk_dataset.csv"
            download_mock.return_value = str(kaggle_dir)

            resolved = ensure_local_credit_risk_dataset(local_path=local_path)

            self.assertEqual(resolved, local_path)
            self.assertTrue(local_path.exists())
            self.assertEqual(local_path.read_text(encoding="utf-8"), "a,b\n1,2\n")
            download_mock.assert_called_once()

    @patch("credit_risk.datasets.credit_risk.kagglehub.dataset_download")
    def test_publish_skips_kaggle_and_upload_when_monthly_object_already_exists(self, download_mock) -> None:
        object_name = "datasets/raw/credit_risk_dataset/2026-04/credit_risk_dataset.csv"
        manifest_name = latest_dataset_manifest_name(
            dataset_name="credit_risk_dataset",
            stage="raw",
        )
        storage = FakeStorage(existing_objects={object_name})
        catalog = DatasetCatalog(storage=storage)

        version = publish_credit_risk_dataset(
            local_path=Path("data/credit_risk_dataset.csv"),
            reference_date=datetime(2026, 4, 1),
            catalog=catalog,
        )

        self.assertEqual(version.object_name, object_name)
        self.assertEqual(len(storage.put_json_calls), 1)
        self.assertEqual(storage.put_json_calls[0]["object_name"], manifest_name)
        download_mock.assert_not_called()
