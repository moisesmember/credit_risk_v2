from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import logging
import shutil

import kagglehub
import pandas as pd

from credit_risk.data.catalog import DatasetCatalog, DatasetVersion
from credit_risk.storage.paths import dataset_object_name, month_key

DEFAULT_CREDIT_RISK_DATASET = "credit_risk_dataset"
DEFAULT_LOCAL_DATASET_PATH = Path("data/credit_risk_dataset.csv")
DEFAULT_KAGGLE_DATASET = "laotse/credit-risk-dataset"
DEFAULT_DATASET_FILENAME = "credit_risk_dataset.csv"


logger = logging.getLogger(__name__)


def ensure_local_credit_risk_dataset(
    *,
    local_path: str | Path = DEFAULT_LOCAL_DATASET_PATH,
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
) -> Path:
    target_path = Path(local_path)
    if target_path.exists():
        logger.info("Using existing local dataset at '%s'.", target_path)
        return target_path

    logger.info(
        "Local dataset not found at '%s'. Downloading '%s' from Kaggle.",
        target_path,
        kaggle_dataset,
    )
    downloaded_dir = Path(kagglehub.dataset_download(kaggle_dataset))
    dataset_candidates = sorted(downloaded_dir.rglob(dataset_filename))
    if not dataset_candidates:
        csv_candidates = sorted(downloaded_dir.rglob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(
                f"No CSV file found after downloading Kaggle dataset '{kaggle_dataset}'."
            )
        source_path = csv_candidates[0]
    else:
        source_path = dataset_candidates[0]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    logger.info("Copied Kaggle dataset from '%s' to '%s'.", source_path, target_path)
    return target_path


def publish_credit_risk_dataset(
    *,
    local_path: str | Path = DEFAULT_LOCAL_DATASET_PATH,
    dataset_name: str = DEFAULT_CREDIT_RISK_DATASET,
    reference_date: date | datetime | None = None,
    catalog: DatasetCatalog | None = None,
    overwrite: bool = False,
) -> DatasetVersion:
    active_catalog = catalog or DatasetCatalog()
    resolved_path = Path(local_path)
    target_object_name = dataset_object_name(
        dataset_name=dataset_name,
        filename=resolved_path.name,
        month=month_key(reference_date),
        stage="raw",
    )

    if active_catalog.storage.object_exists(object_name=target_object_name) and not overwrite:
        logger.info(
            "Dataset object '%s' already exists in MinIO. Skipping Kaggle download and upload.",
            target_object_name,
        )
        return active_catalog.register_existing_dataset(
            dataset_name=dataset_name,
            filename=resolved_path.name,
            stage="raw",
            reference_date=reference_date,
        )

    resolved_local_path = ensure_local_credit_risk_dataset(local_path=local_path)
    return active_catalog.publish_local_dataset(
        local_path=resolved_local_path,
        dataset_name=dataset_name,
        stage="raw",
        reference_date=reference_date,
        content_type="text/csv",
        overwrite=overwrite,
    )


def load_credit_risk_dataset(
    *,
    month: str | None = None,
    dataset_name: str = DEFAULT_CREDIT_RISK_DATASET,
    catalog: DatasetCatalog | None = None,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    active_catalog = catalog or DatasetCatalog()
    logger.info(
        "Loading credit risk dataset '%s' from MinIO%s.",
        dataset_name,
        f" for month '{month}'" if month else " using the latest manifest",
    )
    return active_catalog.read_csv(
        dataset_name=dataset_name,
        stage="raw",
        month=month,
        **read_csv_kwargs,
    )
