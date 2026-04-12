from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from credit_risk.datasets.credit_risk import load_credit_risk_dataset, publish_credit_risk_dataset
from credit_risk.logging_utils import configure_logging


def _build_ingest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the credit risk dataset from Kaggle when needed and publish it to MinIO."
    )
    parser.add_argument(
        "--path",
        default=str(Path("data") / "credit_risk_dataset.csv"),
        help="Local staging path for the CSV. If missing, it is downloaded from Kaggle.",
    )
    parser.add_argument(
        "--month",
        help="Reference month in YYYY-MM format. Defaults to the current month.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the object in MinIO even if the monthly file already exists.",
    )
    return parser


def _build_read_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read the unified credit risk dataset directly from MinIO."
    )
    parser.add_argument(
        "--month",
        help="Reference month in YYYY-MM format. If omitted, uses the latest published dataset.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to print.",
    )
    return parser


def ingest_credit_risk_dataset_main() -> None:
    configure_logging()
    parser = _build_ingest_parser()
    args = parser.parse_args()

    reference_date = None
    if args.month:
        reference_date = datetime.strptime(f"{args.month}-01", "%Y-%m-%d")

    version = publish_credit_risk_dataset(
        local_path=Path(args.path),
        reference_date=reference_date,
        overwrite=args.force,
    )
    print(f"Object: {version.object_name}")
    print(f"Published dataset month: {version.month}")


def read_credit_risk_dataset_main() -> None:
    configure_logging()
    parser = _build_read_parser()
    args = parser.parse_args()

    df = load_credit_risk_dataset(month=args.month)
    print(df.head(args.rows).to_string(index=False))
    print(f"\nshape={df.shape}")
