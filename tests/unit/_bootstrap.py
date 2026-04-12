from __future__ import annotations

import sys
import types


def install_test_stubs() -> None:
    if "pandas" not in sys.modules:
        pandas_module = types.ModuleType("pandas")
        pandas_module.DataFrame = object
        pandas_module.read_csv = lambda *args, **kwargs: None
        sys.modules["pandas"] = pandas_module

    if "kagglehub" not in sys.modules:
        kagglehub_module = types.ModuleType("kagglehub")
        kagglehub_module.dataset_download = lambda *args, **kwargs: ""
        sys.modules["kagglehub"] = kagglehub_module

    if "minio" not in sys.modules:
        minio_module = types.ModuleType("minio")

        class Minio:  # pragma: no cover - stub for import isolation only
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

        minio_module.Minio = Minio
        sys.modules["minio"] = minio_module

    if "minio.error" not in sys.modules:
        minio_error_module = types.ModuleType("minio.error")

        class S3Error(Exception):
            def __init__(self, code: str, *args, **kwargs) -> None:
                super().__init__(*args)
                self.code = code

        minio_error_module.S3Error = S3Error
        sys.modules["minio.error"] = minio_error_module
