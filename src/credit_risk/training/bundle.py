from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CreditRiskQuorumModel:
    model_name: str
    model_month: str
    feature_engineer: Any
    preprocessor: Any
    selected_mask: np.ndarray
    selected_feature_names: list[str]
    top_model_names: list[str]
    trained_models: dict[str, Any]
    threshold: float = 0.5

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        engineered = self.feature_engineer.transform(X)
        processed = self.preprocessor.transform(engineered)
        return processed[:, self.selected_mask]

    def predict_default_probability(self, X: pd.DataFrame) -> np.ndarray:
        selected = self._transform(X)
        probabilities = [
            model.predict_proba(selected)[:, 1]
            for _, model in sorted(self.trained_models.items(), key=lambda item: item[0])
        ]
        return np.mean(np.vstack(probabilities), axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = self.predict_default_probability(X)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        active_threshold = self.threshold if threshold is None else threshold
        return (self.predict_default_probability(X) >= active_threshold).astype(int)
