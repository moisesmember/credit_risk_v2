from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.datasets.credit_risk import DEFAULT_CREDIT_RISK_DATASET, load_credit_risk_dataset
from credit_risk.models.registry import ModelArtifactVersion, ModelRegistry
from credit_risk.storage.minio_client import MinioStorage
from credit_risk.training.bundle import CreditRiskQuorumModel
from credit_risk.training.feature_engineering import DomainFeatureEngineer


logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "loan_status"
DEFAULT_MODEL_NAME = "credit_risk_quorum_top3"


@dataclass(frozen=True)
class MetricsSnapshot:
    split: str
    accuracy: float
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    predicted_event_rate: float
    observed_event_rate: float


@dataclass(frozen=True)
class CandidateSummary:
    model: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_average_precision_mean: float
    cv_f1_mean: float
    test_accuracy: float
    test_roc_auc: float
    implementation_accuracy: float
    implementation_roc_auc: float


@dataclass(frozen=True)
class TunedModelSummary:
    model: str
    optuna_best_cv_roc_auc: float
    test_accuracy: float
    test_roc_auc: float
    implementation_accuracy: float
    implementation_roc_auc: float
    best_params: dict[str, Any]


@dataclass(frozen=True)
class TrainingRunResult:
    model_name: str
    dataset_name: str
    dataset_month: str | None
    stored_model: ModelArtifactVersion
    top3_models: list[str]
    benchmark_ranking: list[CandidateSummary]
    tuned_ranking: list[TunedModelSummary]
    test_metrics: MetricsSnapshot
    implementation_metrics: MetricsSnapshot
    minimum_accuracy: float
    threshold: float


class CreditRiskTrainer:
    def __init__(
        self,
        *,
        model_registry: ModelRegistry | None = None,
        storage: MinioStorage | None = None,
    ) -> None:
        self.model_registry = model_registry or ModelRegistry()
        self.storage = storage or self.model_registry.storage

    def train(
        self,
        *,
        dataset_month: str | None = None,
        dataset_name: str = DEFAULT_CREDIT_RISK_DATASET,
        model_name: str = DEFAULT_MODEL_NAME,
        minimum_accuracy: float = 0.80,
        optuna_trials: int = 10,
        threshold: float = 0.50,
        reference_date: datetime | None = None,
    ) -> TrainingRunResult:
        logger.info(
            "Starting training for dataset='%s', dataset_month='%s', model='%s'.",
            dataset_name,
            dataset_month,
            model_name,
        )
        df = load_credit_risk_dataset(month=dataset_month, dataset_name=dataset_name)
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)

        train_df, test_df, implementation_df = self._split_data(df)
        feature_engineer = DomainFeatureEngineer()

        X_train = train_df.drop(columns=TARGET)
        y_train = train_df[TARGET]
        X_test = test_df.drop(columns=TARGET)
        y_test = test_df[TARGET]
        X_implementation = implementation_df.drop(columns=TARGET)
        y_implementation = implementation_df[TARGET]

        X_train_fe = feature_engineer.fit_transform(X_train)
        X_test_fe = feature_engineer.transform(X_test)
        X_implementation_fe = feature_engineer.transform(X_implementation)

        preprocessor = self._build_preprocessor(X_train_fe)
        X_train_processed = preprocessor.fit_transform(X_train_fe)
        X_test_processed = preprocessor.transform(X_test_fe)
        X_implementation_processed = preprocessor.transform(X_implementation_fe)
        feature_names = list(preprocessor.get_feature_names_out())

        selected_mask, selected_feature_names = self._select_features(
            X_train_processed=X_train_processed,
            y_train=y_train,
            feature_names=feature_names,
        )
        X_train_selected = X_train_processed[:, selected_mask]
        X_test_selected = X_test_processed[:, selected_mask]
        X_implementation_selected = X_implementation_processed[:, selected_mask]

        benchmark_rows = self._benchmark_candidates(
            X_train_selected=X_train_selected,
            y_train=y_train,
            X_test_selected=X_test_selected,
            y_test=y_test,
            X_implementation_selected=X_implementation_selected,
            y_implementation=y_implementation,
            threshold=threshold,
        )
        top3_model_names = [row.model for row in benchmark_rows[:3]]

        tuned_ranking, trained_models = self._tune_top3(
            top3_model_names=top3_model_names,
            X_train_selected=X_train_selected,
            y_train=y_train,
            X_test_selected=X_test_selected,
            y_test=y_test,
            X_implementation_selected=X_implementation_selected,
            y_implementation=y_implementation,
            optuna_trials=optuna_trials,
            threshold=threshold,
        )

        model_month = self._resolve_month(reference_date)
        bundle = CreditRiskQuorumModel(
            model_name=model_name,
            model_month=model_month,
            feature_engineer=feature_engineer,
            preprocessor=preprocessor,
            selected_mask=selected_mask,
            selected_feature_names=selected_feature_names,
            top_model_names=top3_model_names,
            trained_models=trained_models,
            threshold=threshold,
        )

        test_metrics = self._evaluate_dataframe("test", X_test, y_test, bundle, threshold)
        implementation_metrics = self._evaluate_dataframe(
            "implementation",
            X_implementation,
            y_implementation,
            bundle,
            threshold,
        )

        if implementation_metrics.accuracy < minimum_accuracy:
            raise ValueError(
                "Implementation accuracy below minimum threshold: "
                f"{implementation_metrics.accuracy:.4f} < {minimum_accuracy:.4f}"
            )

        stored_model = self.model_registry.save_pickle(
            model=bundle,
            model_name=model_name,
            artifact_name="model.pkl",
            reference_date=reference_date,
        )
        result = TrainingRunResult(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_month=dataset_month,
            stored_model=stored_model,
            top3_models=top3_model_names,
            benchmark_ranking=benchmark_rows,
            tuned_ranking=tuned_ranking,
            test_metrics=test_metrics,
            implementation_metrics=implementation_metrics,
            minimum_accuracy=minimum_accuracy,
            threshold=threshold,
        )
        self._save_training_report(result)
        return result

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_full_df, implementation_df = train_test_split(
            df,
            test_size=0.10,
            stratify=df[TARGET],
            random_state=RANDOM_STATE,
        )
        train_df, test_df = train_test_split(
            train_full_df,
            test_size=0.1111111111,
            stratify=train_full_df[TARGET],
            random_state=RANDOM_STATE,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), implementation_df.reset_index(drop=True)

    def _build_preprocessor(self, X_train_fe: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X_train_fe.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

    def _select_features(
        self,
        *,
        X_train_processed: np.ndarray,
        y_train: pd.Series,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

        mi_scores = mutual_info_classif(X_train_resampled, y_train_resampled, random_state=RANDOM_STATE)
        selector_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=30,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        selector_model.fit(X_train_resampled, y_train_resampled)

        selection_df = pd.DataFrame(
            {"feature": feature_names, "mutual_info": mi_scores, "rf_importance": selector_model.feature_importances_}
        )
        selection_df["mi_rank_pct"] = selection_df["mutual_info"].rank(pct=True, ascending=True)
        selection_df["rf_rank_pct"] = selection_df["rf_importance"].rank(pct=True, ascending=True)
        selection_df["combined_score"] = 0.5 * selection_df["mi_rank_pct"] + 0.5 * selection_df["rf_rank_pct"]
        selection_df = selection_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        top_k = max(15, int(np.ceil(len(selection_df) * 0.30)))
        selected_feature_names = selection_df.loc[: top_k - 1, "feature"].tolist()
        selected_mask = np.isin(np.array(feature_names), selected_feature_names)
        return selected_mask, selected_feature_names

    def _benchmark_candidates(
        self,
        *,
        X_train_selected: np.ndarray,
        y_train: pd.Series,
        X_test_selected: np.ndarray,
        y_test: pd.Series,
        X_implementation_selected: np.ndarray,
        y_implementation: pd.Series,
        threshold: float,
    ) -> list[CandidateSummary]:
        candidates = {
            "logistic_regression": LogisticRegression(solver="lbfgs", max_iter=4000, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=RANDOM_STATE,
            ),
        }
        benchmark_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        summaries: list[CandidateSummary] = []

        for model_name, estimator in candidates.items():
            pipeline = ImbPipeline(
                steps=[
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("model", estimator),
                ]
            )
            cv_result = cross_validate(
                pipeline,
                X_train_selected,
                y_train,
                cv=benchmark_cv,
                scoring={"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1"},
                n_jobs=-1,
                return_train_score=False,
            )
            pipeline.fit(X_train_selected, y_train)
            test_metrics = self._evaluate_array("test", X_test_selected, y_test, pipeline, threshold)
            implementation_metrics = self._evaluate_array(
                "implementation",
                X_implementation_selected,
                y_implementation,
                pipeline,
                threshold,
            )
            summaries.append(
                CandidateSummary(
                    model=model_name,
                    cv_roc_auc_mean=float(cv_result["test_roc_auc"].mean()),
                    cv_roc_auc_std=float(cv_result["test_roc_auc"].std()),
                    cv_average_precision_mean=float(cv_result["test_average_precision"].mean()),
                    cv_f1_mean=float(cv_result["test_f1"].mean()),
                    test_accuracy=test_metrics.accuracy,
                    test_roc_auc=test_metrics.roc_auc,
                    implementation_accuracy=implementation_metrics.accuracy,
                    implementation_roc_auc=implementation_metrics.roc_auc,
                )
            )
        return sorted(
            summaries,
            key=lambda item: (item.cv_roc_auc_mean, item.test_roc_auc, item.implementation_roc_auc),
            reverse=True,
        )

    def _tune_top3(
        self,
        *,
        top3_model_names: list[str],
        X_train_selected: np.ndarray,
        y_train: pd.Series,
        X_test_selected: np.ndarray,
        y_test: pd.Series,
        X_implementation_selected: np.ndarray,
        y_implementation: pd.Series,
        optuna_trials: int,
        threshold: float,
    ) -> tuple[list[TunedModelSummary], dict[str, Any]]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        summaries: list[TunedModelSummary] = []
        trained_models: dict[str, Any] = {}

        for index, model_name in enumerate(top3_model_names, start=1):
            if optuna_trials > 0:
                sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE + index)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(
                    lambda trial, current_model=model_name: self._optuna_objective(
                        trial=trial,
                        model_name=current_model,
                        X_train_selected=X_train_selected,
                        y_train=y_train,
                        optuna_cv=optuna_cv,
                    ),
                    n_trials=optuna_trials,
                    show_progress_bar=False,
                )
                best_params = study.best_params
                best_value = float(study.best_value)
            else:
                best_params = self._default_params(model_name)
                best_value = float("nan")

            tuned_model = self._build_tunable_model(model_name, best_params)
            tuned_pipeline = ImbPipeline(
                steps=[
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("model", tuned_model),
                ]
            )
            tuned_pipeline.fit(X_train_selected, y_train)
            trained_models[model_name] = tuned_pipeline

            test_metrics = self._evaluate_array("test", X_test_selected, y_test, tuned_pipeline, threshold)
            implementation_metrics = self._evaluate_array(
                "implementation",
                X_implementation_selected,
                y_implementation,
                tuned_pipeline,
                threshold,
            )
            summaries.append(
                TunedModelSummary(
                    model=model_name,
                    optuna_best_cv_roc_auc=best_value,
                    test_accuracy=test_metrics.accuracy,
                    test_roc_auc=test_metrics.roc_auc,
                    implementation_accuracy=implementation_metrics.accuracy,
                    implementation_roc_auc=implementation_metrics.roc_auc,
                    best_params=best_params,
                )
            )
        summaries = sorted(
            summaries,
            key=lambda item: (item.test_roc_auc, item.implementation_roc_auc, item.optuna_best_cv_roc_auc),
            reverse=True,
        )
        return summaries, trained_models

    def _optuna_objective(
        self,
        *,
        trial: optuna.trial.Trial,
        model_name: str,
        X_train_selected: np.ndarray,
        y_train: pd.Series,
        optuna_cv: StratifiedKFold,
    ) -> float:
        params = self._sample_params(trial, model_name)
        estimator = self._build_tunable_model(model_name, params)
        pipeline = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_train_selected,
            y_train,
            cv=optuna_cv,
            scoring="roc_auc",
            n_jobs=-1,
            return_train_score=False,
        )
        return float(scores["test_score"].mean())

    def _sample_params(self, trial: optuna.trial.Trial, model_name: str) -> dict[str, Any]:
        if model_name == "logistic_regression":
            return {"C": trial.suggest_float("C", 1e-3, 50.0, log=True)}
        if model_name in {"random_forest", "extra_trees"}:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        if model_name == "hist_gradient_boosting":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
            }
        raise ValueError(f"Unsupported model: {model_name}")

    def _default_params(self, model_name: str) -> dict[str, Any]:
        if model_name == "logistic_regression":
            return {"C": 1.0}
        if model_name == "random_forest":
            return {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"}
        if model_name == "extra_trees":
            return {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 10, "max_features": "sqrt"}
        if model_name == "hist_gradient_boosting":
            return {
                "learning_rate": 0.05,
                "max_depth": 6,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 1e-4,
            }
        raise ValueError(f"Unsupported model: {model_name}")

    def _build_tunable_model(self, model_name: str, params: dict[str, Any]) -> Any:
        if model_name == "logistic_regression":
            return LogisticRegression(C=params["C"], solver="lbfgs", max_iter=4000, random_state=RANDOM_STATE)
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        if model_name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        if model_name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                max_leaf_nodes=params["max_leaf_nodes"],
                min_samples_leaf=params["min_samples_leaf"],
                l2_regularization=params["l2_regularization"],
                random_state=RANDOM_STATE,
            )
        raise ValueError(f"Unsupported model: {model_name}")

    def _evaluate_dataframe(
        self,
        split: str,
        X: pd.DataFrame,
        y: pd.Series,
        model: CreditRiskQuorumModel,
        threshold: float,
    ) -> MetricsSnapshot:
        probabilities = model.predict_default_probability(X)
        predictions = (probabilities >= threshold).astype(int)
        return self._snapshot(split, y, probabilities, predictions)

    def _evaluate_array(
        self,
        split: str,
        X: np.ndarray,
        y: pd.Series,
        model: Any,
        threshold: float,
    ) -> MetricsSnapshot:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return self._snapshot(split, y, probabilities, predictions)

    def _snapshot(
        self,
        split: str,
        y: pd.Series,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> MetricsSnapshot:
        return MetricsSnapshot(
            split=split,
            accuracy=float(accuracy_score(y, predictions)),
            roc_auc=float(roc_auc_score(y, probabilities)),
            average_precision=float(average_precision_score(y, probabilities)),
            precision=float(precision_score(y, predictions, zero_division=0)),
            recall=float(recall_score(y, predictions, zero_division=0)),
            f1=float(f1_score(y, predictions, zero_division=0)),
            predicted_event_rate=float(predictions.mean()),
            observed_event_rate=float(y.mean()),
        )

    def _save_training_report(self, result: TrainingRunResult) -> None:
        prefix = result.stored_model.object_name.rsplit("/", 1)[0]
        self.storage.put_json(
            object_name=f"{prefix}/training_report.json",
            payload={
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "dataset_month": result.dataset_month,
                "stored_model": asdict(result.stored_model),
                "top3_models": result.top3_models,
                "benchmark_ranking": [asdict(item) for item in result.benchmark_ranking],
                "tuned_ranking": [asdict(item) for item in result.tuned_ranking],
                "test_metrics": asdict(result.test_metrics),
                "implementation_metrics": asdict(result.implementation_metrics),
                "minimum_accuracy": result.minimum_accuracy,
                "threshold": result.threshold,
                "trained_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            },
        )

    def _resolve_month(self, reference_date: datetime | None) -> str:
        current = reference_date or datetime.now(UTC)
        return f"{current.year:04d}-{current.month:02d}"
