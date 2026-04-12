from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DomainFeatureEngineer":
        self.input_features_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        income = transformed["person_income"].replace(0, np.nan)
        age = transformed["person_age"].replace(0, np.nan)
        credit_hist = transformed["cb_person_cred_hist_length"].replace(0, np.nan)
        emp_length = transformed["person_emp_length"]
        loan_share = transformed["loan_percent_income"].replace(0, np.nan)

        transformed["loan_to_income_ratio"] = transformed["loan_amnt"] / income
        transformed["credit_hist_to_age_ratio"] = transformed["cb_person_cred_hist_length"] / age
        transformed["emp_length_to_age_ratio"] = emp_length / age
        transformed["income_per_credit_year"] = transformed["person_income"] / (credit_hist + 1)
        transformed["rate_x_amount"] = transformed["loan_int_rate"] * transformed["loan_amnt"]
        transformed["rate_to_income_share"] = transformed["loan_int_rate"] / loan_share
        transformed["person_income_log"] = np.log1p(transformed["person_income"].clip(lower=0))
        transformed["loan_amnt_log"] = np.log1p(transformed["loan_amnt"].clip(lower=0))
        transformed["person_emp_length_missing_flag"] = transformed["person_emp_length"].isna().astype(int)
        transformed["loan_int_rate_missing_flag"] = transformed["loan_int_rate"].isna().astype(int)
        transformed["high_income_commitment"] = (transformed["loan_percent_income"] >= 0.35).astype(int)
        transformed["has_default_history"] = (transformed["cb_person_default_on_file"] == "Y").astype(int)
        transformed["age_bucket"] = pd.cut(
            transformed["person_age"],
            bins=[0, 25, 35, 45, 60, np.inf],
            labels=["18_25", "26_35", "36_45", "46_60", "60_plus"],
        )
        transformed["employment_bucket"] = pd.cut(
            transformed["person_emp_length"],
            bins=[-np.inf, 0, 2, 5, 10, np.inf],
            labels=["missing_or_zero", "1_2", "3_5", "6_10", "10_plus"],
        )

        ratio_columns = [
            "loan_to_income_ratio",
            "credit_hist_to_age_ratio",
            "emp_length_to_age_ratio",
            "income_per_credit_year",
            "rate_to_income_share",
        ]
        transformed[ratio_columns] = transformed[ratio_columns].replace([np.inf, -np.inf], np.nan)
        return transformed
