"""
imputer.py
==========
ML-based log curve imputation using Quantile XGBoost.
Fills missing petrophysical log curves (NPHI, DT, RS, etc.)
by training on wells where both predictor and target curves exist,
then predicting on wells where the target is missing.

Produces P10 / P50 / P90 imputed values for uncertainty propagation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")


# Features used to predict each target curve
IMPUTATION_CONFIG = {
    "nphi_vv": {
        "features": ["gr_api", "rhob_gcc", "drho_gcc", "TVD_M"],
        "description": "Neutron Porosity predicted from GR, RHOB, DRHO, TVD"
    },
    "dt_usft": {
        "features": ["gr_api", "rhob_gcc", "nphi_vv", "TVD_M"],
        "description": "Compressional Slowness predicted from GR, RHOB, NPHI, TVD"
    },
    "dts_usft": {
        "features": ["gr_api", "rhob_gcc", "dt_usft", "TVD_M"],
        "description": "Shear Slowness predicted from GR, RHOB, DT, TVD"
    },
    "rs_ohmm": {
        "features": ["gr_api", "rhob_gcc", "nphi_vv", "TVD_M"],
        "description": "Shallow Resistivity predicted from GR, RHOB, NPHI, TVD"
    },
}

QUANTILES = [0.10, 0.50, 0.90]
QUANTILE_LABELS = {0.10: "p10", 0.50: "p50", 0.90: "p90"}


class LogImputer:
    """
    Imputes missing well log curves using XGBoost Quantile Regression.

    Usage
    -----
    imputer = LogImputer()
    imputer.fit(unified_df, target_curve="nphi_vv")
    df_imputed = imputer.transform(unified_df)
    imputer.save("outputs/models/nphi_imputer.pkl")
    """

    def __init__(self, target_curve: str, n_estimators: int = 300,
                 max_depth: int = 5, learning_rate: float = 0.05):
        if target_curve not in IMPUTATION_CONFIG:
            raise ValueError(
                f"Unknown target curve '{target_curve}'. "
                f"Available: {list(IMPUTATION_CONFIG.keys())}"
            )
        self.target_curve = target_curve
        self.features = IMPUTATION_CONFIG[target_curve]["features"]
        self.description = IMPUTATION_CONFIG[target_curve]["description"]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = {}   # keyed by quantile
        self.metrics = {}
        self._is_fitted = False

    def _get_training_data(self, df: pd.DataFrame):
        """Extract rows where both features and target are non-null."""
        cols = self.features + [self.target_curve]
        # Only use Rotliegend interval for training (most relevant geology)
        if "is_rotliegend" in df.columns:
            train_df = df[df["is_rotliegend"]].copy()
        else:
            train_df = df.copy()

        train_df = train_df[cols].dropna()
        X = train_df[self.features].values
        y = train_df[self.target_curve].values
        print(f"  Training data: {len(X):,} rows with all curves present")
        return X, y

    def fit(self, df: pd.DataFrame) -> "LogImputer":
        """
        Train one XGBoost model per quantile on available data.

        Parameters
        ----------
        df : pd.DataFrame
            Unified schema DataFrame (all wells combined)
        """
        print(f"\nFitting LogImputer for: {self.target_curve}")
        print(f"  {self.description}")
        print(f"  Features: {self.features}")

        X, y = self._get_training_data(df)

        if len(X) < 50:
            print(f"  Warning: Only {len(X)} training samples — imputation may be unreliable")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))

        for q in QUANTILES:
            label = QUANTILE_LABELS[q]
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective="reg:quantileerror",
                quantile_alpha=q,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X, y)
            self.models[q] = model

            # OOF validation on P50 only
            if q == 0.50:
                for train_idx, val_idx in kf.split(X):
                    val_model = xgb.XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        objective="reg:quantileerror",
                        quantile_alpha=0.50,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                    val_model.fit(X[train_idx], y[train_idx])
                    oof_preds[val_idx] = val_model.predict(X[val_idx])

                self.metrics = {
                    "mae": round(mean_absolute_error(y, oof_preds), 4),
                    "r2": round(r2_score(y, oof_preds), 4),
                    "n_train": len(y)
                }
                print(f"  P50 OOF — MAE: {self.metrics['mae']}, R²: {self.metrics['r2']}")

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict imputed values for rows where target curve is missing.
        Adds columns: {target}_imp_p10, {target}_imp_p50, {target}_imp_p90.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame with imputation columns added
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .transform()")

        df_out = df.copy()
        t = self.target_curve

        # Identify rows where target is missing but all features are present
        missing_mask = df_out[t].isna()
        feature_present = df_out[self.features].notna().all(axis=1)
        impute_mask = missing_mask & feature_present

        n_impute = impute_mask.sum()
        print(f"\nTransforming {t}: {n_impute:,} rows to impute")

        for q in QUANTILES:
            label = QUANTILE_LABELS[q]
            col_name = f"{t}_imp_{label}"
            df_out[col_name] = np.nan

            if n_impute > 0:
                X_pred = df_out.loc[impute_mask, self.features].values
                preds = self.models[q].predict(X_pred)
                df_out.loc[impute_mask, col_name] = preds

            # Fill with original where it exists (so the column is complete)
            df_out.loc[~missing_mask, col_name] = df_out.loc[~missing_mask, t]

        # Create a single "best estimate" column that merges original + P50 imputed
        df_out[f"{t}_final"] = df_out[t].fillna(df_out[f"{t}_imp_p50"])

        return df_out

    def save(self, path: str | Path):
        """Save fitted imputer to disk."""
        joblib.dump(self, str(path))
        print(f"Imputer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LogImputer":
        """Load a fitted imputer from disk."""
        return joblib.load(str(path))


def run_all_imputers(df: pd.DataFrame,
                     save_dir: str | Path = None) -> pd.DataFrame:
    """
    Convenience function: fit and apply all configured imputers in sequence.
    Each imputer can use the output of the previous one (e.g. NPHI imputed
    before DT, since DT uses NPHI as a feature).

    Parameters
    ----------
    df : pd.DataFrame
        Unified schema DataFrame
    save_dir : str or Path, optional
        Directory to save fitted models

    Returns
    -------
    pd.DataFrame
        Input with all imputation columns added
    """
    df_out = df.copy()

    # Order matters — impute in dependency order
    order = ["nphi_vv", "dt_usft", "dts_usft", "rs_ohmm"]

    for target in order:
        # Temporarily fill feature with imputed P50 from previous step if available
        features = IMPUTATION_CONFIG[target]["features"]
        for feat in features:
            final_col = f"{feat}_final"
            if final_col in df_out.columns and feat in df_out.columns:
                df_out[feat] = df_out[feat].fillna(df_out[final_col])

        imputer = LogImputer(target_curve=target)
        # Only fit if we have some training data for this curve
        if df_out[target].notna().sum() >= 50:
            imputer.fit(df_out)
            df_out = imputer.transform(df_out)

            if save_dir:
                path = Path(save_dir) / f"{target}_imputer.pkl"
                imputer.save(path)
        else:
            print(f"\nSkipping {target} — insufficient training data "
                  f"({df_out[target].notna().sum()} rows)")

    return df_out
