"""
permeability.py
===============
Depth-resolved permeability prediction from well log data.

Strategy:
1. Compute total porosity (PHIT) from density-neutron crossplot
2. Train XGBoost on [PHIT, GR, RHOB, DT, TVD] → log10(Permeability)
3. Calibrate against ThermoGIS P90/P50/P10 point estimates per well
4. Output depth-resolved permeability profiles with uncertainty bounds
5. Generate SHAP feature importance plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")


# Fluid and matrix constants for porosity computation
RHO_MATRIX = 2.65    # g/cc — quartz sandstone
RHO_FLUID  = 1.00    # g/cc — water
NPHI_MATRIX = 0.0    # v/v — quartz
NPHI_FLUID  = 1.0    # v/v — water


def compute_density_porosity(rhob: np.ndarray) -> np.ndarray:
    """
    Compute density porosity (PHID) from bulk density log.
    PHID = (rho_matrix - RHOB) / (rho_matrix - rho_fluid)
    """
    phid = (RHO_MATRIX - rhob) / (RHO_MATRIX - RHO_FLUID)
    phid = np.clip(phid, 0.0, 0.50)  # physical bounds
    return phid


def compute_neutron_density_porosity(rhob: np.ndarray,
                                     nphi: np.ndarray) -> np.ndarray:
    """
    Compute total porosity from density-neutron crossplot (average method).
    PHIT = (PHID + NPHI) / 2
    """
    phid = compute_density_porosity(rhob)
    phit = (phid + nphi) / 2.0
    phit = np.clip(phit, 0.0, 0.45)
    return phit


def kozeny_carman_permeability(porosity: np.ndarray,
                                grain_size_um: float = 150.0) -> np.ndarray:
    """
    Kozeny-Carman permeability estimate (mD) as baseline / initial guess.
    k = (grain_size^2 * phi^3) / (180 * (1-phi)^2)
    Returns permeability in millidarcies.

    grain_size_um : float
        Mean grain diameter in micrometres (150 µm = medium sandstone)
    """
    phi = np.clip(porosity, 0.001, 0.499)
    d_cm = grain_size_um * 1e-4  # µm → cm
    k_cm2 = (d_cm**2 * phi**3) / (180.0 * (1.0 - phi)**2)
    k_darcy = k_cm2 / 9.869233e-9  # cm² → darcy
    k_md = k_darcy * 1000.0        # darcy → mD
    return np.clip(k_md, 0.001, 10_000.0)


class PermeabilityModel:
    """
    XGBoost-based permeability prediction model for the Rotliegend formation.

    Calibrated against ThermoGIS P90/P50/P10 estimates.
    Outputs depth-resolved permeability with uncertainty.

    Usage
    -----
    model = PermeabilityModel()
    model.fit(rotliegend_df, thermogis_df)
    profiles = model.predict_profiles(rotliegend_df)
    model.save("outputs/models/permeability_model.pkl")
    """

    FEATURES = ["phit", "gr_api", "rhob_gcc", "dt_usft", "TVD_M"]
    QUANTILES = [0.10, 0.50, 0.90]

    def __init__(self, n_estimators: int = 500, max_depth: int = 5,
                 learning_rate: float = 0.03):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = {}
        self.metrics = {}
        self._is_fitted = False
        self.shap_values = None
        self.explainer = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed porosity and prepare feature matrix."""
        df = df.copy()

        # Use imputed versions where available
        rhob = df.get("rhob_gcc_final", df.get("rhob_gcc"))
        nphi_col = "nphi_vv_final" if "nphi_vv_final" in df.columns else "nphi_vv"

        if nphi_col in df.columns and df[nphi_col].notna().any():
            df["phit"] = compute_neutron_density_porosity(
                rhob.values, df[nphi_col].values
            )
        else:
            df["phit"] = compute_density_porosity(rhob.values)

        return df

    def _build_pseudo_labels(self, df: pd.DataFrame,
                              thermogis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build pseudo-labels from ThermoGIS point estimates.

        ThermoGIS provides P90/P50/P10 permeability per well (single value).
        We assign these to depth intervals within the Rotliegend to create
        training targets, then add Kozeny-Carman as additional anchor points.
        """
        df = df.copy()
        df["log_k_target"] = np.nan
        df["k_source"] = "none"

        # Map ThermoGIS well names to our well IDs
        tg_wells = thermogis_df["WELL_ID"].unique() if "WELL_ID" in thermogis_df.columns else []

        for well in df["WELL_ID"].unique():
            mask = (df["WELL_ID"] == well) & df["is_rotliegend"]
            n_rows = mask.sum()

            # Try to get ThermoGIS P50 for this well
            tg_row = None
            if "WELL_ID" in thermogis_df.columns:
                tg_match = thermogis_df[
                    thermogis_df["WELL_ID"].str.upper() == well.upper()
                ]
                if not tg_match.empty:
                    tg_row = tg_match.iloc[0]

            if tg_row is not None:
                # Use ThermoGIS P50 as the representative permeability
                k_p50 = float(tg_row.get("PERM_P50_MD", tg_row.get("K_P50", np.nan)))
                if not np.isnan(k_p50) and k_p50 > 0:
                    df.loc[mask, "log_k_target"] = np.log10(k_p50)
                    df.loc[mask, "k_source"] = "thermogis_p50"
                    print(f"  [{well}] ThermoGIS P50 = {k_p50} mD → assigned to {n_rows} rows")
                    continue

            # Fallback: Kozeny-Carman from porosity
            if mask.sum() > 0 and "phit" in df.columns:
                phit_vals = df.loc[mask, "phit"].values
                k_kc = kozeny_carman_permeability(phit_vals)
                df.loc[mask, "log_k_target"] = np.log10(np.clip(k_kc, 0.001, None))
                df.loc[mask, "k_source"] = "kozeny_carman"
                print(f"  [{well}] Using Kozeny-Carman fallback for {n_rows} rows")

        return df

    def fit(self, df: pd.DataFrame, thermogis_df: pd.DataFrame) -> "PermeabilityModel":
        """
        Train permeability models (P10, P50, P90) on Rotliegend interval data.

        Parameters
        ----------
        df : pd.DataFrame
            Unified schema DataFrame with imputed curves
        thermogis_df : pd.DataFrame
            ThermoGIS data with per-well P90/P50/P10 permeability
        """
        print("\nFitting PermeabilityModel...")
        df = self._prepare_features(df)
        df = self._build_pseudo_labels(df, thermogis_df)

        # Training data: Rotliegend rows with valid features and target
        rotl = df[df["is_rotliegend"]].copy()
        required = self.FEATURES + ["log_k_target"]
        train = rotl[required].dropna()

        X = train[self.FEATURES].values
        y = train["log_k_target"].values
        groups = rotl.loc[train.index, "WELL_ID"].values

        print(f"  Training samples: {len(X):,} from wells: {np.unique(groups).tolist()}")

        for q in self.QUANTILES:
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
            print(f"  Trained Q{int(q*100)} model")

        # SHAP explanation on P50
        self.explainer = shap.TreeExplainer(self.models[0.50])
        self.shap_values = self.explainer.shap_values(X)
        print("  SHAP values computed")

        # Compute OOF metrics (leave-one-well-out)
        logo = LeaveOneGroupOut()
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in logo.split(X, y, groups):
            m = xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, objective="reg:quantileerror",
                quantile_alpha=0.50, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0
            )
            m.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = m.predict(X[val_idx])

        self.metrics = {
            "mae_log10": round(mean_absolute_error(y, oof_preds), 3),
            "r2": round(r2_score(y, oof_preds), 3),
            "n_train": len(y)
        }
        print(f"  LOWO OOF — MAE(log10 k): {self.metrics['mae_log10']}, "
              f"R²: {self.metrics['r2']}")

        self._is_fitted = True
        return self

    def predict_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate depth-resolved permeability profiles for all wells.

        Returns
        -------
        pd.DataFrame
            Input with columns: k_p10_md, k_p50_md, k_p90_md added
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict_profiles()")

        df = self._prepare_features(df)
        df_out = df.copy()

        feature_present = df_out[self.FEATURES].notna().all(axis=1)
        X_all = df_out.loc[feature_present, self.FEATURES].values

        for q in self.QUANTILES:
            label = f"k_p{int(q*100)}_md"
            df_out[label] = np.nan
            if len(X_all) > 0:
                log_k = self.models[q].predict(X_all)
                df_out.loc[feature_present, label] = 10.0 ** log_k  # back to mD

        # Also add Kozeny-Carman as reference
        if "phit" in df_out.columns:
            df_out["k_kc_md"] = kozeny_carman_permeability(df_out["phit"].values)

        print(f"Permeability profiles generated for {df_out['WELL_ID'].nunique()} wells")
        return df_out

    def get_well_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean P10/P50/P90 permeability per well in Rotliegend interval.

        Returns
        -------
        pd.DataFrame
            One row per well with mean permeability statistics
        """
        rotl = df[df["is_rotliegend"]].copy()
        summary = rotl.groupby("WELL_ID").agg(
            k_p10_md=("k_p10_md", "mean"),
            k_p50_md=("k_p50_md", "mean"),
            k_p90_md=("k_p90_md", "mean"),
            phit_mean=("phit", "mean"),
            n_samples=("TVD_M", "count")
        ).round(2)
        return summary

    def save(self, path: str | Path):
        joblib.dump(self, str(path))
        print(f"PermeabilityModel saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PermeabilityModel":
        return joblib.load(str(path))
