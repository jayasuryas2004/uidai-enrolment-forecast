"""
phase4_lgbm_experiment.py
=========================

LightGBM experiment for Phase-4 UIDAI enrolment forecasting.

╔════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  EXPERIMENTAL CODE - NOT USED IN PRODUCTION                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║  This module is used ONLY for offline experimentation and CV comparison   ║
║  against the official XGBoost v3 model. It is NOT wired into inference    ║
║  or the Streamlit dashboard.                                               ║
║                                                                            ║
║  EXPERIMENT RESULTS (Jan 2026):                                            ║
║    - XGBoost v3: R² = 0.955, MAE = 116.6 (official, production)           ║
║    - LightGBM:   R² = 0.865, MAE = 242.8 (experiment, not production)     ║
║                                                                            ║
║  CONCLUSION: XGBoost v3 significantly outperforms LightGBM on this        ║
║  dataset. No further LightGBM tuning planned. Focus moves to SHAP         ║
║  explainability and uncertainty quantification.                            ║
╚════════════════════════════════════════════════════════════════════════════╝

**PURPOSE:**
Provide an alternative model (LightGBM) to compare against the frozen
Phase-4 v3 XGBoost baseline. This module:
    - Reuses the SAME feature engineering functions as v3 XGBoost
    - Reuses the SAME expanding-window CV split utility
    - Saves metrics to a SEPARATE experimental artifacts path
    - Does NOT modify any production v3 files

**USAGE:**
    from src.models.phase4_lgbm_experiment import Phase4LGBMExperiment

    exp = Phase4LGBMExperiment(params={"n_estimators": 700})
    cv_results, oof_df = exp.fit_cv()
    summary = exp.summarize_cv(cv_results)
    exp.save_cv_metrics(summary, Path("artifacts/phase4_lgbm_experiment_metrics.json"))

**IMPORTANT:**
    - This is EXPERIMENTAL code. NOT integrated into production.
    - All outputs go to artifacts/experiments/ or artifacts/phase4_lgbm_*
    - The official production model is XGBoost v3 (PHASE4_V3_FINAL)

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Import existing utilities (same as v3 XGBoost)
try:
    from src.validation.time_series_cv import generate_expanding_folds
    from src.features.timeseries_features import (
        add_lag_features,
        add_rolling_features,
        add_diff_features,
    )
    from src.features.holiday_features import (
        build_holiday_calendar,
        add_holiday_features,
        add_month_sin_cos,
    )
    from src.features.policy_features import (
        add_policy_phase_features,
        add_time_trend_features,
    )
    from src.features.timeseries_lag_utils import get_recommended_lags, DEFAULT_LAGS
except ImportError:
    from validation.time_series_cv import generate_expanding_folds
    from features.timeseries_features import (
        add_lag_features,
        add_rolling_features,
        add_diff_features,
    )
    from features.holiday_features import (
        build_holiday_calendar,
        add_holiday_features,
        add_month_sin_cos,
    )
    from features.policy_features import (
        add_policy_phase_features,
        add_time_trend_features,
    )
    from features.timeseries_lag_utils import get_recommended_lags, DEFAULT_LAGS


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data/processed/district_month_modeling.csv")
DATE_COL = "month_date"
TARGET_COL = "total_enrolment"
GROUP_COLS = ["state", "district"]

# Default LightGBM hyperparameters (modest, no Optuna tuning)
DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "n_estimators": 700,
    "learning_rate": 0.04,
    "num_leaves": 64,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "min_child_samples": 20,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CVFoldResult:
    """Results from a single CV fold."""
    fold_id: int
    r2: float
    mae: float
    rmse: float
    train_size: int
    val_size: int


@dataclass
class Phase4LGBMExperiment:
    """
    LightGBM experiment class for Phase-4 UIDAI forecasting.
    
    Reuses the same feature engineering and CV utilities as v3 XGBoost.
    """
    params: Dict[str, Any] = field(default_factory=lambda: DEFAULT_LGBM_PARAMS.copy())
    target_col: str = TARGET_COL
    prediction_col: str = "y_pred_lgbm"
    date_col: str = DATE_COL
    group_cols: List[str] = field(default_factory=lambda: GROUP_COLS.copy())
    data_path: Path = DATA_PATH
    
    # CV configuration (same as v3)
    n_folds: int = 4
    gap_months: int = 1
    min_train_months: int = 3
    
    # Feature configuration (same as v3)
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 6])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6])
    
    def _make_model(self) -> LGBMRegressor:
        """Create a LightGBM model with configured parameters."""
        return LGBMRegressor(**self.params)
    
    def _load_data(self) -> pd.DataFrame:
        """Load the Phase-4 dataset."""
        df = pd.read_csv(self.data_path)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic calendar features (leakage-safe)."""
        df = df.copy()
        df["year"] = df[self.date_col].dt.year
        df["month"] = df[self.date_col].dt.month
        df["quarter"] = df[self.date_col].dt.quarter
        df["financial_year"] = np.where(df["month"] >= 4, df["year"], df["year"] - 1)
        return df
    
    def _build_features_for_fold(
        self,
        df_train: pd.DataFrame,
        df_apply: pd.DataFrame,
        holiday_calendar: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build features for a single CV fold (same logic as v3 XGBoost).
        
        CRITICAL: Uses only training data for statistics/imputation.
        """
        df = df_apply.copy()
        stats = {}
        
        # 1. Calendar features (safe - date only)
        df = self._add_calendar_features(df)
        
        # 2. Holiday/event features (safe - calendar only)
        df = add_holiday_features(df, self.date_col, holiday_calendar)
        
        # 3. Policy/intervention features (safe - calendar only)
        df = add_policy_phase_features(df, self.date_col)
        df = add_time_trend_features(df, self.date_col)
        
        # 4. Cyclical month encoding (safe - calendar only)
        df = add_month_sin_cos(df, self.date_col)
        
        # 5. Lag features (LEAKAGE-SAFE: uses shift)
        df = add_lag_features(
            df, self.group_cols, self.date_col, self.target_col,
            self.lags, fill_na=False
        )
        
        # 6. Rolling features (LEAKAGE-SAFE: uses shift before rolling)
        df = add_rolling_features(
            df, self.group_cols, self.date_col, self.target_col,
            self.rolling_windows, include_std=True, fill_na=False
        )
        
        # 7. Difference features (safe - uses past values)
        df = add_diff_features(
            df, self.group_cols, self.date_col, self.target_col,
            [1], fill_na=False
        )
        
        # 8. Group statistics for imputation (from TRAIN only)
        group_means = df_train.groupby(self.group_cols, observed=True)[self.target_col].mean()
        group_stds = df_train.groupby(self.group_cols, observed=True)[self.target_col].std()
        global_mean = df_train[self.target_col].mean()
        global_std = df_train[self.target_col].std()
        
        stats["group_means"] = group_means
        stats["group_stds"] = group_stds
        stats["global_mean"] = global_mean
        stats["global_std"] = global_std
        
        # 9. Fill NaN values using TRAIN statistics
        lag_cols = [c for c in df.columns if "_lag_" in c]
        rolling_mean_cols = [c for c in df.columns if "_rolling_" in c and "_mean" in c]
        rolling_std_cols = [c for c in df.columns if "_rolling_" in c and "_std" in c]
        diff_cols = [c for c in df.columns if "_diff_" in c]
        
        # Create group key for mapping
        if self.group_cols:
            df["_group_key"] = df[self.group_cols].apply(tuple, axis=1)
            group_mean_map = group_means.to_dict()
            group_std_map = group_stds.to_dict()
            
            for col in lag_cols + rolling_mean_cols:
                fill_values = df["_group_key"].map(group_mean_map).fillna(global_mean)
                df[col] = df[col].fillna(fill_values)
            
            for col in rolling_std_cols:
                fill_values = df["_group_key"].map(group_std_map).fillna(global_std)
                df[col] = df[col].fillna(fill_values)
            
            df = df.drop(columns=["_group_key"])
        else:
            for col in lag_cols + rolling_mean_cols:
                df[col] = df[col].fillna(global_mean)
            for col in rolling_std_cols:
                df[col] = df[col].fillna(global_std)
        
        # Diff features: fill with 0 (no change)
        for col in diff_cols:
            df[col] = df[col].fillna(0)
        
        return df, stats
    
    def _encode_categoricals(
        self,
        df: pd.DataFrame,
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Encode categorical columns."""
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if encoders is None:
            encoders = {}
        
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                le.fit(df[col].astype(str).unique())
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is None:
                    continue
            
            # Handle unseen categories
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        
        return df, encoders
    
    def _prepare_xy(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features (X) and target (y)."""
        if exclude_cols is None:
            exclude_cols = []
        
        drop_cols = [self.target_col, self.date_col] + exclude_cols
        drop_cols = [c for c in drop_cols if c in df.columns]
        
        X = df.drop(columns=drop_cols)
        y = df[self.target_col]
        
        return X, y
    
    def fit_cv(self) -> Tuple[List[CVFoldResult], pd.DataFrame]:
        """
        Run expanding-window CV using the same folds as Phase-4 v3 XGBoost.
        
        Returns:
            - List of CVFoldResult objects
            - DataFrame with out-of-fold predictions (for optional ensembling)
        """
        logger = setup_logging()
        
        logger.info("=" * 70)
        logger.info("PHASE-4 LightGBM EXPERIMENT")
        logger.info("=" * 70)
        
        # Load data
        logger.info(f"Loading data from: {self.data_path}")
        df = self._load_data()
        logger.info(f"Loaded: {len(df):,} rows")
        
        # Build holiday calendar
        logger.info("Building holiday calendar...")
        start_date = (df[self.date_col].min() - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        end_date = (df[self.date_col].max() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        holiday_calendar = build_holiday_calendar(start_date, end_date)
        
        # Generate CV folds
        logger.info(f"Generating {self.n_folds} expanding-window folds...")
        folds = generate_expanding_folds(
            df=df,
            date_col=self.date_col,
            n_folds=self.n_folds,
            gap_months=self.gap_months,
            min_train_months=self.min_train_months,
        )
        
        oof_preds: List[pd.DataFrame] = []
        cv_results: List[CVFoldResult] = []
        feature_names: Optional[List[str]] = None
        
        for fold_id, (train_mask, val_mask) in enumerate(folds, start=1):
            logger.info(f"\n{'='*40}")
            logger.info(f"FOLD {fold_id}/{self.n_folds}")
            logger.info(f"{'='*40}")
            
            df_train = df[train_mask].copy()
            df_val = df[val_mask].copy()
            
            logger.info(f"Train: {len(df_train):,} rows")
            logger.info(f"Val: {len(df_val):,} rows")
            
            # Build features (using train stats for imputation)
            df_train_feat, _ = self._build_features_for_fold(
                df_train, df_train, holiday_calendar
            )
            df_val_feat, _ = self._build_features_for_fold(
                df_train, df_val, holiday_calendar
            )
            
            # Encode categoricals (fit on train only)
            df_train_feat, encoders = self._encode_categoricals(df_train_feat, fit=True)
            df_val_feat, _ = self._encode_categoricals(df_val_feat, encoders=encoders, fit=False)
            
            # Prepare X, y
            X_train, y_train = self._prepare_xy(df_train_feat)
            X_val, y_val = self._prepare_xy(df_val_feat)
            
            if feature_names is None:
                feature_names = X_train.columns.tolist()
                logger.info(f"Features: {len(feature_names)}")
            
            # Train LightGBM
            model = self._make_model()
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    # Early stopping callback
                    __import__("lightgbm").early_stopping(
                        stopping_rounds=50, verbose=False
                    )
                ],
            )
            
            # Predict on validation
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info(f"R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
            
            cv_results.append(CVFoldResult(
                fold_id=fold_id,
                r2=r2,
                mae=mae,
                rmse=rmse,
                train_size=len(df_train),
                val_size=len(df_val),
            ))
            
            # Store OOF predictions
            oof_preds.append(pd.DataFrame({
                "fold_id": fold_id,
                "row_id": df_val.index,
                self.target_col: y_val.values,
                self.prediction_col: y_pred,
            }))
        
        # Combine OOF predictions
        oof_df = pd.concat(oof_preds, axis=0).set_index("row_id").sort_index()
        
        logger.info("\n" + "=" * 70)
        logger.info("CV COMPLETE")
        logger.info("=" * 70)
        
        return cv_results, oof_df
    
    @staticmethod
    def summarize_cv(cv_results: List[CVFoldResult]) -> Dict[str, Any]:
        """Summarize CV results into mean ± std format."""
        r2_vals = np.array([r.r2 for r in cv_results])
        mae_vals = np.array([r.mae for r in cv_results])
        rmse_vals = np.array([r.rmse for r in cv_results])
        
        return {
            "model": "LightGBM",
            "experiment": "phase4_lgbm_experiment",
            "cv_metrics": {
                "r2_mean": float(r2_vals.mean()),
                "r2_std": float(r2_vals.std()),
                "mae_mean": float(mae_vals.mean()),
                "mae_std": float(mae_vals.std()),
                "rmse_mean": float(rmse_vals.mean()),
                "rmse_std": float(rmse_vals.std()),
                "n_folds": len(cv_results),
                "fold_metrics": [
                    {
                        "fold": r.fold_id,
                        "r2": r.r2,
                        "mae": r.mae,
                        "rmse": r.rmse,
                        "train_rows": r.train_size,
                        "val_rows": r.val_size,
                    }
                    for r in cv_results
                ],
            },
        }
    
    @staticmethod
    def save_cv_metrics(summary: Dict[str, Any], path: Path) -> None:
        """Save CV metrics summary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved metrics to: {path}")
    
    def print_summary(self, cv_results: List[CVFoldResult]) -> None:
        """Print human-readable CV summary."""
        summary = self.summarize_cv(cv_results)
        cv = summary["cv_metrics"]
        
        print("\n" + "=" * 70)
        print("PHASE-4 LightGBM EXPERIMENT RESULTS")
        print("=" * 70)
        
        print("\n📊 PER-FOLD METRICS:")
        print(f"{'Fold':<6} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'Train':>10} {'Val':>8}")
        print("-" * 60)
        
        for fold in cv["fold_metrics"]:
            print(
                f"{fold['fold']:<6} "
                f"{fold['r2']:>10.4f} "
                f"{fold['mae']:>12.2f} "
                f"{fold['rmse']:>12.2f} "
                f"{fold['train_rows']:>10,} "
                f"{fold['val_rows']:>8,}"
            )
        
        print("-" * 60)
        
        print("\n📈 AGGREGATE METRICS (mean ± std):")
        print("-" * 50)
        print(f"R²:   {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
        print(f"MAE:  {cv['mae_mean']:.2f} ± {cv['mae_std']:.2f}")
        print(f"RMSE: {cv['rmse_mean']:.2f} ± {cv['rmse_std']:.2f}")
        print("-" * 50)
        
        print("\n💡 INTERPRETATION:")
        if cv["r2_mean"] >= 0.90:
            print("  ✅ R² ≥ 90%: Excellent predictive performance!")
        elif cv["r2_mean"] >= 0.80:
            print("  ✅ R² 80-90%: Strong performance.")
        else:
            print("  ⚠️  R² < 80%: Consider more features or tuning.")
        
        print("=" * 70 + "\n")
