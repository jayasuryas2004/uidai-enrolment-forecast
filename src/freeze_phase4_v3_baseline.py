#!/usr/bin/env python
"""
freeze_phase4_v3_baseline.py
============================

One-time script to train and freeze the Phase-4 v3 baseline model as the
official, leakage-safe, production-style model for UIDAI forecasting.

This script:
    1. Loads the district-month dataset
    2. Builds leakage-safe features (lags, rolling, holidays, policy)
    3. Trains an XGBoost model with the v3 baseline hyperparameters
    4. Evaluates with expanding-window time-series CV (4 folds, 1-month gap)
    5. Saves the model, encoders, metrics, and params to the frozen v3 paths

After running this script, the v3 baseline artifacts are FROZEN and protected
from accidental overwriting by the model registry.

╔════════════════════════════════════════════════════════════════════════════╗
║  WARNING: This script should be run ONCE to establish the v3 baseline.    ║
║  After that, all experiments must use get_experiment_paths() instead.     ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage:
    python -m src.freeze_phase4_v3_baseline

Author: UIDAI Forecast Team
Date: January 2026
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Local imports
from src.phase4_model_registry import (
    PHASE4_V3_FINAL,
    check_not_overwriting_v3_final,
)
from src.features.timeseries_features import (
    add_lag_features,
    add_rolling_features,
    add_diff_features,
    add_pct_change_features,
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
from src.validation.time_series_cv import generate_expanding_folds, print_fold_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_PATH = Path("data/processed/district_month_modeling.csv")

# Column definitions
DATE_COL = "month_date"
TARGET_COL = "total_enrolment"
GROUP_COLS = ["state", "district"]

# v3 baseline feature settings
V3_LAGS = [1, 2, 3, 6]
V3_ROLLING_WINDOWS = [3, 6]

# v3 baseline hyperparameters (proven stable in CV)
V3_BASELINE_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# CV settings
N_FOLDS = 4
GAP_MONTHS = 1
MIN_TRAIN_MONTHS = 3


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_v3_features(
    df: pd.DataFrame,
    holiday_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build all v3 baseline features in a leakage-safe manner.
    
    All lag/rolling features use shift() to exclude current value.
    Features are computed per-district to prevent cross-district leakage.
    """
    df = df.copy()
    
    # Ensure date column is datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    # Sort by group and date for proper lag computation
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    
    # 1. Lag features (per-district)
    df = add_lag_features(
        df,
        target_col=TARGET_COL,
        group_cols=GROUP_COLS,
        lags=V3_LAGS,
        date_col=DATE_COL,
    )
    
    # 2. Rolling features (per-district, leakage-safe with shift)
    df = add_rolling_features(
        df,
        target_col=TARGET_COL,
        group_cols=GROUP_COLS,
        windows=V3_ROLLING_WINDOWS,
        date_col=DATE_COL,
    )
    
    # 3. Diff features
    df = add_diff_features(
        df,
        target_col=TARGET_COL,
        group_cols=GROUP_COLS,
        periods=[1],
        date_col=DATE_COL,
    )
    
    # 4. Pct change features
    df = add_pct_change_features(
        df,
        target_col=TARGET_COL,
        group_cols=GROUP_COLS,
        periods=[1],
        date_col=DATE_COL,
    )
    
    # 5. Holiday/calendar features
    if holiday_calendar is not None:
        df = add_holiday_features(df, date_col=DATE_COL, holiday_df=holiday_calendar)
    
    # 6. Month sin/cos encoding
    df = add_month_sin_cos(df, date_col=DATE_COL)
    
    # 7. Policy phase features
    df = add_policy_phase_features(df, date_col=DATE_COL)
    
    # 8. Time trend features
    df = add_time_trend_features(df, date_col=DATE_COL)
    
    # 9. Calendar features
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter
    
    return df


def prepare_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    fit_encoders: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, LabelEncoder], list[str]]:
    """
    Prepare feature matrix X and target y for modeling.
    """
    df = df.copy()
    
    # Target
    y = df[TARGET_COL].values
    
    # Drop non-feature columns
    drop_cols = [TARGET_COL, DATE_COL]
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in drop_cols]
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in cat_cols:
        if fit_encoders:
            le = LabelEncoder()
            unique_vals = df[col].unique().tolist()
            le.fit(unique_vals)
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column: {col}")
        
        df[col] = df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    
    # Drop non-feature columns
    X = df.drop(columns=drop_cols, errors="ignore")
    
    # Handle NaN values (from lag/rolling features)
    X = X.fillna(-999)
    
    feature_names = X.columns.tolist()
    
    return X, y, label_encoders, feature_names


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train and freeze the v3 baseline model."""
    
    logger.info("=" * 70)
    logger.info("FREEZING PHASE-4 v3 BASELINE MODEL")
    logger.info("=" * 70)
    logger.info("")
    
    # -------------------------------------------------------------------------
    # Step 0: Check if already frozen
    # -------------------------------------------------------------------------
    if PHASE4_V3_FINAL.final_model.exists():
        logger.error("v3 baseline artifacts already exist!")
        logger.error("Cannot overwrite frozen artifacts.")
        logger.error("If you really need to re-freeze, manually delete the existing files first.")
        return 1
    
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Sort by date
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    
    unique_months = sorted(df[DATE_COL].unique())
    logger.info(f"Date range: {unique_months[0]} to {unique_months[-1]}")
    logger.info(f"Unique months: {len(unique_months)}")
    
    # -------------------------------------------------------------------------
    # Step 2: Build features
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Building v3 baseline features...")
    logger.info(f"  Lags: {V3_LAGS}")
    logger.info(f"  Rolling windows: {V3_ROLLING_WINDOWS}")
    
    # Build holiday calendar
    holiday_calendar = build_holiday_calendar(
        start=str(df[DATE_COL].min().date()),
        end=str(df[DATE_COL].max().date()),
    )
    
    # Build features
    df_feat = build_v3_features(df, holiday_calendar=holiday_calendar)
    
    # Drop rows with NaN from lag features
    n_before = len(df_feat)
    df_feat = df_feat.dropna(subset=[f"{TARGET_COL}_lag_{V3_LAGS[0]}"])
    n_after = len(df_feat)
    logger.info(f"Dropped {n_before - n_after} rows with insufficient history")
    
    # -------------------------------------------------------------------------
    # Step 3: Time-series CV evaluation
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Running expanding-window time-series CV...")
    logger.info("-" * 70)
    
    folds = generate_expanding_folds(
        df=df_feat,
        date_col=DATE_COL,
        n_folds=N_FOLDS,
        gap_months=GAP_MONTHS,
        min_train_months=MIN_TRAIN_MONTHS,
    )
    
    print_fold_summary(df_feat, DATE_COL, folds)
    
    fold_metrics = []
    
    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        df_train = df_feat[train_mask].copy()
        df_val = df_feat[val_mask].copy()
        
        # Prepare features (fit encoders on train only)
        X_train, y_train, encoders, feature_names = prepare_features(
            df_train, fit_encoders=True
        )
        X_val, y_val, _, _ = prepare_features(
            df_val, label_encoders=encoders, fit_encoders=False
        )
        
        # Train model
        model = xgb.XGBRegressor(**V3_BASELINE_PARAMS)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        })
        
        logger.info(f"Fold {fold_idx + 1}: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    # Aggregate metrics
    r2_vals = [m["r2"] for m in fold_metrics]
    mae_vals = [m["mae"] for m in fold_metrics]
    rmse_vals = [m["rmse"] for m in fold_metrics]
    
    cv_metrics = {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_std": float(np.std(rmse_vals)),
        "fold_metrics": fold_metrics,
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("CV RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"R²:   {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
    logger.info(f"MAE:  {cv_metrics['mae_mean']:.2f} ± {cv_metrics['mae_std']:.2f}")
    logger.info(f"RMSE: {cv_metrics['rmse_mean']:.2f} ± {cv_metrics['rmse_std']:.2f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Train final model on all data
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Training final model on all data...")
    logger.info("-" * 70)
    
    X_all, y_all, final_encoders, feature_names = prepare_features(
        df_feat, fit_encoders=True
    )
    
    final_model = xgb.XGBRegressor(**V3_BASELINE_PARAMS)
    final_model.fit(X_all, y_all)
    
    logger.info(f"Final model trained on {len(X_all):,} rows, {len(feature_names)} features")
    
    # -------------------------------------------------------------------------
    # Step 5: Save artifacts
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Saving frozen v3 baseline artifacts...")
    logger.info("-" * 70)
    
    # Ensure artifacts directory exists
    PHASE4_V3_FINAL.final_model.parent.mkdir(parents=True, exist_ok=True)
    
    # Check overwrite protection (should pass since we checked at start)
    for path in PHASE4_V3_FINAL.all_final_paths():
        check_not_overwriting_v3_final(path)
    
    # Save model
    joblib.dump(final_model, PHASE4_V3_FINAL.final_model)
    logger.info(f"Saved: {PHASE4_V3_FINAL.final_model}")
    
    # Save encoders
    joblib.dump(final_encoders, PHASE4_V3_FINAL.final_encoders)
    logger.info(f"Saved: {PHASE4_V3_FINAL.final_encoders}")
    
    # Save metrics
    full_metrics = {
        "model_version": "phase4_v3_baseline",
        "frozen_at": datetime.now().isoformat(),
        "data_path": str(DATA_PATH),
        "data_rows": len(df),
        "training_rows": len(X_all),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "cv_config": {
            "n_folds": N_FOLDS,
            "gap_months": GAP_MONTHS,
            "min_train_months": MIN_TRAIN_MONTHS,
        },
        "feature_config": {
            "lags": V3_LAGS,
            "rolling_windows": V3_ROLLING_WINDOWS,
        },
        "cv_metrics": cv_metrics,
    }
    
    with open(PHASE4_V3_FINAL.final_metrics, "w") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved: {PHASE4_V3_FINAL.final_metrics}")
    
    # Save params
    with open(PHASE4_V3_FINAL.final_params, "w") as f:
        json.dump(V3_BASELINE_PARAMS, f, indent=2, default=str)
    logger.info(f"Saved: {PHASE4_V3_FINAL.final_params}")
    
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("v3 BASELINE FROZEN SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Frozen artifacts:")
    logger.info(f"  Model:    {PHASE4_V3_FINAL.final_model}")
    logger.info(f"  Encoders: {PHASE4_V3_FINAL.final_encoders}")
    logger.info(f"  Metrics:  {PHASE4_V3_FINAL.final_metrics}")
    logger.info(f"  Params:   {PHASE4_V3_FINAL.final_params}")
    logger.info("")
    logger.info("CV Performance:")
    logger.info(f"  R²:  {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
    logger.info(f"  MAE: {cv_metrics['mae_mean']:.2f} ± {cv_metrics['mae_std']:.2f}")
    logger.info("")
    logger.info("⚠️  These artifacts are now PROTECTED from overwriting.")
    logger.info("    All future experiments must use get_experiment_paths().")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
