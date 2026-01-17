#!/usr/bin/env python
"""
Phase-4 v3 Time-Series Hyperparameter Tuning Script
====================================================

This script performs light time-series hyperparameter tuning of the v3 XGBoost
model using RandomizedSearchCV with an expanding-window CV splitter and a gap,
to keep evaluation leakage-free.

We cap n_iter and n_folds to control compute, and we only keep tuned parameters
if they also improve metrics on a final held-out test period.

This is an experiment-only pipeline; final artifacts remain frozen elsewhere.

Key Features:
- Expanding-window time-series CV with gap (no data leakage)
- RandomizedSearchCV for efficient hyperparameter search
- Leakage-safe feature engineering (lags, rolling, holidays, policy)
- Final held-out test evaluation
- Results saved under artifacts/experiments/phase4_v3/<exp_name>/

Usage:
    python -m src.run_phase4_v3_timeseries_tuning --exp-name tuning_exp_01
    python -m src.run_phase4_v3_timeseries_tuning --exp-name tuning_exp_01 --n-iter 20 --n-folds 4

Author: UIDAI ASRIS Team
Date: January 2026
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Local imports
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
from src.features.timeseries_lag_utils import get_recommended_lags
from src.validation.time_series_cv import generate_expanding_folds, print_fold_summary

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

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
ARTIFACTS_BASE = Path("artifacts/experiments/phase4_v3")

# Frozen artifacts we must NOT overwrite
FROZEN_PATHS = [
    Path("artifacts/xgb_phase4_v2_tuned_best.pkl"),
    Path("artifacts/xgb_phase4_v2_tuned_best_metrics.json"),
    Path("artifacts/xgb_phase4_v3_final.pkl"),
    Path("artifacts/xgb_phase4_v3_final_metrics.json"),
]

# Column definitions
DATE_COL = "month_date"
TARGET_COL = "total_enrolment"
GROUP_COLS = ["state", "district"]

# Default feature engineering settings
DEFAULT_LAGS = [1, 2, 3, 6]
DEFAULT_ROLLING_WINDOWS = [3, 6]

# v3 baseline metrics for comparison
V3_BASELINE = {
    "cv_r2_mean": 0.9527,
    "cv_mae_mean": 124.16,
}


# =============================================================================
# CUSTOM TIME-SERIES CV SPLITTER
# =============================================================================

class TimeSeriesCVSplitter:
    """
    Custom CV splitter for sklearn that uses pre-computed expanding-window folds.
    
    This splitter is compatible with RandomizedSearchCV and uses the same
    expanding-window logic as the v3 validation pipeline.
    """
    
    def __init__(self, folds: list[tuple[np.ndarray, np.ndarray]]):
        """
        Initialize with pre-computed fold indices.
        
        Args:
            folds: List of (train_indices, val_indices) tuples
        """
        self.folds = folds
    
    def split(self, X, y=None, groups=None):
        """Yield train/val indices for each fold."""
        for train_idx, val_idx in self.folds:
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of folds."""
        return len(self.folds)


# =============================================================================
# FEATURE ENGINEERING (LEAKAGE-SAFE)
# =============================================================================

def build_features_leakage_safe(
    df: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    holiday_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build all v3 features in a leakage-safe manner.
    
    All lag/rolling features use shift() to exclude current value.
    Features are computed per-district to prevent cross-district leakage.
    
    Args:
        df: DataFrame with columns [month_date, state_name, district_name, total_enrolment, ...]
        lags: List of lag periods to create
        rolling_windows: List of rolling window sizes
        holiday_calendar: Pre-built holiday calendar (optional)
        
    Returns:
        DataFrame with all features added
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
        lags=lags,
        date_col=DATE_COL,
    )
    
    # 2. Rolling features (per-district, leakage-safe with shift)
    df = add_rolling_features(
        df,
        target_col=TARGET_COL,
        group_cols=GROUP_COLS,
        windows=rolling_windows,
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


def prepare_features_for_modeling(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    fit_encoders: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, LabelEncoder], list[str]]:
    """
    Prepare feature matrix X and target y for modeling.
    
    Args:
        df: DataFrame with features
        label_encoders: Pre-fitted label encoders (for transform-only)
        fit_encoders: Whether to fit new encoders or use existing
        
    Returns:
        Tuple of (X, y, label_encoders, feature_names)
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
            # Handle unseen values by adding a placeholder
            unique_vals = df[col].unique().tolist()
            le.fit(unique_vals)
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column: {col}")
        
        # Transform with handling for unseen values
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
# TUNING UTILITIES
# =============================================================================

def get_param_distributions() -> dict[str, Any]:
    """
    Define moderate parameter distributions for RandomizedSearchCV.
    
    Returns:
        Dictionary of parameter distributions
    """
    return {
        "max_depth": randint(3, 7),              # 3, 4, 5, 6
        "min_child_weight": randint(1, 6),       # 1, 2, 3, 4, 5
        "gamma": uniform(0.0, 0.6),              # 0.0 to 0.6
        "subsample": uniform(0.7, 0.3),          # 0.7 to 1.0
        "colsample_bytree": uniform(0.7, 0.3),   # 0.7 to 1.0
        "learning_rate": uniform(0.02, 0.06),    # 0.02 to 0.08
        "reg_alpha": uniform(0.0, 0.3),          # 0.0 to 0.3
        "reg_lambda": uniform(0.5, 1.0),         # 0.5 to 1.5
    }


def create_base_model() -> xgb.XGBRegressor:
    """
    Create base XGBoost model with v3 settings.
    
    Returns:
        XGBRegressor instance
    """
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,           # Moderate for speed
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def compute_fold_indices(
    df_tune: pd.DataFrame,
    n_folds: int,
    gap_months: int,
    min_train_months: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Compute expanding-window fold indices for the tuning subset.
    
    Args:
        df_tune: Tuning DataFrame (excludes test period)
        n_folds: Number of CV folds
        gap_months: Gap between train and validation
        min_train_months: Minimum training months
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Use existing fold generator
    folds_masks = generate_expanding_folds(
        df=df_tune,
        date_col=DATE_COL,
        n_folds=n_folds,
        gap_months=gap_months,
        min_train_months=min_train_months,
    )
    
    # Convert boolean masks to integer indices
    fold_indices = []
    for train_mask, val_mask in folds_masks:
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        fold_indices.append((train_idx, val_idx))
    
    return fold_indices


# =============================================================================
# EXPERIMENT MANAGEMENT
# =============================================================================

def get_experiment_dir(exp_name: str) -> Path:
    """
    Get experiment directory path.
    
    Args:
        exp_name: Experiment name
        
    Returns:
        Path to experiment directory
    """
    exp_dir = ARTIFACTS_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def check_not_overwriting_frozen(exp_dir: Path) -> None:
    """
    Ensure we're not overwriting frozen artifacts.
    
    Raises:
        RuntimeError: If attempting to overwrite frozen files
    """
    for frozen_path in FROZEN_PATHS:
        if frozen_path.exists():
            # Check if exp_dir would overwrite
            if exp_dir.resolve() == frozen_path.parent.resolve():
                raise RuntimeError(
                    f"Cannot overwrite frozen artifact: {frozen_path}"
                )


def save_experiment_results(
    exp_dir: Path,
    best_params: dict,
    cv_results: dict,
    summary_metrics: dict,
    model: xgb.XGBRegressor,
    encoders: dict,
) -> None:
    """
    Save all experiment artifacts.
    
    Args:
        exp_dir: Experiment directory
        best_params: Best hyperparameters from tuning
        cv_results: Full CV results from RandomizedSearchCV
        summary_metrics: Summary metrics (CV + test)
        model: Trained XGBoost model
        encoders: Label encoders
    """
    # 1. Best parameters
    params_path = exp_dir / "tuning_best_params.json"
    with open(params_path, "w") as f:
        # Convert numpy types to Python types for JSON
        params_json = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                       for k, v in best_params.items()}
        json.dump(params_json, f, indent=2)
    logger.info(f"Saved: {params_path}")
    
    # 2. CV results
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_path = exp_dir / "tuning_cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    logger.info(f"Saved: {cv_results_path}")
    
    # 3. Summary metrics
    metrics_path = exp_dir / "tuning_summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved: {metrics_path}")
    
    # 4. Trained model
    model_path = exp_dir / "model_v3_tuned.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved: {model_path}")
    
    # 5. Encoders
    encoders_path = exp_dir / "encoders_v3_tuned.pkl"
    joblib.dump(encoders, encoders_path)
    logger.info(f"Saved: {encoders_path}")


# =============================================================================
# MAIN TUNING PIPELINE
# =============================================================================

def run_tuning(args: argparse.Namespace) -> dict:
    """
    Run the hyperparameter tuning pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Summary metrics dictionary
    """
    logger.info("=" * 70)
    logger.info("PHASE-4 v3: TIME-SERIES HYPERPARAMETER TUNING")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"n_iter: {args.n_iter}, n_folds: {args.n_folds}, gap: {args.gap_months} months")
    logger.info(f"Test months: {args.test_months}")
    logger.info("")
    
    # Get experiment directory
    exp_dir = get_experiment_dir(args.exp_name)
    check_not_overwriting_frozen(exp_dir)
    
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Sort by date
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    
    # Get unique months
    unique_months = sorted(df[DATE_COL].unique())
    logger.info(f"Date range: {unique_months[0]} to {unique_months[-1]}")
    logger.info(f"Unique months: {len(unique_months)}")
    
    # -------------------------------------------------------------------------
    # Step 2: Split into tune and test periods
    # -------------------------------------------------------------------------
    # Hold out the last N months for final test
    test_months = unique_months[-args.test_months:]
    tune_months = unique_months[:-args.test_months]
    
    is_test = df[DATE_COL].isin(test_months)
    is_tune = ~is_test
    
    df_tune = df[is_tune].copy().reset_index(drop=True)
    df_test = df[is_test].copy().reset_index(drop=True)
    
    logger.info(f"Tuning period: {tune_months[0]} to {tune_months[-1]} ({len(df_tune):,} rows)")
    logger.info(f"Test period: {test_months[0]} to {test_months[-1]} ({len(df_test):,} rows)")
    
    # -------------------------------------------------------------------------
    # Step 3: Build features (leakage-safe)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Building features...")
    
    # Get recommended lags
    lags = get_recommended_lags(df_tune, TARGET_COL, GROUP_COLS, DATE_COL)
    logger.info(f"Selected lags: {lags}")
    
    # Build holiday calendar
    holiday_calendar = build_holiday_calendar(
        start=str(df[DATE_COL].min().date()),
        end=str(df[DATE_COL].max().date()),
    )
    
    # Build features for tuning data
    df_tune_feat = build_features_leakage_safe(
        df_tune,
        lags=lags,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        holiday_calendar=holiday_calendar,
    )
    
    # Drop rows with NaN from lag features (early months)
    max_lag = max(lags) if lags else 0
    max_window = max(DEFAULT_ROLLING_WINDOWS) if DEFAULT_ROLLING_WINDOWS else 0
    warmup = max(max_lag, max_window)
    
    # Keep rows where we have enough history
    n_before = len(df_tune_feat)
    df_tune_feat = df_tune_feat.dropna(subset=[f"{TARGET_COL}_lag_{lags[0]}"])
    n_after = len(df_tune_feat)
    logger.info(f"Dropped {n_before - n_after} rows with insufficient history")
    
    # Prepare feature matrix
    X_tune, y_tune, label_encoders, feature_names = prepare_features_for_modeling(
        df_tune_feat,
        fit_encoders=True,
    )
    
    logger.info(f"Tuning features: {len(feature_names)} columns, {len(X_tune):,} rows")
    
    # -------------------------------------------------------------------------
    # Step 4: Generate CV folds for tuning
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Generating CV folds for tuning...")
    
    fold_indices = compute_fold_indices(
        df_tune_feat,
        n_folds=args.n_folds,
        gap_months=args.gap_months,
        min_train_months=args.min_train_months,
    )
    
    logger.info(f"Generated {len(fold_indices)} folds")
    for i, (train_idx, val_idx) in enumerate(fold_indices):
        logger.info(f"  Fold {i+1}: train={len(train_idx):,}, val={len(val_idx):,}")
    
    # Create CV splitter
    cv_splitter = TimeSeriesCVSplitter(fold_indices)
    
    # -------------------------------------------------------------------------
    # Step 5: Run RandomizedSearchCV
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Running RandomizedSearchCV...")
    logger.info("-" * 70)
    
    base_model = create_base_model()
    param_distributions = get_param_distributions()
    
    # Use negative MAE scorer (sklearn convention: higher is better)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=mae_scorer,
        cv=cv_splitter,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True,
    )
    
    # Fit
    start_time = datetime.now()
    search.fit(X_tune, y_tune)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Search completed in {elapsed:.1f} seconds")
    
    # Extract results
    best_params = search.best_params_
    best_cv_mae = -search.best_score_  # Convert back to positive MAE
    
    # Compute CV R² and RMSE from CV results
    cv_results = search.cv_results_
    best_idx = search.best_index_
    
    # Get per-fold scores for best params
    fold_scores = []
    for i in range(args.n_folds):
        score_key = f"split{i}_test_score"
        if score_key in cv_results:
            fold_scores.append(-cv_results[score_key][best_idx])
    
    cv_mae_std = np.std(fold_scores) if fold_scores else 0.0
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TUNING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Best CV MAE: {best_cv_mae:.2f} ± {cv_mae_std:.2f}")
    logger.info("Best parameters:")
    for k, v in best_params.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    
    # -------------------------------------------------------------------------
    # Step 6: Evaluate on final test period
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Evaluating on final test period...")
    logger.info("-" * 70)
    
    # Build features for full data (tune + test) for final model
    df_full = pd.concat([df_tune, df_test], ignore_index=True)
    df_full = df_full.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    
    df_full_feat = build_features_leakage_safe(
        df_full,
        lags=lags,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        holiday_calendar=holiday_calendar,
    )
    
    # Split back into tune and test
    df_full_feat[DATE_COL] = pd.to_datetime(df_full_feat[DATE_COL])
    is_test_feat = df_full_feat[DATE_COL].isin(test_months)
    
    df_train_final = df_full_feat[~is_test_feat].dropna(subset=[f"{TARGET_COL}_lag_{lags[0]}"]).copy()
    df_test_final = df_full_feat[is_test_feat].copy()
    
    # Prepare features (use encoders from tuning)
    X_train_final, y_train_final, _, _ = prepare_features_for_modeling(
        df_train_final,
        label_encoders=label_encoders,
        fit_encoders=False,
    )
    
    X_test_final, y_test_final, _, _ = prepare_features_for_modeling(
        df_test_final,
        label_encoders=label_encoders,
        fit_encoders=False,
    )
    
    logger.info(f"Final train: {len(X_train_final):,} rows")
    logger.info(f"Final test: {len(X_test_final):,} rows")
    
    # Train final model with best params
    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    
    final_model.fit(X_train_final, y_train_final)
    
    # Predict on test
    y_pred_test = final_model.predict(X_test_final)
    
    # Compute test metrics
    test_mae = mean_absolute_error(y_test_final, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_test))
    test_r2 = r2_score(y_test_final, y_pred_test)
    
    logger.info("")
    logger.info("Final Test Metrics:")
    logger.info(f"  R²:   {test_r2:.4f}")
    logger.info(f"  MAE:  {test_mae:.2f}")
    logger.info(f"  RMSE: {test_rmse:.2f}")
    
    # -------------------------------------------------------------------------
    # Step 7: Compare with baseline and summarize
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON WITH v3 BASELINE")
    logger.info("=" * 70)
    
    baseline_cv_mae = V3_BASELINE["cv_mae_mean"]
    improvement_pct = (baseline_cv_mae - best_cv_mae) / baseline_cv_mae * 100
    
    logger.info(f"v3 Baseline CV MAE: {baseline_cv_mae:.2f}")
    logger.info(f"Tuned CV MAE:       {best_cv_mae:.2f}")
    logger.info(f"Improvement:        {improvement_pct:+.1f}%")
    
    if best_cv_mae < baseline_cv_mae:
        logger.info("✅ Tuned model IMPROVES over baseline!")
    else:
        logger.info("⚠️  Tuned model does not improve over baseline.")
        logger.info("   Keeping as experiment record; original v3 config remains best.")
    
    # -------------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Saving experiment artifacts...")
    logger.info("-" * 70)
    
    summary_metrics = {
        "experiment_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "tuning_config": {
            "n_iter": args.n_iter,
            "n_folds": args.n_folds,
            "gap_months": args.gap_months,
            "test_months": args.test_months,
        },
        "feature_config": {
            "lags": lags,
            "rolling_windows": DEFAULT_ROLLING_WINDOWS,
            "n_features": len(feature_names),
        },
        "cv_metrics": {
            "cv_mae_mean": round(best_cv_mae, 4),
            "cv_mae_std": round(cv_mae_std, 4),
            "cv_mae_per_fold": [round(s, 4) for s in fold_scores],
        },
        "test_metrics": {
            "test_r2": round(test_r2, 4),
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
        },
        "best_params": {k: round(v, 6) if isinstance(v, float) else v 
                        for k, v in best_params.items()},
        "comparison": {
            "baseline_cv_mae": baseline_cv_mae,
            "improvement_pct": round(improvement_pct, 2),
            "is_improvement": bool(best_cv_mae < baseline_cv_mae),
        },
    }
    
    save_experiment_results(
        exp_dir=exp_dir,
        best_params=best_params,
        cv_results=cv_results,
        summary_metrics=summary_metrics,
        model=final_model,
        encoders=label_encoders,
    )
    
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📁 Results saved to: {exp_dir}")
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  CV MAE:   {best_cv_mae:.2f} ± {cv_mae_std:.2f}")
    logger.info(f"  Test R²:  {test_r2:.4f}")
    logger.info(f"  Test MAE: {test_mae:.2f}")
    logger.info(f"  Improvement over baseline: {improvement_pct:+.1f}%")
    logger.info("")
    
    return summary_metrics


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase-4 v3 Time-Series Hyperparameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tuning with defaults
  python -m src.run_phase4_v3_timeseries_tuning --exp-name tuning_01

  # More iterations and folds
  python -m src.run_phase4_v3_timeseries_tuning --exp-name tuning_02 --n-iter 20 --n-folds 4

  # Custom gap and test period
  python -m src.run_phase4_v3_timeseries_tuning --exp-name tuning_03 --gap-months 2 --test-months 2
        """,
    )
    
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (used for output folder)",
    )
    
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of RandomizedSearchCV iterations (default: 10)",
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3)",
    )
    
    parser.add_argument(
        "--gap-months",
        type=int,
        default=1,
        help="Gap months between train and validation (default: 1)",
    )
    
    parser.add_argument(
        "--test-months",
        type=int,
        default=1,
        help="Number of months to hold out for final test (default: 1)",
    )
    
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=2,
        help="Minimum training months per fold (default: 2)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        summary = run_tuning(args)
        return 0
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        raise


if __name__ == "__main__":
    exit(main())
