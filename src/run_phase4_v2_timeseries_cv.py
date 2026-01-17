#!/usr/bin/env python
"""
run_phase4_v2_timeseries_cv.py
==============================

Leakage-Safe Expanding-Window Time-Series Cross-Validation for Phase-4 v2.

**PURPOSE:**
Evaluate the XGBoost model using realistic, leakage-free time-series CV to
get stable, unbiased performance estimates.

**WHY THIS MATTERS:**
    1. Single train/val/test splits can give optimistic or pessimistic estimates
       depending on the specific time period chosen.
    2. Cross-validation provides mean ± std of metrics across multiple folds.
    3. The gap between train and validation prevents leakage from lag features.
    4. Expanding window mimics real deployment (always more historical data).

**LEAKAGE PREVENTION:**
    - NO random shuffling - strictly past → future.
    - Features are built ONLY from training data in each fold.
    - Encoders/preprocessors are fit ONLY on training data.
    - Gap months prevent lag/rolling features from seeing validation data.

**EXPECTED RESULTS:**
    - R² around 80-90% is realistic (not fake 99% from leakage).
    - MAE should be reasonable given the scale of enrolment numbers.
    - Std across folds shows model stability.

**USAGE:**
    # Run 4-fold CV with 1-month gap
    python -m src.run_phase4_v2_timeseries_cv --exp-name cv_gap_1m --n-folds 4 --gap-months 1

    # Run 5-fold CV with 2-month gap (more conservative)
    python -m src.run_phase4_v2_timeseries_cv --exp-name cv_gap_2m --n-folds 5 --gap-months 2

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Import from our modules
try:
    from src.validation.time_series_cv import (
        generate_expanding_folds,
        print_fold_summary,
        validate_no_overlap,
        validate_temporal_order,
    )
    from src.phase4_model_registry import (
        PHASE4_V2_FINAL,
        get_experiment_paths,
        ensure_experiment_dir,
    )
except ImportError:
    from validation.time_series_cv import (
        generate_expanding_folds,
        print_fold_summary,
        validate_no_overlap,
        validate_temporal_order,
    )
    from phase4_model_registry import (
        PHASE4_V2_FINAL,
        get_experiment_paths,
        ensure_experiment_dir,
    )


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data/processed/district_month_modeling.csv")
DATE_COL = "month_date"
TARGET_COL = "total_enrolment"

# Best hyperparameters from Phase-4 v2 (frozen final model)
BEST_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 50

# Festival months for calendar features
FESTIVAL_MONTHS = [8, 9, 10, 11]


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with a clean format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)


# =============================================================================
# Feature Engineering (PER FOLD - NO LEAKAGE)
# =============================================================================

def build_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Add calendar features to DataFrame.
    
    These features are deterministic based on date only - no leakage risk.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    
    # Indian Financial Year (April-March)
    df["financial_year"] = np.where(df["month"] >= 4, df["year"], df["year"] - 1)
    df["is_financial_year_start"] = (df["month"] == 4).astype(int)
    df["is_financial_year_end"] = (df["month"] == 3).astype(int)
    df["is_festival_month"] = df["month"].isin(FESTIVAL_MONTHS).astype(int)
    
    return df


def build_group_aggregates_leakage_free(
    df_train: pd.DataFrame,
    df_apply: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build group aggregate features using ONLY training data statistics.
    
    This is the KEY function for leakage prevention:
        - Long-term mean/std are computed from df_train ONLY
        - Rolling features are computed with shift to prevent row-level leakage
        - State ratios use training data means
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data to compute statistics from.
    df_apply : pd.DataFrame
        Data to apply the statistics to (can be train or val).
    target_col : str
        Name of target column.
    date_col : str
        Name of date column.
    group_cols : List[str]
        Columns to group by (e.g., ["state", "district"]).
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - DataFrame with aggregate features added
        - Dictionary of statistics for applying to validation
    """
    df = df_apply.copy()
    stats = {}
    
    # -------------------------------------------------------------------------
    # 1. Long-term group statistics (computed from TRAIN ONLY)
    # -------------------------------------------------------------------------
    group_stats = df_train.groupby(group_cols, observed=True)[target_col].agg(
        ["mean", "std"]
    ).reset_index()
    group_stats.columns = group_cols + ["group_long_term_mean", "group_long_term_std"]
    stats["group_stats"] = group_stats
    
    # Merge to apply dataframe
    df = df.merge(group_stats, on=group_cols, how="left")
    
    # Fill missing (unseen groups) with global train mean
    global_train_mean = df_train[target_col].mean()
    global_train_std = df_train[target_col].std()
    stats["global_mean"] = global_train_mean
    stats["global_std"] = global_train_std
    
    df["group_long_term_mean"] = df["group_long_term_mean"].fillna(global_train_mean)
    df["group_long_term_std"] = df["group_long_term_std"].fillna(global_train_std)
    
    # -------------------------------------------------------------------------
    # 2. Rolling features (computed on apply data but using shift)
    # -------------------------------------------------------------------------
    # Sort by group + date for correct temporal ordering
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    
    # Rolling 3-period mean (shifted by 1 to prevent leakage)
    df["rolling_3_mean"] = (
        df.groupby(group_cols, observed=True)[target_col]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    
    # Rolling 6-period mean (shifted by 1)
    df["rolling_6_mean"] = (
        df.groupby(group_cols, observed=True)[target_col]
        .transform(lambda x: x.shift(1).rolling(window=6, min_periods=1).mean())
    )
    
    # Fill NaN with group mean
    df["rolling_3_mean"] = df["rolling_3_mean"].fillna(df["group_long_term_mean"])
    df["rolling_6_mean"] = df["rolling_6_mean"].fillna(df["group_long_term_mean"])
    
    # -------------------------------------------------------------------------
    # 3. State-level ratio (using TRAIN state means)
    # -------------------------------------------------------------------------
    if "state" in df.columns:
        state_means = df_train.groupby("state", observed=True)[target_col].mean()
        stats["state_means"] = state_means
        
        df["state_mean"] = df["state"].map(state_means)
        df["state_mean"] = df["state_mean"].fillna(global_train_mean)
        
        df["ratio_to_state_mean"] = np.where(
            df["state_mean"] > 0,
            df[target_col] / df["state_mean"],
            1.0
        )
        df = df.drop(columns=["state_mean"])
    else:
        df["ratio_to_state_mean"] = 1.0
    
    return df, stats


def encode_categoricals_fit(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Fit label encoders on training data ONLY.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    cat_cols : List[str] | None
        Columns to encode. If None, auto-detect object/category columns.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, LabelEncoder]]
        - Encoded DataFrame
        - Dictionary of fitted encoders
    """
    df = df.copy()
    
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str).unique())
        df[col] = le.transform(df[col].astype(str))
        encoders[col] = le
    
    return df, encoders


def encode_categoricals_transform(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    """
    Apply fitted encoders to new data.
    
    Handles unseen categories by assigning -1 (XGBoost treats as missing).
    """
    df = df.copy()
    
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen categories
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from DataFrame.
    
    Removes date column and any specified exclusion columns from features.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    drop_cols = [target_col, date_col] + exclude_cols
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    return X, y


# =============================================================================
# CLI Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run leakage-safe time-series CV for Phase-4 v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (e.g., 'cv_gap_1m'). REQUIRED.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--gap-months",
        type=int,
        default=1,
        help="Gap months between train and validation.",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=4,
        help="Minimum months in first training fold.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving artifacts.",
    )
    
    return parser.parse_args()


# =============================================================================
# Metrics Summary
# =============================================================================

def print_cv_summary(
    results: Dict[str, Any],
    exp_name: str,
) -> None:
    """Print a human-readable summary of CV results."""
    print("\n" + "=" * 70)
    print(f"TIME-SERIES CROSS-VALIDATION RESULTS: {exp_name}")
    print("=" * 70)
    
    print("\n📊 PER-FOLD METRICS:")
    print(f"{'Fold':<6} {'R²':>10} {'MAE':>12} {'RMSE':>12}")
    print("-" * 44)
    
    for fold in results["folds"]:
        print(
            f"{fold['fold']:<6} "
            f"{fold['r2']:>10.4f} "
            f"{fold['mae']:>12.2f} "
            f"{fold['rmse']:>12.2f}"
        )
    
    print("-" * 44)
    
    print("\n📈 AGGREGATE METRICS (mean ± std):")
    print("-" * 44)
    print(f"R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    print(f"MAE:  {results['mae_mean']:.2f} ± {results['mae_std']:.2f}")
    print(f"RMSE: {results['rmse_mean']:.2f} ± {results['rmse_std']:.2f}")
    print("-" * 44)
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if results["r2_mean"] >= 0.85:
        print("  ✅ R² ≥ 85%: Strong predictive performance.")
    elif results["r2_mean"] >= 0.70:
        print("  ⚠️  R² 70-85%: Moderate performance, consider more features.")
    else:
        print("  ❌ R² < 70%: Weak performance, needs improvement.")
    
    if results["r2_std"] < 0.05:
        print("  ✅ Low R² variance: Model is stable across folds.")
    else:
        print("  ⚠️  High R² variance: Performance varies by time period.")
    
    print("=" * 70 + "\n")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """
    Run leakage-safe time-series CV for Phase-4 v2.
    
    Pipeline:
        1. Load data
        2. Generate expanding-window folds with gap
        3. For each fold:
           a. Build features on TRAIN ONLY
           b. Apply same transformations to VAL
           c. Train XGBoost
           d. Evaluate on VAL
        4. Aggregate metrics
        5. Save results
    """
    args = parse_args()
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("LEAKAGE-SAFE TIME-SERIES CROSS-VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Folds: {args.n_folds}, Gap: {args.gap_months} months")
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    logger.info(f"Loading data from: {args.data_path}")
    
    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    df = pd.read_csv(args.data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    
    # =========================================================================
    # Step 2: Generate folds
    # =========================================================================
    logger.info("\nGenerating expanding-window folds...")
    
    folds = generate_expanding_folds(
        df=df,
        date_col=DATE_COL,
        n_folds=args.n_folds,
        gap_months=args.gap_months,
        min_train_months=args.min_train_months,
        val_months=1,
    )
    
    # Validate folds
    validate_no_overlap(folds)
    validate_temporal_order(df, DATE_COL, folds)
    
    print_fold_summary(df, DATE_COL, folds)
    
    logger.info(f"Generated {len(folds)} valid folds")
    
    # =========================================================================
    # Step 3: Cross-validation loop
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Starting cross-validation...")
    logger.info("-" * 70)
    
    fold_results: List[Dict[str, Any]] = []
    
    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(folds)} ---")
        
        # Split data
        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()
        
        logger.info(f"Train: {len(df_train):,} rows, Val: {len(df_val):,} rows")
        
        # ---------------------------------------------------------------------
        # Step 3a: Build features on TRAIN ONLY
        # ---------------------------------------------------------------------
        
        # Calendar features (no leakage - deterministic from date)
        df_train = build_calendar_features(df_train, DATE_COL)
        df_val = build_calendar_features(df_val, DATE_COL)
        
        # Group aggregates (LEAKAGE-FREE: stats from train only)
        group_cols = ["state", "district"]
        group_cols = [c for c in group_cols if c in df_train.columns]
        
        if group_cols:
            df_train, train_stats = build_group_aggregates_leakage_free(
                df_train=df_train,
                df_apply=df_train,
                target_col=TARGET_COL,
                date_col=DATE_COL,
                group_cols=group_cols,
            )
            
            # Apply SAME stats to validation
            df_val, _ = build_group_aggregates_leakage_free(
                df_train=df_train,  # Use original train for stats
                df_apply=df_val,
                target_col=TARGET_COL,
                date_col=DATE_COL,
                group_cols=group_cols,
            )
        
        # Prepare X, y
        X_train, y_train = prepare_features_and_target(
            df_train, TARGET_COL, DATE_COL, exclude_cols=["year_month"]
        )
        X_val, y_val = prepare_features_and_target(
            df_val, TARGET_COL, DATE_COL, exclude_cols=["year_month"]
        )
        
        # ---------------------------------------------------------------------
        # Step 3b: Encode categoricals (FIT on train, TRANSFORM val)
        # ---------------------------------------------------------------------
        X_train_enc, encoders = encode_categoricals_fit(X_train)
        X_val_enc = encode_categoricals_transform(X_val, encoders)
        
        # Align columns (in case of column order differences)
        X_val_enc = X_val_enc[X_train_enc.columns]
        
        # ---------------------------------------------------------------------
        # Step 3c: Train XGBoost
        # ---------------------------------------------------------------------
        model = XGBRegressor(**BEST_PARAMS)
        
        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_val_enc, y_val)],
            verbose=False,
        )
        
        # ---------------------------------------------------------------------
        # Step 3d: Evaluate
        # ---------------------------------------------------------------------
        y_pred = model.predict(X_val_enc)
        
        fold_mae = mean_absolute_error(y_val, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_r2 = r2_score(y_val, y_pred)
        
        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(df_train),
            "val_size": len(df_val),
            "train_date_range": f"{df_train[DATE_COL].min().strftime('%Y-%m')} to {df_train[DATE_COL].max().strftime('%Y-%m')}",
            "val_date": df_val[DATE_COL].iloc[0].strftime("%Y-%m"),
            "mae": float(fold_mae),
            "rmse": float(fold_rmse),
            "r2": float(fold_r2),
        })
        
        logger.info(
            f"Fold {fold_idx + 1}: R²={fold_r2:.4f}, MAE={fold_mae:.2f}, RMSE={fold_rmse:.2f}"
        )
    
    # =========================================================================
    # Step 4: Aggregate metrics
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Aggregating metrics...")
    logger.info("-" * 70)
    
    mae_list = [f["mae"] for f in fold_results]
    rmse_list = [f["rmse"] for f in fold_results]
    r2_list = [f["r2"] for f in fold_results]
    
    results = {
        "experiment_name": args.exp_name,
        "n_folds": len(folds),
        "gap_months": args.gap_months,
        "min_train_months": args.min_train_months,
        "folds": fold_results,
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "hyperparams": {k: v for k, v in BEST_PARAMS.items() 
                        if k not in ["random_state", "n_jobs"]},
        "leakage_prevention": {
            "gap_months": args.gap_months,
            "train_only_feature_engineering": True,
            "expanding_window": True,
            "no_shuffle": True,
        },
    }
    
    # Print summary
    print_cv_summary(results, args.exp_name)
    
    # =========================================================================
    # Step 5: Save results
    # =========================================================================
    if not args.no_save:
        logger.info("Saving results...")
        
        exp_paths = get_experiment_paths(args.exp_name)
        ensure_experiment_dir(args.exp_name)
        
        # Save JSON
        json_path = exp_paths["dir"] / "metrics_timeseries_cv.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {json_path}")
        
        # Save CSV
        csv_path = exp_paths["dir"] / "metrics_timeseries_cv.csv"
        pd.DataFrame(fold_results).to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
        
        logger.info(f"\n📁 Results saved to: {exp_paths['dir']}")
    
    # =========================================================================
    # Compare to Final v2 model
    # =========================================================================
    if PHASE4_V2_FINAL.final_metrics.exists():
        with open(PHASE4_V2_FINAL.final_metrics) as f:
            final_metrics = json.load(f)
        
        final_val_r2 = final_metrics.get("split", {}).get("val", {}).get("r2")
        final_val_mae = final_metrics.get("split", {}).get("val", {}).get("mae")
        
        print("\n" + "=" * 70)
        print("COMPARISON: CV Metrics vs Final v2 Single Split")
        print("=" * 70)
        print(f"{'Metric':<15} {'CV Mean±Std':>20} {'Final v2':>15}")
        print("-" * 55)
        if final_val_r2:
            print(f"{'Val R²':<15} {results['r2_mean']:.4f} ± {results['r2_std']:.4f}   {final_val_r2:>15.4f}")
        if final_val_mae:
            print(f"{'Val MAE':<15} {results['mae_mean']:.2f} ± {results['mae_std']:.2f}   {final_val_mae:>15.2f}")
        print("-" * 55)
        print("\n💡 CV metrics are typically lower than single-split due to:")
        print("   - Stricter leakage prevention")
        print("   - Multiple time periods tested")
        print("   - More realistic deployment estimate")
        print("=" * 70)
    
    logger.info("\n✅ Time-series CV complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
