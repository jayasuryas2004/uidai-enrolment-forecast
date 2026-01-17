#!/usr/bin/env python
"""
run_phase4_v3_enhanced_cv.py
============================

Phase-4 v3: Enhanced Feature Engineering with Leakage-Safe Time-Series CV.

**PURPOSE:**
Evaluate XGBoost with a comprehensive, leakage-safe feature set including:
    - ACF/PACF-guided lag features
    - Rolling window statistics
    - Holiday/festival features
    - Policy/intervention features

**KEY IMPROVEMENTS OVER v2:**
    1. ACF/PACF-guided lag selection (not arbitrary lags)
    2. Richer holiday/event calendar (Indian context)
    3. Policy phase dummies for intervention effects
    4. Time trend features for secular growth patterns
    5. Stricter leakage prevention in CV

**LEAKAGE PREVENTION:**
    - NO shuffling - strictly past → future
    - Lag/rolling features use shift() to exclude current value
    - Features computed on TRAIN fold only, applied to VAL
    - Encoders fit on TRAIN only
    - Gap months between train and validation

**EXPECTED PERFORMANCE:**
    - Realistic R² > 0.9 on CV (not fake 0.999 from leakage)
    - Low MAE relative to target scale
    - Stable performance across folds (low std)

**USAGE:**
    # Run 4-fold CV with enhanced features
    python -m src.run_phase4_v3_enhanced_cv --exp-name v3_enhanced_4fold --n-folds 4

    # With more lags
    python -m src.run_phase4_v3_enhanced_cv --exp-name v3_more_lags --n-folds 4 --lags 1 2 3 6

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
    from src.features.timeseries_lag_utils import (
        get_recommended_lags,
        DEFAULT_LAGS,
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
    from features.timeseries_lag_utils import (
        get_recommended_lags,
        DEFAULT_LAGS,
    )


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data/processed/district_month_modeling.csv")
DATE_COL = "month_date"
TARGET_COL = "total_enrolment"
GROUP_COLS = ["state", "district"]

# Best hyperparameters from Phase-4 v2 (frozen)
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
# Feature Engineering (Leakage-Safe)
# =============================================================================

def add_calendar_features_safe(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Add basic calendar features (year, month, quarter, FY).
    
    These are deterministic from date only - no leakage risk.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    
    # Indian Financial Year (April-March)
    df["financial_year"] = np.where(df["month"] >= 4, df["year"], df["year"] - 1)
    
    return df


def build_features_for_fold(
    df_train: pd.DataFrame,
    df_apply: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    lags: List[int],
    rolling_windows: List[int],
    holiday_calendar: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build all features for a single CV fold.
    
    **CRITICAL LEAKAGE PREVENTION:**
    - Lag/rolling features computed on df_apply but using shift()
    - Group statistics (for imputation) computed from df_train only
    - Holiday/policy features are calendar-based (no target leakage)
    - Encoders fit on training data only
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data for this fold (for computing statistics).
    df_apply : pd.DataFrame
        Data to apply features to (can be train or val).
    group_cols : List[str]
        Group columns.
    date_col : str
        Date column.
    target_col : str
        Target column.
    lags : List[int]
        Lag periods to use.
    rolling_windows : List[int]
        Rolling window sizes.
    holiday_calendar : pd.DataFrame
        Pre-built holiday calendar.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - DataFrame with features
        - Dictionary of statistics/encoders for applying to validation
    """
    df = df_apply.copy()
    stats = {}
    
    # 1. Calendar features (safe - date only)
    df = add_calendar_features_safe(df, date_col)
    
    # 2. Holiday/event features (safe - calendar only)
    df = add_holiday_features(df, date_col, holiday_calendar)
    
    # 3. Policy/intervention features (safe - calendar only)
    df = add_policy_phase_features(df, date_col)
    df = add_time_trend_features(df, date_col)
    
    # 4. Cyclical month encoding (safe - calendar only)
    df = add_month_sin_cos(df, date_col)
    
    # 5. Lag features (LEAKAGE-SAFE: uses shift)
    df = add_lag_features(
        df, group_cols, date_col, target_col, lags, fill_na=False
    )
    
    # 6. Rolling features (LEAKAGE-SAFE: uses shift before rolling)
    df = add_rolling_features(
        df, group_cols, date_col, target_col, rolling_windows, 
        include_std=True, fill_na=False
    )
    
    # 7. Difference features (safe - uses past values)
    df = add_diff_features(df, group_cols, date_col, target_col, [1], fill_na=False)
    
    # 8. Group statistics for imputation (from TRAIN only)
    group_means = df_train.groupby(group_cols, observed=True)[target_col].mean()
    group_stds = df_train.groupby(group_cols, observed=True)[target_col].std()
    global_mean = df_train[target_col].mean()
    global_std = df_train[target_col].std()
    
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
    if group_cols:
        df["_group_key"] = df[group_cols].apply(tuple, axis=1)
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


def encode_categoricals_fit(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Fit encoders on training data."""
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
    """Apply fitted encoders to new data."""
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
    """Separate X and y."""
    if exclude_cols is None:
        exclude_cols = []
    
    drop_cols = [target_col, date_col] + exclude_cols
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    return X, y


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase-4 v3 enhanced CV with rich features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (e.g., 'v3_enhanced_4fold').",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of CV folds.",
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
        default=3,
        help="Minimum months in first training fold.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=None,
        help="Lag periods (e.g., --lags 1 2 3 6). If not provided, uses ACF/PACF analysis.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=[3, 6],
        help="Rolling window sizes.",
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
# Summary Printing
# =============================================================================

def print_cv_summary(results: Dict[str, Any], exp_name: str) -> None:
    """Print human-readable CV results."""
    print("\n" + "=" * 70)
    print(f"PHASE-4 v3 ENHANCED CV RESULTS: {exp_name}")
    print("=" * 70)
    
    print("\n📊 PER-FOLD METRICS:")
    print(f"{'Fold':<6} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'Train':>10} {'Val':>8}")
    print("-" * 60)
    
    for fold in results["folds"]:
        print(
            f"{fold['fold']:<6} "
            f"{fold['r2']:>10.4f} "
            f"{fold['mae']:>12.2f} "
            f"{fold['rmse']:>12.2f} "
            f"{fold['train_size']:>10,} "
            f"{fold['val_size']:>8,}"
        )
    
    print("-" * 60)
    
    print("\n📈 AGGREGATE METRICS (mean ± std):")
    print("-" * 50)
    print(f"R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    print(f"MAE:  {results['mae_mean']:.2f} ± {results['mae_std']:.2f}")
    print(f"RMSE: {results['rmse_mean']:.2f} ± {results['rmse_std']:.2f}")
    print("-" * 50)
    
    print("\n🔧 FEATURES USED:")
    print(f"  Lags: {results['config']['lags']}")
    print(f"  Rolling windows: {results['config']['rolling_windows']}")
    print(f"  Total features: {results['config']['n_features']}")
    
    print("\n💡 INTERPRETATION:")
    if results["r2_mean"] >= 0.90:
        print("  ✅ R² ≥ 90%: Excellent predictive performance!")
    elif results["r2_mean"] >= 0.80:
        print("  ✅ R² 80-90%: Strong performance.")
    else:
        print("  ⚠️  R² < 80%: Consider more features or tuning.")
    
    if results["r2_std"] < 0.05:
        print("  ✅ Low variance: Model is stable across time periods.")
    else:
        print("  ⚠️  High variance: Performance varies by time period.")
    
    print("=" * 70 + "\n")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """Run Phase-4 v3 enhanced CV pipeline."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("PHASE-4 v3: ENHANCED FEATURE ENGINEERING + TIME-SERIES CV")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Folds: {args.n_folds}, Gap: {args.gap_months} months")
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    logger.info(f"\nLoading data from: {args.data_path}")
    
    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    df = pd.read_csv(args.data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    
    # =========================================================================
    # Step 2: Determine lags (ACF/PACF or user-specified)
    # =========================================================================
    if args.lags:
        lags = args.lags
        logger.info(f"Using user-specified lags: {lags}")
    else:
        logger.info("Running ACF/PACF analysis to determine lags...")
        try:
            lags = get_recommended_lags(
                df, GROUP_COLS, DATE_COL, TARGET_COL,
                max_lag=6, sample_size=5, threshold=0.2, top_k=4
            )
        except Exception as e:
            logger.warning(f"ACF/PACF analysis failed: {e}. Using defaults.")
            lags = DEFAULT_LAGS
        logger.info(f"Selected lags: {lags}")
    
    rolling_windows = args.rolling_windows
    logger.info(f"Rolling windows: {rolling_windows}")
    
    # =========================================================================
    # Step 3: Build holiday calendar (once, covers all dates)
    # =========================================================================
    logger.info("Building holiday calendar...")
    start_date = (df[DATE_COL].min() - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    end_date = (df[DATE_COL].max() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    holiday_calendar = build_holiday_calendar(start_date, end_date)
    logger.info(f"Holiday calendar: {len(holiday_calendar)} months")
    
    # =========================================================================
    # Step 4: Generate CV folds
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
    
    validate_no_overlap(folds)
    validate_temporal_order(df, DATE_COL, folds)
    print_fold_summary(df, DATE_COL, folds)
    
    logger.info(f"Generated {len(folds)} valid folds")
    
    # =========================================================================
    # Step 5: Cross-validation loop
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Starting cross-validation with enhanced features...")
    logger.info("-" * 70)
    
    fold_results: List[Dict[str, Any]] = []
    n_features = 0
    
    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(folds)} ---")
        
        # Split data
        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()
        
        logger.info(f"Train: {len(df_train):,} rows, Val: {len(df_val):,} rows")
        
        # ---------------------------------------------------------------------
        # Build features (leakage-safe)
        # ---------------------------------------------------------------------
        
        # For TRAINING: build features on training data
        df_train_feat, train_stats = build_features_for_fold(
            df_train=df_train,
            df_apply=df_train,
            group_cols=GROUP_COLS,
            date_col=DATE_COL,
            target_col=TARGET_COL,
            lags=lags,
            rolling_windows=rolling_windows,
            holiday_calendar=holiday_calendar,
        )
        
        # For VALIDATION: build features but use train stats for imputation
        df_val_feat, _ = build_features_for_fold(
            df_train=df_train,  # Use TRAIN for statistics
            df_apply=df_val,
            group_cols=GROUP_COLS,
            date_col=DATE_COL,
            target_col=TARGET_COL,
            lags=lags,
            rolling_windows=rolling_windows,
            holiday_calendar=holiday_calendar,
        )
        
        # Prepare X, y
        exclude_cols = ["year_month"] if "year_month" in df_train_feat.columns else []
        X_train, y_train = prepare_features_and_target(
            df_train_feat, TARGET_COL, DATE_COL, exclude_cols
        )
        X_val, y_val = prepare_features_and_target(
            df_val_feat, TARGET_COL, DATE_COL, exclude_cols
        )
        
        # ---------------------------------------------------------------------
        # Encode categoricals (fit on TRAIN only)
        # ---------------------------------------------------------------------
        X_train_enc, encoders = encode_categoricals_fit(X_train)
        X_val_enc = encode_categoricals_transform(X_val, encoders)
        
        # Align columns
        common_cols = [c for c in X_train_enc.columns if c in X_val_enc.columns]
        X_train_enc = X_train_enc[common_cols]
        X_val_enc = X_val_enc[common_cols]
        
        n_features = len(common_cols)
        
        # ---------------------------------------------------------------------
        # Train XGBoost
        # ---------------------------------------------------------------------
        model = XGBRegressor(**BEST_PARAMS)
        
        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_val_enc, y_val)],
            verbose=False,
        )
        
        # ---------------------------------------------------------------------
        # Evaluate
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
            "n_features": n_features,
        })
        
        logger.info(
            f"Fold {fold_idx + 1}: R²={fold_r2:.4f}, MAE={fold_mae:.2f}, "
            f"RMSE={fold_rmse:.2f}, Features={n_features}"
        )
    
    # =========================================================================
    # Step 6: Aggregate metrics
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Aggregating metrics...")
    logger.info("-" * 70)
    
    mae_list = [f["mae"] for f in fold_results]
    rmse_list = [f["rmse"] for f in fold_results]
    r2_list = [f["r2"] for f in fold_results]
    
    results = {
        "experiment_name": args.exp_name,
        "model_version": "phase4_v3_enhanced",
        "n_folds": len(folds),
        "folds": fold_results,
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "config": {
            "lags": lags,
            "rolling_windows": rolling_windows,
            "gap_months": args.gap_months,
            "min_train_months": args.min_train_months,
            "n_features": n_features,
        },
        "hyperparams": {k: v for k, v in BEST_PARAMS.items() 
                        if k not in ["random_state", "n_jobs"]},
        "leakage_prevention": {
            "gap_months": args.gap_months,
            "train_only_feature_engineering": True,
            "train_only_encoders": True,
            "expanding_window": True,
            "no_shuffle": True,
            "lag_shift_used": True,
            "rolling_shift_used": True,
        },
    }
    
    # Print summary
    print_cv_summary(results, args.exp_name)
    
    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    if not args.no_save:
        logger.info("Saving results...")
        
        exp_paths = get_experiment_paths(args.exp_name)
        ensure_experiment_dir(args.exp_name)
        
        # Save JSON
        json_path = exp_paths["dir"] / "metrics_timeseries_cv_with_features.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {json_path}")
        
        # Save CSV
        csv_path = exp_paths["dir"] / "metrics_timeseries_cv_with_features.csv"
        pd.DataFrame(fold_results).to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
        
        # Save config
        config_path = exp_paths["dir"] / "params.json"
        with open(config_path, "w") as f:
            json.dump(results["config"], f, indent=2)
        logger.info(f"Saved: {config_path}")
        
        logger.info(f"\n📁 Results saved to: {exp_paths['dir']}")
    
    # =========================================================================
    # Compare to Phase-4 v2
    # =========================================================================
    if PHASE4_V2_FINAL.final_metrics.exists():
        with open(PHASE4_V2_FINAL.final_metrics) as f:
            v2_metrics = json.load(f)
        
        v2_val_r2 = v2_metrics.get("split", {}).get("val", {}).get("r2")
        v2_val_mae = v2_metrics.get("split", {}).get("val", {}).get("mae")
        
        print("\n" + "=" * 70)
        print("COMPARISON: v3 Enhanced CV vs v2 Single Split")
        print("=" * 70)
        print(f"{'Metric':<15} {'v3 CV Mean±Std':>22} {'v2 Single':>15}")
        print("-" * 55)
        if v2_val_r2:
            print(f"{'Val R²':<15} {results['r2_mean']:.4f} ± {results['r2_std']:.4f}   {v2_val_r2:>15.4f}")
        if v2_val_mae:
            print(f"{'Val MAE':<15} {results['mae_mean']:.2f} ± {results['mae_std']:.2f}   {v2_val_mae:>15.2f}")
        print("-" * 55)
        print("\n💡 v3 CV metrics are more realistic (leakage-safe, multi-fold).")
        print("=" * 70)
    
    logger.info("\n✅ Phase-4 v3 enhanced CV complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
