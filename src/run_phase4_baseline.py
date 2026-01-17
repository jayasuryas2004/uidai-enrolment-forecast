#!/usr/bin/env python
"""
run_phase4_baseline.py
======================

Phase-4 Baseline Experiment: Train a regularized XGBoost model with early stopping
on UIDAI time-series data.

**PURPOSE:**
This script establishes the OFFICIAL Phase-4 BASELINE for the UIDAI demand forecasting
project. It trains an XGBoost regressor with regularization (L1/L2) and early stopping,
using time-based train/val/test splits to prevent data leakage.

**DATA:**
This baseline uses the PROCESSED UIDAI district-month dataset:
    - Path: data/processed/district_month_modeling.csv
    - Features: Pre-computed lag features, age demographics, segment features
    - Target: total_enrolment (monthly enrolment counts per district)
    - Date range: April 2025 - December 2025 (9 months)

**WHY --skip-feature-eng:**
We use `--skip-feature-eng` because:
    1. The processed dataset already has lag/segment features.
    2. ctx_v3 rolling features (7-day, 30-day windows) would drop ALL rows
       since we only have 9 monthly observations per group.
    3. The existing features are sufficient for baseline modeling.

**BASELINE REFERENCE:**
This baseline is the REFERENCE POINT before hyperparameter tuning via random search.
Phase-4 random search will explore combinations of:
    - max_depth: 3-6
    - learning_rate: 0.01-0.1
    - subsample: 0.7-1.0
    - reg_lambda: 0.1-10.0
    - reg_alpha: 0.0-1.0

**BASELINE RESULTS (2025-01-12):**
    Split     R²        MAE       RMSE
    -----     ------    ------    ------
    Train     1.0000    0.74      0.97
    Val       0.9852    42.80     123.91
    Test      0.9920    51.82     110.46
    Best iteration: 1049 / 2000

Usage:
    # Standard CLI usage
    python -m src.run_phase4_baseline \
        --data-path data/processed/district_month_modeling.csv \
        --date-col month_date \
        --target-col total_enrolment \
        --train-end 2025-09-30 \
        --val-end 2025-10-31 \
        --skip-feature-eng

    # Or use the convenience function:
    python -c "from src.run_phase4_baseline import run_phase4_baseline_for_uidai; run_phase4_baseline_for_uidai()"

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

# Import from our modules (use relative imports for -m execution)
try:
    from src.ctx_v3_features import build_ctx_v3_features, time_based_split
    from src.xgb_trainer import (
        encode_categoricals,
        evaluate_model,
        log_training_summary,
        train_xgb_with_early_stopping,
    )
    from src.phase4_feature_enhancements import (
        add_calendar_features,
        add_group_aggregates,
    )
except ImportError:
    # Fallback for direct script execution
    from ctx_v3_features import build_ctx_v3_features, time_based_split
    from xgb_trainer import (
        encode_categoricals,
        evaluate_model,
        log_training_summary,
        train_xgb_with_early_stopping,
    )
    from phase4_feature_enhancements import (
        add_calendar_features,
        add_group_aggregates,
    )


# =============================================================================
# Configuration
# =============================================================================

# Default XGBoost baseline params (will be tuned in Phase-4 random search later)
BASELINE_PARAMS: Dict[str, Any] = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "random_state": 42,
}

# Early stopping configuration
EARLY_STOPPING_ROUNDS = 50

# UIDAI-specific configuration for processed dataset
UIDAI_CONFIG = {
    "data_path": Path("data/processed/district_month_modeling.csv"),
    "date_col": "month_date",
    "target_col": "total_enrolment",
    "train_end": "2025-09-30",
    "val_end": "2025-10-31",
    "output_model_path": Path("artifacts/xgb_phase4_baseline.pkl"),
}


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging with a simple format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase-4 baseline XGBoost experiment on UIDAI data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="date",
        help="Name of the date column.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="y",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        required=True,
        help="End date for training set (inclusive), format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        required=True,
        help="End date for validation set (inclusive), format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        nargs="+",
        default=None,
        help="Group columns for feature engineering (e.g., state district).",
    )
    parser.add_argument(
        "--skip-feature-eng",
        action="store_true",
        help="Skip ctx_v3 feature engineering (use when data already has features).",
    )
    parser.add_argument(
        "--enhance-features",
        action="store_true",
        help="Apply Phase-4 feature enhancements (calendar, group aggregates).",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=Path("artifacts/xgb_phase4_baseline.pkl"),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the model to disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose XGBoost training output.",
    )

    return parser.parse_args()


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """
    Run the Phase-4 baseline experiment.

    Steps:
        1. Load data from CSV
        2. Build ctx_v3 features
        3. Create time-based splits
        4. Encode categorical columns
        5. Train XGBoost with early stopping
        6. Evaluate on train/val/test
        7. Save model to disk
    """
    # Parse arguments and setup logging
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("PHASE-4 BASELINE EXPERIMENT")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info(f"Loading data from: {args.data_path}")

    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.data_path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Validate required columns exist
    if args.date_col not in df.columns:
        logger.error(f"Date column '{args.date_col}' not found in data")
        sys.exit(1)
    if args.target_col not in df.columns:
        logger.error(f"Target column '{args.target_col}' not found in data")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: Build ctx_v3 features (or skip if data already has features)
    # -------------------------------------------------------------------------
    if args.skip_feature_eng:
        logger.info("Skipping feature engineering (--skip-feature-eng flag set)")
        df_ctx = df.copy()
        # Ensure date column is datetime
        df_ctx[args.date_col] = pd.to_datetime(df_ctx[args.date_col])
    else:
        logger.info("Building ctx_v3 features...")
        try:
            df_ctx = build_ctx_v3_features(
                df,
                date_col=args.date_col,
                target_col=args.target_col,
                group_cols=args.group_cols,
            )
            logger.info(f"After feature engineering: {df_ctx.shape}")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Step 2b: Apply Phase-4 feature enhancements (optional)
    # -------------------------------------------------------------------------
    if args.enhance_features:
        logger.info("Applying Phase-4 feature enhancements...")
        df_ctx = add_calendar_features(df_ctx, date_col=args.date_col)
        df_ctx = add_group_aggregates(
            df_ctx,
            target_col=args.target_col,
            date_col=args.date_col,
            group_cols=["state", "district"] if "state" in df_ctx.columns else None,
        )
        logger.info(f"After enhancements: {df_ctx.shape} columns")
    
    logger.info(f"Data shape for splitting: {df_ctx.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Parse split dates and create time-based splits
    # -------------------------------------------------------------------------
    try:
        train_end = datetime.strptime(args.train_end, "%Y-%m-%d")
        val_end = datetime.strptime(args.val_end, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    logger.info(f"Split boundaries: train <= {train_end.date()}, val <= {val_end.date()}")

    try:
        X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
            df_ctx,
            date_col=args.date_col,
            target_col=args.target_col,
            train_end=train_end,
            val_end=val_end,
        )
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    except ValueError as e:
        logger.error(f"Time-based split failed: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 4: Encode categorical columns
    # -------------------------------------------------------------------------
    logger.info("Encoding categorical columns...")

    # Use encode_categoricals which handles train/val/test together
    # to ensure consistent encoding across splits
    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categoricals(
        X_train, X_val, X_test if len(X_test) > 0 else None
    )
    
    if encoders:
        logger.info(f"Encoded columns: {list(encoders.keys())}")
    else:
        logger.info("No categorical columns to encode")

    # Handle case where X_test was None
    if X_test_enc is None:
        X_test_enc = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Step 5: Train XGBoost with early stopping
    # -------------------------------------------------------------------------
    logger.info("Training XGBoost baseline with early stopping...")
    logger.info(f"Baseline params: max_depth={BASELINE_PARAMS['max_depth']}, "
                f"lr={BASELINE_PARAMS['learning_rate']}, "
                f"reg_lambda={BASELINE_PARAMS['reg_lambda']}")

    # =========================================================================
    # NOTE: Phase-4 random search will replace this with a hyperparameter loop
    # that explores different combinations of max_depth, learning_rate,
    # subsample, colsample_bytree, reg_lambda, reg_alpha.
    # =========================================================================

    try:
        model, train_val_metrics = train_xgb_with_early_stopping(
            X_train=X_train_enc,
            y_train=y_train,
            X_val=X_val_enc,
            y_val=y_val,
            params=BASELINE_PARAMS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=args.verbose,
        )
        logger.info(f"Training complete. Best iteration: {model.best_iteration}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 6: Evaluate on all splits
    # -------------------------------------------------------------------------
    logger.info("Evaluating model on all splits...")

    # Train metrics (already computed during training, but recompute for consistency)
    train_metrics = evaluate_model(model, X_train_enc, y_train, split_name="train")

    # Val metrics (use what training returned, which includes best iteration info)
    val_metrics = {
        "val_r2": train_val_metrics["val_r2"],
        "val_mae": train_val_metrics["val_mae"],
        "val_rmse": train_val_metrics["val_rmse"],
    }

    # Test metrics (if test set exists)
    if len(X_test_enc) > 0:
        test_metrics = evaluate_model(model, X_test_enc, y_test, split_name="test")
    else:
        test_metrics = {}
        logger.warning("Test set is empty, skipping test evaluation")

    # Combine all metrics
    all_metrics = {
        **train_metrics,
        **val_metrics,
        **test_metrics,
        "best_iteration": model.best_iteration,
    }

    # -------------------------------------------------------------------------
    # Step 7: Log summary
    # -------------------------------------------------------------------------
    log_training_summary(all_metrics, params=BASELINE_PARAMS)

    # Print a concise final summary
    print("\n" + "=" * 60)
    print("PHASE-4 BASELINE RESULTS")
    print("=" * 60)
    print(f"\n{'Split':<10} {'R²':>10} {'MAE':>12} {'RMSE':>12}")
    print("-" * 46)
    print(f"{'Train':<10} {train_metrics['train_r2']:>10.4f} "
          f"{train_metrics['train_mae']:>12.2f} {train_metrics['train_rmse']:>12.2f}")
    print(f"{'Val':<10} {val_metrics['val_r2']:>10.4f} "
          f"{val_metrics['val_mae']:>12.2f} {val_metrics['val_rmse']:>12.2f}")
    if test_metrics:
        print(f"{'Test':<10} {test_metrics['test_r2']:>10.4f} "
              f"{test_metrics['test_mae']:>12.2f} {test_metrics['test_rmse']:>12.2f}")
    print("-" * 46)
    print(f"Best iteration: {model.best_iteration} / {BASELINE_PARAMS['n_estimators']}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 8: Save model
    # -------------------------------------------------------------------------
    if not args.no_save:
        output_path = args.output_model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to: {output_path}")
        try:
            joblib.dump(model, output_path)
            logger.info("Model saved successfully")

            # Also save encoders for inference
            encoders_path = output_path.with_suffix(".encoders.pkl")
            joblib.dump(encoders, encoders_path)
            logger.info(f"Encoders saved to: {encoders_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            sys.exit(1)

    logger.info("Phase-4 baseline experiment complete!")


# =============================================================================
# UIDAI Convenience Wrapper
# =============================================================================

def run_phase4_baseline_for_uidai(
    save_model: bool = True,
    verbose: bool = False,
) -> None:
    """
    Convenience wrapper to run the Phase-4 baseline on the real UIDAI processed dataset.

    This function provides a simple way to run the baseline experiment without
    needing to specify CLI arguments. It uses the standard UIDAI configuration:
        - Data: data/processed/district_month_modeling.csv
        - Target: total_enrolment
        - Train end: 2025-09-30
        - Val end: 2025-10-31
        - Skip feature engineering (data already processed)

    Parameters
    ----------
    save_model : bool, default True
        If True, save the trained model to artifacts/xgb_phase4_baseline.pkl
    verbose : bool, default False
        If True, enable verbose XGBoost training output.

    Returns
    -------
    None
        Prints results to console and optionally saves model.

    Examples
    --------
    >>> from src.run_phase4_baseline import run_phase4_baseline_for_uidai
    >>> run_phase4_baseline_for_uidai()
    >>> # Or with verbose output:
    >>> run_phase4_baseline_for_uidai(verbose=True)
    """
    # Build equivalent of CLI args
    sys.argv = [
        "run_phase4_baseline",
        "--data-path", str(UIDAI_CONFIG["data_path"]),
        "--date-col", UIDAI_CONFIG["date_col"],
        "--target-col", UIDAI_CONFIG["target_col"],
        "--train-end", UIDAI_CONFIG["train_end"],
        "--val-end", UIDAI_CONFIG["val_end"],
        "--output-model-path", str(UIDAI_CONFIG["output_model_path"]),
        "--skip-feature-eng",
    ]

    if not save_model:
        sys.argv.append("--no-save")
    if verbose:
        sys.argv.append("--verbose")

    # Run the main function
    main()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
