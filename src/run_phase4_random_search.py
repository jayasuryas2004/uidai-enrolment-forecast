#!/usr/bin/env python
"""
run_phase4_random_search.py
===========================

Phase-4 Random Search Tuning: Hyperparameter optimization for XGBoost on UIDAI data.

**PURPOSE:**
This script performs a MANUAL RANDOM SEARCH over XGBoost hyperparameters to improve
upon the Phase-4 baseline. It uses the SAME data and splits as the baseline to ensure
fair comparison.

**DATA:**
Uses the same processed UIDAI district-month dataset as the baseline:
    - Path: data/processed/district_month_modeling.csv
    - Target: total_enrolment
    - Train end: 2025-09-30
    - Val end: 2025-10-31
    - Skip feature engineering: True (data already processed)

**SEARCH SPACE:**
Compact ranges around the baseline hyperparameters:
    - max_depth: [3, 4, 5]
    - learning_rate: [0.03, 0.05, 0.07]
    - subsample: [0.7, 0.9, 1.0]
    - colsample_bytree: [0.7, 0.9, 1.0]
    - reg_lambda: [0.5, 1.0, 2.0]
    - reg_alpha: [0.0, 0.1, 0.3]

**BASELINE REFERENCE (to beat):**
    Val R²:   0.9852
    Val MAE:  42.80

Usage:
    python -m src.run_phase4_random_search --n-trials 20

    # With custom settings:
    python -m src.run_phase4_random_search \
        --n-trials 30 \
        --random-seed 123 \
        --top-k 10

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

# Import from our modules
try:
    from src.ctx_v3_features import build_ctx_v3_features, time_based_split
    from src.xgb_trainer import (
        encode_categoricals,
        evaluate_model,
        train_xgb_with_early_stopping,
    )
    from src.run_phase4_baseline import UIDAI_CONFIG
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
        train_xgb_with_early_stopping,
    )
    from run_phase4_baseline import UIDAI_CONFIG
    from phase4_feature_enhancements import (
        add_calendar_features,
        add_group_aggregates,
    )


# =============================================================================
# Configuration
# =============================================================================

# Hyperparameter search space (compact ranges around baseline)
SEARCH_SPACE: Dict[str, List[Any]] = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.07],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.1, 0.3],
}

# Fixed hyperparameters (not tuned)
FIXED_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,  # Large value, will early stop
    "random_state": 42,
    "n_jobs": -1,
}

# Early stopping configuration
EARLY_STOPPING_ROUNDS = 50

# Baseline metrics (for reference)
BASELINE_METRICS = {
    "val_r2": 0.9852,
    "val_mae": 42.80,
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
# Hyperparameter Sampling
# =============================================================================

def sample_params(rng: random.Random) -> Dict[str, Any]:
    """
    Sample a random hyperparameter combination from the search space.

    Parameters
    ----------
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Sampled hyperparameters merged with fixed params.
    """
    sampled = {
        key: rng.choice(values)
        for key, values in SEARCH_SPACE.items()
    }
    # Merge with fixed params
    return {**FIXED_PARAMS, **sampled}


def get_search_space_size() -> int:
    """Calculate total number of possible combinations in search space."""
    size = 1
    for values in SEARCH_SPACE.values():
        size *= len(values)
    return size


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase-4 random search hyperparameter tuning for XGBoost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration (defaults from UIDAI_CONFIG)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=UIDAI_CONFIG["data_path"],
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=UIDAI_CONFIG["date_col"],
        help="Name of the date column.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=UIDAI_CONFIG["target_col"],
        help="Name of the target column.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=UIDAI_CONFIG["train_end"],
        help="End date for training set (inclusive), format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=UIDAI_CONFIG["val_end"],
        help="End date for validation set (inclusive), format: YYYY-MM-DD.",
    )

    # Search configuration
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of random search trials.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configurations to display.",
    )

    # Feature engineering
    parser.add_argument(
        "--skip-feature-eng",
        action="store_true",
        default=True,
        help="Skip ctx_v3 feature engineering (use when data already has features).",
    )
    parser.add_argument(
        "--enhance-features",
        action="store_true",
        help="Apply Phase-4 feature enhancements (calendar, group aggregates).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save results and best model.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        default=True,
        help="Save the best model after retraining on train+val.",
    )
    parser.add_argument(
        "--no-save-best",
        action="store_true",
        help="Do not save the best model.",
    )

    return parser.parse_args()


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """
    Run Phase-4 random search hyperparameter tuning.

    Steps:
        1. Load data and create time-based splits (same as baseline)
        2. Encode categorical columns
        3. Run N random search trials
        4. Report top-K configurations
        5. Retrain best config on train+val and evaluate on test
        6. Save best model
    """
    # Parse arguments and setup
    args = parse_args()
    logger = setup_logging()

    # Set random seeds for reproducibility
    rng = random.Random(args.random_seed)
    np.random.seed(args.random_seed)

    logger.info("=" * 60)
    logger.info("PHASE-4 RANDOM SEARCH TUNING")
    logger.info("=" * 60)
    logger.info(f"Search space size: {get_search_space_size()} combinations")
    logger.info(f"Running {args.n_trials} random trials")
    logger.info(f"Random seed: {args.random_seed}")

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info(f"Loading data from: {args.data_path}")

    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # -------------------------------------------------------------------------
    # Step 2: Feature engineering (skip if data already processed)
    # -------------------------------------------------------------------------
    if args.skip_feature_eng:
        logger.info("Skipping feature engineering (--skip-feature-eng)")
        df_ctx = df.copy()
        df_ctx[args.date_col] = pd.to_datetime(df_ctx[args.date_col])
    else:
        logger.info("Building ctx_v3 features...")
        df_ctx = build_ctx_v3_features(
            df,
            date_col=args.date_col,
            target_col=args.target_col,
        )

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
        logger.info(f"After enhancements: {df_ctx.shape[1]} columns")

    # -------------------------------------------------------------------------
    # Step 3: Create time-based splits (SAME as baseline)
    # -------------------------------------------------------------------------
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d")
    val_end = datetime.strptime(args.val_end, "%Y-%m-%d")

    logger.info(f"Split boundaries: train <= {train_end.date()}, val <= {val_end.date()}")

    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        df_ctx,
        date_col=args.date_col,
        target_col=args.target_col,
        train_end=train_end,
        val_end=val_end,
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Encode categorical columns
    # -------------------------------------------------------------------------
    logger.info("Encoding categorical columns...")

    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categoricals(
        X_train, X_val, X_test if len(X_test) > 0 else None
    )

    if encoders:
        logger.info(f"Encoded columns: {list(encoders.keys())}")

    if X_test_enc is None:
        X_test_enc = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Step 5: Run random search trials
    # -------------------------------------------------------------------------
    logger.info("\n" + "-" * 60)
    logger.info("Starting random search...")
    logger.info("-" * 60)

    results: List[Dict[str, Any]] = []

    for trial in range(args.n_trials):
        # Sample hyperparameters
        params = sample_params(rng)

        # Train model with early stopping
        try:
            model, metrics = train_xgb_with_early_stopping(
                X_train=X_train_enc,
                y_train=y_train,
                X_val=X_val_enc,
                y_val=y_val,
                params=params,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )

            # Store results
            result = {
                "trial": trial + 1,
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "reg_lambda": params["reg_lambda"],
                "reg_alpha": params["reg_alpha"],
                "val_r2": metrics["val_r2"],
                "val_mae": metrics["val_mae"],
                "val_rmse": metrics["val_rmse"],
                "best_iteration": metrics.get("best_iteration", model.best_iteration),
            }
            results.append(result)

            # Log progress
            improvement = BASELINE_METRICS["val_mae"] - metrics["val_mae"]
            status = "✓" if improvement > 0 else " "
            logger.info(
                f"Trial {trial+1:3d}/{args.n_trials}: "
                f"MAE={metrics['val_mae']:7.2f} R²={metrics['val_r2']:.4f} "
                f"depth={params['max_depth']} lr={params['learning_rate']:.2f} {status}"
            )

        except Exception as e:
            logger.warning(f"Trial {trial+1} failed: {e}")
            continue

    # -------------------------------------------------------------------------
    # Step 6: Analyze and report results
    # -------------------------------------------------------------------------
    if not results:
        logger.error("No successful trials. Exiting.")
        sys.exit(1)

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["val_mae", "val_r2"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # Print top-K configurations
    logger.info("\n" + "=" * 60)
    logger.info(f"TOP {args.top_k} CONFIGURATIONS (sorted by val_mae)")
    logger.info("=" * 60)

    print("\n")
    print(f"{'Rank':<5} {'Trial':<6} {'Depth':<6} {'LR':<6} {'Sub':<5} {'Col':<5} "
          f"{'λ':<5} {'α':<5} {'Val MAE':<10} {'Val R²':<10} {'Iters':<6}")
    print("-" * 85)

    for i, row in results_df.head(args.top_k).iterrows():
        rank = i + 1
        print(
            f"{rank:<5} "
            f"{int(row['trial']):<6} "
            f"{int(row['max_depth']):<6} "
            f"{row['learning_rate']:<6.2f} "
            f"{row['subsample']:<5.1f} "
            f"{row['colsample_bytree']:<5.1f} "
            f"{row['reg_lambda']:<5.1f} "
            f"{row['reg_alpha']:<5.1f} "
            f"{row['val_mae']:<10.2f} "
            f"{row['val_r2']:<10.4f} "
            f"{int(row['best_iteration']):<6}"
        )

    print("-" * 85)

    # Compare to baseline
    best_mae = results_df.iloc[0]["val_mae"]
    best_r2 = results_df.iloc[0]["val_r2"]
    mae_improvement = BASELINE_METRICS["val_mae"] - best_mae
    r2_improvement = best_r2 - BASELINE_METRICS["val_r2"]

    print(f"\nBaseline:  Val MAE = {BASELINE_METRICS['val_mae']:.2f}, "
          f"Val R² = {BASELINE_METRICS['val_r2']:.4f}")
    print(f"Best:      Val MAE = {best_mae:.2f}, Val R² = {best_r2:.4f}")
    print(f"Change:    MAE {mae_improvement:+.2f}, R² {r2_improvement:+.4f}")

    if mae_improvement > 0:
        print(f"\n✅ Improved over baseline by {mae_improvement:.2f} MAE!")
    else:
        print(f"\n⚠️ No improvement over baseline (best MAE diff: {mae_improvement:.2f})")

    # -------------------------------------------------------------------------
    # Step 7: Retrain best config on train+val and evaluate on test
    # -------------------------------------------------------------------------
    if len(X_test_enc) > 0:
        logger.info("\n" + "-" * 60)
        logger.info("Retraining best config on train+val, evaluating on test...")
        logger.info("-" * 60)

        # Get best params
        best_row = results_df.iloc[0]
        best_params = {
            "max_depth": int(best_row["max_depth"]),
            "learning_rate": best_row["learning_rate"],
            "subsample": best_row["subsample"],
            "colsample_bytree": best_row["colsample_bytree"],
            "reg_lambda": best_row["reg_lambda"],
            "reg_alpha": best_row["reg_alpha"],
            **FIXED_PARAMS,
        }

        # Combine train + val
        X_trainval_enc = pd.concat([X_train_enc, X_val_enc], ignore_index=True)
        y_trainval = pd.concat([y_train, y_val], ignore_index=True)

        logger.info(f"Training on combined train+val: {X_trainval_enc.shape}")

        # Train final model (no early stopping since we're using all data before test)
        # We'll use a smaller n_estimators based on best_iteration from search
        final_params = best_params.copy()
        final_params["n_estimators"] = int(best_row["best_iteration"] * 1.1)  # 10% buffer

        from xgboost import XGBRegressor
        final_model = XGBRegressor(**final_params)
        final_model.fit(X_trainval_enc, y_trainval)

        # Evaluate on test
        test_metrics = evaluate_model(final_model, X_test_enc, y_test, split_name="test")

        print("\n" + "=" * 60)
        print("PHASE-4 TUNED BEST MODEL - TEST RESULTS")
        print("=" * 60)
        print(f"\nBest hyperparameters:")
        for key in SEARCH_SPACE.keys():
            print(f"  {key}: {best_params[key]}")

        print(f"\nTest metrics (trained on train+val):")
        print(f"  R²:   {test_metrics['test_r2']:.4f}")
        print(f"  MAE:  {test_metrics['test_mae']:.2f}")
        print(f"  RMSE: {test_metrics['test_rmse']:.2f}")
        print("=" * 60)

        # -------------------------------------------------------------------------
        # Step 8: Save best model and results
        # -------------------------------------------------------------------------
        save_best = args.save_best and not args.no_save_best

        if save_best:
            args.output_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = args.output_dir / "xgb_phase4_tuned.pkl"
            joblib.dump(final_model, model_path)
            logger.info(f"Best model saved to: {model_path}")

            # Save encoders
            encoders_path = args.output_dir / "xgb_phase4_tuned.encoders.pkl"
            joblib.dump(encoders, encoders_path)
            logger.info(f"Encoders saved to: {encoders_path}")

            # Save search results
            results_path = args.output_dir / "phase4_random_search_results.csv"
            results_df.to_csv(results_path, index=False)
            logger.info(f"Search results saved to: {results_path}")

            # Save best params as JSON
            import json
            params_path = args.output_dir / "phase4_best_params.json"
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2, default=str)
            logger.info(f"Best params saved to: {params_path}")

    logger.info("\nPhase-4 random search complete!")


# =============================================================================
# Convenience Function
# =============================================================================

def run_phase4_random_search_for_uidai(
    n_trials: int = 20,
    random_seed: int = 42,
    save_best: bool = True,
) -> None:
    """
    Convenience wrapper to run Phase-4 random search on the UIDAI dataset.

    Parameters
    ----------
    n_trials : int, default 20
        Number of random search trials.
    random_seed : int, default 42
        Random seed for reproducibility.
    save_best : bool, default True
        If True, save the best model and results.

    Examples
    --------
    >>> from src.run_phase4_random_search import run_phase4_random_search_for_uidai
    >>> run_phase4_random_search_for_uidai(n_trials=30)
    """
    sys.argv = [
        "run_phase4_random_search",
        "--n-trials", str(n_trials),
        "--random-seed", str(random_seed),
        "--skip-feature-eng",
    ]

    if not save_best:
        sys.argv.append("--no-save-best")

    main()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
