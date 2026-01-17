#!/usr/bin/env python
"""
run_phase4_v2_with_enhanced_features.py
========================================

Phase-4 v2 Pipeline: XGBoost with Enhanced Features for UIDAI Forecasting.

**PURPOSE:**
This script creates a Phase-4 v2 model that includes additional features
(calendar features, group aggregates) on top of the baseline processed data.
It produces SEPARATE artifacts from v1 and compares performance.

**DIFFERENCES FROM v1:**
- v1: Uses only the base district_month_modeling.csv features
- v2: Adds calendar features (quarter, FY, festival months) + group aggregates
      (rolling means, ratio to state mean, long-term stats)

**DATA:**
Same base dataset as v1:
    - Path: data/processed/district_month_modeling.csv
    - Target: total_enrolment
    - Splits: train <= 2025-09-30, val <= 2025-10-31, test > 2025-10-31

**ARTIFACTS (v2):**
    - artifacts/xgb_phase4_v2_tuned_best.pkl
    - artifacts/xgb_phase4_v2_tuned_best.encoders.pkl
    - artifacts/xgb_phase4_v2_random_search_results.csv
    - artifacts/xgb_phase4_v2_tuned_best_params.json
    - artifacts/xgb_phase4_v2_tuned_best_metrics.json

Usage:
    # Run Phase-4 v2 tuning with 15 trials
    python -m src.run_phase4_v2_with_enhanced_features --n-trials 15

    # Compare to v1 after completion
    python -m src.run_phase4_v2_with_enhanced_features --n-trials 20 --random-seed 123

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Import from our modules
try:
    from src.phase4_feature_enhancements import (
        add_calendar_features,
        add_group_aggregates,
    )
    from src.ctx_v3_features import time_based_split
    from src.xgb_trainer import (
        encode_categoricals,
        evaluate_model,
        train_xgb_with_early_stopping,
    )
    from src.phase4_model_registry import (
        PHASE4_V2_FINAL,
        check_not_overwriting_final,
        get_experiment_paths,
    )
except ImportError:
    # Fallback for direct script execution
    from phase4_feature_enhancements import (
        add_calendar_features,
        add_group_aggregates,
    )
    from ctx_v3_features import time_based_split
    from xgb_trainer import (
        encode_categoricals,
        evaluate_model,
        train_xgb_with_early_stopping,
    )
    from phase4_model_registry import (
        PHASE4_V2_FINAL,
        check_not_overwriting_final,
        get_experiment_paths,
    )


# =============================================================================
# Configuration
# =============================================================================

PHASE4_V2_CONFIG: Dict[str, Any] = {
    "data_path": Path("data/processed/district_month_modeling.csv"),
    "date_col": "month_date",
    "target_col": "total_enrolment",
    "train_end": "2025-09-30",
    "val_end": "2025-10-31",
    "artifacts_dir": Path("artifacts"),
    "model_type": "xgboost_phase4_v2_enhanced_features",
}

# Hyperparameter search space (same as v1 for fair comparison)
SEARCH_SPACE: Dict[str, List[Any]] = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.07],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.1, 0.3],
}

# Fixed hyperparameters
FIXED_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 50

# V1 metrics path for comparison
V1_METRICS_PATH = Path("artifacts/xgb_phase4_tuned_best_metrics.json")


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging with a simple format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)


# =============================================================================
# Hyperparameter Sampling
# =============================================================================

def sample_params(rng: random.Random) -> Dict[str, Any]:
    """Sample a random hyperparameter combination."""
    sampled = {
        key: rng.choice(values)
        for key, values in SEARCH_SPACE.items()
    }
    return {**FIXED_PARAMS, **sampled}


def get_search_space_size() -> int:
    """Calculate total combinations in search space."""
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
        description="Run Phase-4 v2 with enhanced features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=PHASE4_V2_CONFIG["data_path"],
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=PHASE4_V2_CONFIG["date_col"],
        help="Name of the date column.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=PHASE4_V2_CONFIG["target_col"],
        help="Name of the target column.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=PHASE4_V2_CONFIG["train_end"],
        help="End date for training set.",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=PHASE4_V2_CONFIG["val_end"],
        help="End date for validation set.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
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
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PHASE4_V2_CONFIG["artifacts_dir"],
        help="Directory to save artifacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving artifacts.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name. If provided, saves to experiments/<exp_name>/ instead of final paths.",
    )
    parser.add_argument(
        "--force-overwrite-final",
        action="store_true",
        help="Allow overwriting FINAL v2 artifacts. USE WITH EXTREME CAUTION.",
    )

    return parser.parse_args()


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """
    Run Phase-4 v2 pipeline with enhanced features.

    Steps:
        1. Load data
        2. Apply Phase-4 feature enhancements
        3. Create time-based splits (same as v1)
        4. Encode categoricals
        5. Run random search tuning
        6. Train final model on train+val
        7. Evaluate and save artifacts
        8. Compare to v1 metrics
    """
    args = parse_args()
    logger = setup_logging()

    # Set random seeds
    rng = random.Random(args.random_seed)
    np.random.seed(args.random_seed)

    logger.info("=" * 70)
    logger.info("PHASE-4 v2: ENHANCED FEATURES PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Search space: {get_search_space_size()} combinations")
    logger.info(f"Running {args.n_trials} random trials")

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    logger.info(f"Loading data from: {args.data_path}")

    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    original_shape = df.shape
    logger.info(f"Original data: {original_shape[0]:,} rows, {original_shape[1]} columns")

    # =========================================================================
    # Step 2: Apply Phase-4 feature enhancements
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Applying Phase-4 v2 feature enhancements...")
    logger.info("-" * 70)

    # Ensure date column is datetime
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    # Add calendar features
    df_feat = add_calendar_features(df, date_col=args.date_col)
    logger.info(f"After calendar features: {df_feat.shape[1]} columns")

    # Add group aggregates
    df_feat = add_group_aggregates(
        df_feat,
        target_col=args.target_col,
        date_col=args.date_col,
        group_cols=["state", "district"] if "state" in df_feat.columns else None,
    )
    logger.info(f"After group aggregates: {df_feat.shape[1]} columns")

    new_cols = [c for c in df_feat.columns if c not in df.columns]
    logger.info(f"New features added ({len(new_cols)}): {new_cols}")

    # =========================================================================
    # Step 3: Time-based split (SAME boundaries as v1)
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Creating time-based splits (same as v1)...")
    logger.info("-" * 70)

    train_end = datetime.fromisoformat(args.train_end)
    val_end = datetime.fromisoformat(args.val_end)

    logger.info(f"Train: <= {train_end.date()}, Val: <= {val_end.date()}, Test: > {val_end.date()}")

    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        df_feat,
        date_col=args.date_col,
        target_col=args.target_col,
        train_end=train_end,
        val_end=val_end,
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # =========================================================================
    # Step 4: Encode categorical columns
    # =========================================================================
    logger.info("Encoding categorical columns...")

    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categoricals(
        X_train, X_val, X_test if len(X_test) > 0 else None
    )

    if encoders:
        logger.info(f"Encoded columns: {list(encoders.keys())}")

    if X_test_enc is None:
        X_test_enc = pd.DataFrame()

    # =========================================================================
    # Step 5: Random search tuning
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Starting random search tuning...")
    logger.info("-" * 70)

    results: List[Dict[str, Any]] = []

    for trial in range(args.n_trials):
        params = sample_params(rng)

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

            logger.info(
                f"Trial {trial+1:3d}/{args.n_trials}: "
                f"MAE={metrics['val_mae']:7.2f} R²={metrics['val_r2']:.4f} "
                f"depth={params['max_depth']} lr={params['learning_rate']:.2f}"
            )

        except Exception as e:
            logger.warning(f"Trial {trial+1} failed: {e}")
            continue

    if not results:
        logger.error("No successful trials!")
        sys.exit(1)

    # Sort results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["val_mae", "val_r2"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # Display top-K
    logger.info("\n" + "=" * 70)
    logger.info(f"TOP {args.top_k} CONFIGURATIONS (v2 with enhanced features)")
    logger.info("=" * 70)

    print("\n")
    print(f"{'Rank':<5} {'Depth':<6} {'LR':<6} {'Sub':<5} {'Col':<5} "
          f"{'λ':<5} {'α':<5} {'Val MAE':<10} {'Val R²':<10}")
    print("-" * 70)

    for i, row in results_df.head(args.top_k).iterrows():
        print(
            f"{i+1:<5} "
            f"{int(row['max_depth']):<6} "
            f"{row['learning_rate']:<6.2f} "
            f"{row['subsample']:<5.1f} "
            f"{row['colsample_bytree']:<5.1f} "
            f"{row['reg_lambda']:<5.1f} "
            f"{row['reg_alpha']:<5.1f} "
            f"{row['val_mae']:<10.2f} "
            f"{row['val_r2']:<10.4f}"
        )
    print("-" * 70)

    # =========================================================================
    # Step 6: Train final model on train+val with best params
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Training final v2 model with best params on train+val...")
    logger.info("-" * 70)

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

    # Combine train + val for final training
    X_trainval_enc = pd.concat([X_train_enc, X_val_enc], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    logger.info(f"Training on combined train+val: {X_trainval_enc.shape}")

    # Use best_iteration with 10% buffer for final model
    final_n_estimators = int(best_row["best_iteration"] * 1.1)
    final_params = best_params.copy()
    final_params["n_estimators"] = final_n_estimators

    final_model = XGBRegressor(**final_params)
    final_model.fit(X_trainval_enc, y_trainval)

    logger.info(f"Final model trained with {final_n_estimators} estimators")

    # =========================================================================
    # Step 7: Evaluate on all splits
    # =========================================================================
    logger.info("Evaluating final v2 model...")

    train_metrics = evaluate_model(final_model, X_train_enc, y_train, split_name="train")
    val_metrics = evaluate_model(final_model, X_val_enc, y_val, split_name="val")

    if len(X_test_enc) > 0:
        test_metrics = evaluate_model(final_model, X_test_enc, y_test, split_name="test")
    else:
        test_metrics = {"test_r2": None, "test_mae": None, "test_rmse": None}

    # Build metrics dict
    metrics_v2 = {
        "split": {
            "train": {
                "r2": train_metrics["train_r2"],
                "mae": train_metrics["train_mae"],
                "rmse": train_metrics["train_rmse"],
            },
            "val": {
                "r2": val_metrics["val_r2"],
                "mae": val_metrics["val_mae"],
                "rmse": val_metrics["val_rmse"],
            },
            "test": {
                "r2": test_metrics.get("test_r2"),
                "mae": test_metrics.get("test_mae"),
                "rmse": test_metrics.get("test_rmse"),
            },
        },
        "model_type": PHASE4_V2_CONFIG["model_type"],
        "best_params": {k: v for k, v in best_params.items() 
                        if k not in ["objective", "random_state", "n_jobs"]},
        "n_features": X_train_enc.shape[1],
        "feature_enhancements": new_cols,
        "config": {
            "data_path": str(args.data_path),
            "train_end": args.train_end,
            "val_end": args.val_end,
            "n_trials": args.n_trials,
            "random_seed": args.random_seed,
        },
    }

    # Print v2 results
    print("\n" + "=" * 70)
    print("PHASE-4 v2 FINAL RESULTS (Enhanced Features)")
    print("=" * 70)
    print(f"\n{'Split':<10} {'R²':>12} {'MAE':>12} {'RMSE':>12}")
    print("-" * 50)
    print(f"{'Train':<10} {train_metrics['train_r2']:>12.4f} "
          f"{train_metrics['train_mae']:>12.2f} {train_metrics['train_rmse']:>12.2f}")
    print(f"{'Val':<10} {val_metrics['val_r2']:>12.4f} "
          f"{val_metrics['val_mae']:>12.2f} {val_metrics['val_rmse']:>12.2f}")
    if test_metrics.get("test_r2") is not None:
        print(f"{'Test':<10} {test_metrics['test_r2']:>12.4f} "
              f"{test_metrics['test_mae']:>12.2f} {test_metrics['test_rmse']:>12.2f}")
    print("-" * 50)
    print(f"\nBest hyperparameters:")
    for k, v in best_params.items():
        if k not in ["objective", "random_state", "n_jobs", "n_estimators"]:
            print(f"  {k}: {v}")
    print("=" * 70)

    # =========================================================================
    # Step 8: Compare to v1 metrics
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Comparing v2 to v1...")
    logger.info("-" * 70)

    if V1_METRICS_PATH.exists():
        with open(V1_METRICS_PATH, "r") as f:
            metrics_v1 = json.load(f)

        print("\n" + "=" * 70)
        print("PHASE-4 v1 vs v2 COMPARISON")
        print("=" * 70)

        # Extract v1 metrics
        v1_val_mae = metrics_v1.get("split", {}).get("val", {}).get("mae")
        v1_val_r2 = metrics_v1.get("split", {}).get("val", {}).get("r2")
        v1_test_mae = metrics_v1.get("split", {}).get("test", {}).get("mae")
        v1_test_r2 = metrics_v1.get("split", {}).get("test", {}).get("r2")

        # v2 metrics
        v2_val_mae = val_metrics["val_mae"]
        v2_val_r2 = val_metrics["val_r2"]
        v2_test_mae = test_metrics.get("test_mae")
        v2_test_r2 = test_metrics.get("test_r2")

        print(f"\n{'Metric':<15} {'v1':>12} {'v2':>12} {'Δ':>12} {'Better?':>10}")
        print("-" * 65)

        if v1_val_mae is not None:
            delta_val_mae = v2_val_mae - v1_val_mae
            better_val_mae = "✓ v2" if delta_val_mae < 0 else ("✓ v1" if delta_val_mae > 0 else "=")
            print(f"{'Val MAE':<15} {v1_val_mae:>12.2f} {v2_val_mae:>12.2f} "
                  f"{delta_val_mae:>+12.2f} {better_val_mae:>10}")

        if v1_val_r2 is not None:
            delta_val_r2 = v2_val_r2 - v1_val_r2
            better_val_r2 = "✓ v2" if delta_val_r2 > 0 else ("✓ v1" if delta_val_r2 < 0 else "=")
            print(f"{'Val R²':<15} {v1_val_r2:>12.4f} {v2_val_r2:>12.4f} "
                  f"{delta_val_r2:>+12.4f} {better_val_r2:>10}")

        if v1_test_mae is not None and v2_test_mae is not None:
            delta_test_mae = v2_test_mae - v1_test_mae
            better_test_mae = "✓ v2" if delta_test_mae < 0 else ("✓ v1" if delta_test_mae > 0 else "=")
            print(f"{'Test MAE':<15} {v1_test_mae:>12.2f} {v2_test_mae:>12.2f} "
                  f"{delta_test_mae:>+12.2f} {better_test_mae:>10}")

        if v1_test_r2 is not None and v2_test_r2 is not None:
            delta_test_r2 = v2_test_r2 - v1_test_r2
            better_test_r2 = "✓ v2" if delta_test_r2 > 0 else ("✓ v1" if delta_test_r2 < 0 else "=")
            print(f"{'Test R²':<15} {v1_test_r2:>12.4f} {v2_test_r2:>12.4f} "
                  f"{delta_test_r2:>+12.4f} {better_test_r2:>10}")

        print("-" * 65)

        # Overall verdict
        improvements = 0
        if v1_val_mae is not None and v2_val_mae < v1_val_mae:
            improvements += 1
        if v1_test_mae is not None and v2_test_mae is not None and v2_test_mae < v1_test_mae:
            improvements += 1

        if improvements >= 2:
            print("\n🎉 v2 (enhanced features) OUTPERFORMS v1!")
        elif improvements == 1:
            print("\n📊 v2 shows MIXED results compared to v1.")
        else:
            print("\n⚠️  v1 still performs better. Consider feature selection or tuning.")

        print("=" * 70)

    else:
        logger.warning(f"v1 metrics not found at {V1_METRICS_PATH}. Skipping comparison.")
        print("\n⚠️  No v1 metrics found for comparison.")

    # =========================================================================
    # Step 9: Save artifacts (with overwrite protection)
    # =========================================================================
    if not args.no_save:
        logger.info("\n" + "-" * 70)
        logger.info("Saving v2 artifacts...")
        logger.info("-" * 70)

        # Determine output paths based on --exp-name
        if args.exp_name:
            # Experiment mode: save to experiments/<exp_name>/
            exp_paths = get_experiment_paths(args.exp_name)
            exp_paths["dir"].mkdir(parents=True, exist_ok=True)
            model_path = exp_paths["model"]
            encoders_path = exp_paths["encoders"]
            results_path = exp_paths["search_results"]
            params_path = exp_paths["params"]
            metrics_path = exp_paths["metrics"]
            logger.info(f"📁 Saving to EXPERIMENT directory: {exp_paths['dir']}")
        else:
            # Default mode: save to final paths (with protection)
            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.artifacts_dir / "xgb_phase4_v2_tuned_best.pkl"
            encoders_path = args.artifacts_dir / "xgb_phase4_v2_tuned_best.encoders.pkl"
            results_path = args.artifacts_dir / "xgb_phase4_v2_random_search_results.csv"
            params_path = args.artifacts_dir / "xgb_phase4_v2_tuned_best_params.json"
            metrics_path = args.artifacts_dir / "xgb_phase4_v2_tuned_best_metrics.json"

            # Check for accidental overwrites of FINAL artifacts
            if not args.force_overwrite_final:
                for path in [model_path, encoders_path, results_path, params_path, metrics_path]:
                    check_not_overwriting_final(path)
                logger.warning(
                    "⚠️  Saving to FINAL artifact paths. "
                    "Use --exp-name to save to experiment directory instead."
                )

        # Save model
        joblib.dump(final_model, model_path)
        logger.info(f"Model saved: {model_path}")

        # Save encoders
        joblib.dump(encoders, encoders_path)
        logger.info(f"Encoders saved: {encoders_path}")

        # Save search results
        results_df.to_csv(results_path, index=False)
        logger.info(f"Search results saved: {results_path}")

        # Save best params
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2, default=str)
        logger.info(f"Best params saved: {params_path}")

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics_v2, f, indent=2, default=str)
        logger.info(f"Metrics saved: {metrics_path}")

    logger.info("\n✅ Phase-4 v2 pipeline complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
