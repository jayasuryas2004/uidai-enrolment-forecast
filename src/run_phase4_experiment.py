#!/usr/bin/env python
"""
run_phase4_experiment.py
========================

Template script for running Phase-4 experiments WITHOUT overwriting the
frozen final v2 model.

This script demonstrates how to:
    1. Use the model registry for experiment paths
    2. Apply feature enhancements
    3. Run hyperparameter tuning
    4. Save artifacts safely to experiment directory
    5. Compare results to the final v2 model

Usage:
    # Run an experiment with a custom name
    python -m src.run_phase4_experiment --exp-name exp1_deeper_trees --n-trials 10

    # Try different hyperparameter ranges
    python -m src.run_phase4_experiment --exp-name exp2_high_lr --n-trials 15

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
from typing import Any, Dict, List

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
        get_experiment_paths,
        ensure_experiment_dir,
    )
except ImportError:
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
        get_experiment_paths,
        ensure_experiment_dir,
    )


# =============================================================================
# Experiment Configuration
# =============================================================================

# Override these for your experiment
EXPERIMENT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "max_depth": [3, 4, 5, 6],          # Try deeper trees
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.1, 0.3],
}

FIXED_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
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
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Phase-4 experiment (saves to experiments directory).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (e.g., 'exp1_deeper_trees'). REQUIRED.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of random search trials.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/district_month_modeling.csv"),
        help="Path to input CSV.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    logger = setup_logging()

    rng = random.Random(args.random_seed)
    np.random.seed(args.random_seed)

    logger.info("=" * 70)
    logger.info(f"PHASE-4 EXPERIMENT: {args.exp_name}")
    logger.info("=" * 70)

    # Get experiment paths (safe - won't overwrite final)
    exp_paths = get_experiment_paths(args.exp_name)
    ensure_experiment_dir(args.exp_name)
    logger.info(f"Artifacts will be saved to: {exp_paths['dir']}")

    # =========================================================================
    # Load and enhance data
    # =========================================================================
    logger.info("Loading and enhancing data...")
    df = pd.read_csv(args.data_path)
    df["month_date"] = pd.to_datetime(df["month_date"])

    df_feat = add_calendar_features(df, date_col="month_date")
    df_feat = add_group_aggregates(
        df_feat,
        target_col="total_enrolment",
        date_col="month_date",
        group_cols=["state", "district"] if "state" in df_feat.columns else None,
    )

    # =========================================================================
    # Time-based split (same as final)
    # =========================================================================
    train_end = datetime.fromisoformat("2025-09-30")
    val_end = datetime.fromisoformat("2025-10-31")

    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        df_feat,
        date_col="month_date",
        target_col="total_enrolment",
        train_end=train_end,
        val_end=val_end,
    )

    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categoricals(
        X_train, X_val, X_test if len(X_test) > 0 else None
    )

    logger.info(f"Train: {X_train_enc.shape}, Val: {X_val_enc.shape}, Test: {X_test_enc.shape if X_test_enc is not None else 'N/A'}")

    # =========================================================================
    # Random search
    # =========================================================================
    logger.info(f"Running {args.n_trials} random search trials...")

    results: List[Dict[str, Any]] = []

    for trial in range(args.n_trials):
        params = {
            key: rng.choice(values)
            for key, values in EXPERIMENT_SEARCH_SPACE.items()
        }
        params.update(FIXED_PARAMS)

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
                **{k: params[k] for k in EXPERIMENT_SEARCH_SPACE},
                "val_r2": metrics["val_r2"],
                "val_mae": metrics["val_mae"],
            }
            results.append(result)

            logger.info(f"Trial {trial+1:3d}/{args.n_trials}: MAE={metrics['val_mae']:.2f} R²={metrics['val_r2']:.4f}")

        except Exception as e:
            logger.warning(f"Trial {trial+1} failed: {e}")

    if not results:
        logger.error("No successful trials!")
        sys.exit(1)

    results_df = pd.DataFrame(results).sort_values("val_mae")
    best = results_df.iloc[0]

    logger.info("\n" + "=" * 70)
    logger.info(f"BEST CONFIG: MAE={best['val_mae']:.2f} R²={best['val_r2']:.4f}")
    logger.info("=" * 70)

    # =========================================================================
    # Train final model with best params
    # =========================================================================
    best_params = {
        k: (int(best[k]) if k == "max_depth" else best[k])
        for k in EXPERIMENT_SEARCH_SPACE
    }
    best_params.update(FIXED_PARAMS)

    X_trainval = pd.concat([X_train_enc, X_val_enc], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_trainval, y_trainval)

    # Evaluate
    train_m = evaluate_model(final_model, X_train_enc, y_train, "train")
    val_m = evaluate_model(final_model, X_val_enc, y_val, "val")
    test_m = evaluate_model(final_model, X_test_enc, y_test, "test") if X_test_enc is not None and len(X_test_enc) > 0 else {}

    metrics_dict = {
        "split": {
            "train": {"r2": train_m.get("train_r2"), "mae": train_m.get("train_mae"), "rmse": train_m.get("train_rmse")},
            "val": {"r2": val_m.get("val_r2"), "mae": val_m.get("val_mae"), "rmse": val_m.get("val_rmse")},
            "test": {"r2": test_m.get("test_r2"), "mae": test_m.get("test_mae"), "rmse": test_m.get("test_rmse")},
        },
        "experiment_name": args.exp_name,
        "best_params": best_params,
    }

    # =========================================================================
    # Save to experiment directory (SAFE)
    # =========================================================================
    logger.info(f"\nSaving artifacts to: {exp_paths['dir']}")

    joblib.dump(final_model, exp_paths["model"])
    joblib.dump(encoders, exp_paths["encoders"])
    results_df.to_csv(exp_paths["search_results"], index=False)

    with open(exp_paths["params"], "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    with open(exp_paths["metrics"], "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)

    logger.info("✅ Artifacts saved!")

    # =========================================================================
    # Compare to final v2 model
    # =========================================================================
    if PHASE4_V2_FINAL.final_metrics.exists():
        with open(PHASE4_V2_FINAL.final_metrics) as f:
            final_metrics = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON: Experiment vs Final v2")
        print("=" * 70)

        final_test_mae = final_metrics.get("split", {}).get("test", {}).get("mae")
        exp_test_mae = test_m.get("test_mae")

        if final_test_mae and exp_test_mae:
            delta = exp_test_mae - final_test_mae
            verdict = "✓ Better" if delta < 0 else ("✗ Worse" if delta > 0 else "= Same")
            print(f"Final v2 Test MAE: {final_test_mae:.2f}")
            print(f"Experiment Test MAE: {exp_test_mae:.2f} ({delta:+.2f}) {verdict}")
        print("=" * 70)

    logger.info("\n🎉 Experiment complete!")


if __name__ == "__main__":
    main()
