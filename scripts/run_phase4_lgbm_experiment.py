#!/usr/bin/env python
"""
run_phase4_lgbm_experiment.py
=============================

CLI entrypoint to run the Phase-4 LightGBM experiment.

╔════════════════════════════════════════════════════════════════════════════╗
║  NOTE: EXPERIMENT CONCLUDED - XGBoost v3 REMAINS PRODUCTION MODEL          ║
╠════════════════════════════════════════════════════════════════════════════╣
║  We ran this LightGBM experiment and found it underperformed XGBoost v3    ║
║  on time-series cross-validation:                                          ║
║    - Lower R² (0.865 vs 0.955)                                             ║
║    - Higher MAE (242.8 vs 116.6)                                           ║
║    - Unstable fold 1 (only 119 training rows)                              ║
║                                                                            ║
║  For the UIDAI hackathon submission, we KEEP XGBoost v3 as the only        ║
║  production model and treat LightGBM as a documented experiment.           ║
║  No further hyperparameter tuning is planned before final submission.      ║
╚════════════════════════════════════════════════════════════════════════════╝

**PURPOSE:**
Run an experimental LightGBM model with the same CV folds and features
as the frozen Phase-4 v3 XGBoost baseline, then save results for comparison.

**USAGE:**
    python scripts/run_phase4_lgbm_experiment.py

    # Or with custom parameters:
    python scripts/run_phase4_lgbm_experiment.py --n-estimators 800 --learning-rate 0.03

**OUTPUTS:**
    - artifacts/phase4_lgbm_experiment_metrics.json  (CV metrics summary)
    - artifacts/phase4_lgbm_experiment_oof.parquet   (OOF predictions for ensembling)

**COMPARISON:**
    Compare LightGBM metrics against the frozen v3 XGBoost baseline at:
        artifacts/xgb_phase4_v3_baseline_metrics.json

Author: UIDAI Forecast Team
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase4_lgbm_experiment import (
    Phase4LGBMExperiment,
    DEFAULT_LGBM_PARAMS,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase-4 LightGBM experiment for UIDAI enrolment forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--n-estimators", type=int, default=700,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.04,
        help="Learning rate (eta)"
    )
    parser.add_argument(
        "--num-leaves", type=int, default=64,
        help="Maximum number of leaves per tree"
    )
    parser.add_argument(
        "--max-depth", type=int, default=-1,
        help="Maximum tree depth (-1 for unlimited)"
    )
    parser.add_argument(
        "--subsample", type=float, default=0.9,
        help="Row subsample ratio"
    )
    parser.add_argument(
        "--colsample-bytree", type=float, default=0.9,
        help="Column subsample ratio"
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=1.0,
        help="L2 regularization (lambda)"
    )
    parser.add_argument(
        "--reg-alpha", type=float, default=0.0,
        help="L1 regularization (alpha)"
    )
    
    # CV configuration
    parser.add_argument(
        "--n-folds", type=int, default=4,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--gap-months", type=int, default=1,
        help="Gap months between train and validation"
    )
    
    # Output paths
    parser.add_argument(
        "--metrics-path", type=str,
        default="artifacts/phase4_lgbm_experiment_metrics.json",
        help="Path to save CV metrics JSON"
    )
    parser.add_argument(
        "--oof-path", type=str,
        default="artifacts/phase4_lgbm_experiment_oof.parquet",
        help="Path to save OOF predictions (parquet)"
    )
    
    # Flags
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results only, do not save to disk"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("🔬 PHASE-4 LightGBM EXPERIMENT")
    print("=" * 70)
    
    # Build hyperparameters from CLI args
    params = DEFAULT_LGBM_PARAMS.copy()
    params.update({
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
    })
    
    print("\n📋 CONFIGURATION:")
    print(f"  n_estimators:     {params['n_estimators']}")
    print(f"  learning_rate:    {params['learning_rate']}")
    print(f"  num_leaves:       {params['num_leaves']}")
    print(f"  max_depth:        {params['max_depth']}")
    print(f"  subsample:        {params['subsample']}")
    print(f"  colsample_bytree: {params['colsample_bytree']}")
    print(f"  reg_lambda:       {params['reg_lambda']}")
    print(f"  reg_alpha:        {params['reg_alpha']}")
    print(f"  n_folds:          {args.n_folds}")
    print(f"  gap_months:       {args.gap_months}")
    
    # Create experiment
    exp = Phase4LGBMExperiment(
        params=params,
        n_folds=args.n_folds,
        gap_months=args.gap_months,
    )
    
    # Run CV
    cv_results, oof_df = exp.fit_cv()
    
    # Print summary
    exp.print_summary(cv_results)
    
    # Save results
    if not args.no_save:
        metrics_path = Path(args.metrics_path)
        oof_path = Path(args.oof_path)
        
        summary = exp.summarize_cv(cv_results)
        
        # Add hyperparameters to summary
        summary["hyperparameters"] = params
        
        exp.save_cv_metrics(summary, metrics_path)
        
        # Save OOF predictions (for ensembling)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_parquet(oof_path)
        print(f"Saved OOF predictions to: {oof_path}")
        
        print("\n📁 OUTPUTS SAVED:")
        print(f"  Metrics: {metrics_path}")
        print(f"  OOF:     {oof_path}")
        
        # Print comparison instructions
        print("\n🔍 COMPARE WITH v3 XGBoost BASELINE:")
        print("  cat artifacts/xgb_phase4_v3_baseline_metrics.json")
        print("  cat artifacts/phase4_lgbm_experiment_metrics.json")
    else:
        print("\n⚠️  --no-save: Results not saved to disk")
    
    print("\n✅ EXPERIMENT COMPLETE\n")


if __name__ == "__main__":
    main()
