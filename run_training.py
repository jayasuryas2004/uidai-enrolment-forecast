#!/usr/bin/env python
"""
UIDAI ASRIS - Training Pipeline Entry Script

This script runs the complete ML pipeline (Phases 2-5) end-to-end:
1. Load and clean data
2. Build features
3. Train champion model with early stopping
4. Evaluate on validation set
5. Run drift checks
6. Generate alerts
7. Save model artifacts

Usage:
    python run_training.py

The script uses the modularized code from src/ to execute the same
logic as the notebook, making it suitable for production/MLOps workflows.

This script:
- Uses centralized configuration (no magic numbers)
- Includes structured logging for monitoring
- Provides clear error messages on failure
- Can be safely called from CI/CD or schedulers
- Contains NO hard-coded secrets or credentials

Author: UIDAI ASRIS Team
Version: 1.1.0
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any

# =============================================================================
# Project Setup
# =============================================================================
# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Verify we're in the correct directory
if not (PROJECT_ROOT / "src").exists():
    print(f"ERROR: src/ directory not found in {PROJECT_ROOT}")
    print("Please run this script from the project root directory.")
    sys.exit(1)


# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure structured logging for the training pipeline.
    
    Logs are written to both console and a timestamped log file.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Return module logger
    logger = logging.getLogger("uidai_asris")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# =============================================================================
# Imports (after path setup)
# =============================================================================
try:
    from src.config import PipelineConfig, DEFAULT_CONFIG
    from src.data_pipeline import run_data_pipeline
    from src.model_train import train_champion_model, save_model
    from src.model_eval import (
        evaluate_model, 
        run_drift_checks, 
        generate_alerts,
        should_retrain,
    )
except ImportError as e:
    print(f"ERROR: Failed to import src modules: {e}")
    print("Ensure the src/ directory contains all required modules.")
    sys.exit(1)


# =============================================================================
# Main Pipeline
# =============================================================================
def main(config: PipelineConfig = None) -> Dict[str, Any]:
    """
    Run the complete training pipeline.
    
    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. Uses DEFAULT_CONFIG if not provided.
        
    Returns
    -------
    dict
        Pipeline results including model, metrics, alerts, and paths.
        
    Raises
    ------
    FileNotFoundError
        If data file is missing.
    ValueError
        If data validation fails.
    RuntimeError
        If training fails.
    """
    # Setup logging
    logger = setup_logging()
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Validate configuration
    config.validate()
    
    print("=" * 70)
    print("UIDAI ASRIS - AADHAAR ENROLMENT FORECASTING PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    results = {}
    
    try:
        # ---------------------------------------------------------------------
        # Phase 2-3: Data Pipeline
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE 2-3: DATA PIPELINE")
        print("=" * 70)
        
        logger.info("Phase 2-3: Starting data pipeline")
        
        data_result = run_data_pipeline(config)
        
        df = data_result["df"]
        X_train = data_result["X_train"]
        X_val = data_result["X_val"]
        y_train = data_result["y_train"]
        y_val = data_result["y_val"]
        numeric_features = data_result["numeric_features"]
        categorical_features = data_result["categorical_features"]
        split_idx = data_result["split_idx"]
        
        logger.info(f"Data pipeline complete: {len(X_train):,} train, {len(X_val):,} val samples")
        
        # ---------------------------------------------------------------------
        # Phase 4: Model Training
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE 4: MODEL TRAINING")
        print("=" * 70)
        
        logger.info("Phase 4: Starting model training")
        
        pipeline, train_info = train_champion_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            use_early_stopping=True,
            verbose=True,
            config=config,
        )
        
        logger.info(f"Training complete. Best iteration: {train_info.get('best_iteration', 'N/A')}")
        logger.info(f"Train MAE: {train_info['train_mae']:.2f}, Val MAE: {train_info.get('val_mae', 'N/A')}")
        
        # ---------------------------------------------------------------------
        # Phase 4.6: Save Model
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE 4.6: SAVE MODEL")
        print("=" * 70)
        
        logger.info("Phase 4.6: Saving model artifacts")
        
        # Generate version from latest training data
        last_train_month = df[config.data.date_col].iloc[split_idx - 1]
        version = last_train_month.strftime("%Y%m")
        
        model_path, metadata_path = save_model(
            model=pipeline,
            version=version,
            train_info=train_info,
            config=config,
        )
        
        results["model_path"] = model_path
        results["metadata_path"] = metadata_path
        
        logger.info(f"Model saved: {model_path}")
        
        # ---------------------------------------------------------------------
        # Phase 5.1: Validation Evaluation
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE 5.1: VALIDATION EVALUATION")
        print("=" * 70)
        
        logger.info("Phase 5.1: Evaluating on validation set")
        
        val_metrics = evaluate_model(pipeline, X_val, y_val, "Validation")
        results["val_metrics"] = val_metrics
        
        # Compare with naive baseline
        lag_col = config.alert.lag_col
        if lag_col in df.columns:
            naive_baseline = df[lag_col].iloc[split_idx:].values
            naive_mae = float(abs(y_val.values - naive_baseline).mean())
            
            # Avoid division by zero
            ss_tot = ((y_val.values - y_val.values.mean()) ** 2).sum()
            ss_res = ((y_val.values - naive_baseline) ** 2).sum()
            naive_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"\n📊 Naive Baseline (last month):")
            print(f"   MAE: {naive_mae:.2f}")
            print(f"   R²:  {naive_r2:.4f}")
            
            improvement = (naive_mae - val_metrics["mae"]) / naive_mae * 100 if naive_mae > 0 else 0
            print(f"\n🏆 XGBoost vs Naive:")
            print(f"   MAE improvement: {improvement:+.1f}%")
            print(f"   R² improvement:  {val_metrics['r2'] - naive_r2:+.4f}")
            
            results["naive_metrics"] = {"mae": naive_mae, "r2": naive_r2}
            results["improvement_pct"] = improvement
            
            logger.info(f"XGBoost MAE: {val_metrics['mae']:.2f}, Naive MAE: {naive_mae:.2f}")
            logger.info(f"MAE improvement: {improvement:+.1f}%")
        
        # ---------------------------------------------------------------------
        # Phase 5.2: Drift Detection
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE 5.2: DRIFT DETECTION")
        print("=" * 70)
        
        logger.info("Phase 5.2: Running drift detection")
        
        drift_result = run_drift_checks(
            df=df,
            verbose=True,
            config=config,
        )
        
        results["drift_result"] = drift_result
        
        logger.info(f"Drift flags: {drift_result['summary']['total_drift_flags']}")
        
        # ---------------------------------------------------------------------
        # Phase 5.4: Generate Alerts
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE 5.4: ALERT GENERATION")
        print("=" * 70)
        
        logger.info("Phase 5.4: Generating alerts")
        
        # Get validation portion of df
        df_val = df.iloc[split_idx:].copy()
        y_val_pred = pipeline.predict(X_val)
        
        alerts = generate_alerts(
            df=df_val,
            y_pred=y_val_pred,
            y_true=y_val.values,
            config=config,
        )
        
        results["alerts"] = alerts
        
        alert_counts = alerts["alert_type"].value_counts()
        print(f"\nAlert Summary:")
        for alert_type, count in alert_counts.items():
            print(f"  {alert_type}: {count}")
        
        total_alerts = (alerts["alert_type"] != "none").sum()
        print(f"\nTotal alerts: {total_alerts} ({100*total_alerts/len(alerts):.1f}%)")
        
        logger.info(f"Generated {total_alerts} alerts ({100*total_alerts/len(alerts):.1f}%)")
        
        # ---------------------------------------------------------------------
        # Retraining Recommendation
        # ---------------------------------------------------------------------
        retrain_flag, retrain_reasons = should_retrain(
            current_metrics=val_metrics,
            drift_summary=drift_result["summary"],
            config=config,
        )
        
        results["retrain_recommended"] = retrain_flag
        results["retrain_reasons"] = retrain_reasons
        
        if retrain_flag:
            print(f"\n⚠️  RETRAINING RECOMMENDED:")
            for reason in retrain_reasons:
                print(f"    - {reason}")
            logger.warning(f"Retraining recommended: {retrain_reasons}")
        
        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n✅ Model saved to: {model_path}")
        print(f"✅ Metadata saved to: {metadata_path}")
        print(f"\n📊 Final Validation Metrics:")
        print(f"   MAE:  {val_metrics['mae']:.2f}")
        print(f"   RMSE: {val_metrics['rmse']:.2f}")
        print(f"   R²:   {val_metrics['r2']:.4f}")
        
        end_time = datetime.now()
        print(f"\n⏱️  Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final metrics - MAE: {val_metrics['mae']:.2f}, R²: {val_metrics['r2']:.4f}")
        
        # Store remaining results
        results["pipeline"] = pipeline
        results["train_info"] = train_info
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the data pipeline has been run first.")
        raise
        
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        print(f"\n❌ ERROR: {e}")
        raise
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"\n❌ PIPELINE FAILED: {e}")
        raise RuntimeError(f"Training pipeline failed: {e}") from e


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception:
        sys.exit(1)
