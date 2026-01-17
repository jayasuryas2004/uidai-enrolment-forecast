# UIDAI ASRIS - Aadhaar Service & Risk Intelligence Suite
# Production-ready ML pipeline modules

"""
UIDAI ASRIS - Aadhaar Service & Risk Intelligence Suite

This package provides production-ready ML pipeline components for
the UIDAI Aadhaar enrolment forecasting project.

Modules:
    config: Centralized configuration (thresholds, paths, hyperparameters)
    data_pipeline: Data loading, cleaning, feature engineering, train/val split
    model_train: XGBoost pipeline building, training with early stopping, model persistence
    model_eval: Evaluation metrics, drift detection, alert generation, retraining decisions

Usage:
    # Import configuration
    from src.config import PipelineConfig, DEFAULT_CONFIG
    
    # Run data pipeline
    from src.data_pipeline import run_data_pipeline
    result = run_data_pipeline()
    
    # Train model
    from src.model_train import train_champion_model, save_model
    pipeline, train_info = train_champion_model(X_train, y_train, X_val, y_val)
    
    # Evaluate and monitor
    from src.model_eval import evaluate_model, run_drift_checks, generate_alerts
    metrics = evaluate_model(pipeline, X_val, y_val)

Version: 1.1.0
Author: UIDAI ASRIS Team
"""

__version__ = "1.1.0"
__author__ = "UIDAI ASRIS Team"

# Expose key classes and functions at package level
from .config import PipelineConfig, DEFAULT_CONFIG
from .data_pipeline import run_data_pipeline, load_raw_data, clean_data, build_features
from .model_train import train_champion_model, save_model, load_model, build_champion_pipeline
from .model_eval import evaluate_model, run_drift_checks, generate_alerts, should_retrain
