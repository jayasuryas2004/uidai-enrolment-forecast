#!/usr/bin/env python
"""
phase4_v3_inference.py
======================

Production inference API for the Phase-4 v3 baseline model.

This module provides a clean, production-style API for loading and using
the frozen v3 baseline model for UIDAI district-level enrolment forecasting.

╔════════════════════════════════════════════════════════════════════════════╗
║  The v3 baseline model is the OFFICIAL model for UIDAI forecasting.        ║
║  It has been validated with leakage-safe time-series CV and is frozen.     ║
╚════════════════════════════════════════════════════════════════════════════╝

Features:
    - Loads frozen v3 baseline model and encoders
    - Builds features using the same leakage-safe pipeline as training
    - Provides predict() method for new district-month data
    - Handles missing values and unseen categories gracefully

Usage:
    from src.phase4_v3_inference import Phase4V3Model
    
    model = Phase4V3Model()
    model.load()
    
    predictions = model.predict(df_new)

Author: UIDAI Forecast Team
Date: January 2026
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from src.phase4_model_registry import PHASE4_V3_FINAL
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

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (must match training configuration)
# =============================================================================

DATE_COL = "month_date"
TARGET_COL = "total_enrolment"
GROUP_COLS = ["state", "district"]

# v3 baseline feature settings (must match freeze_phase4_v3_baseline.py)
V3_LAGS = [1, 2, 3, 6]
V3_ROLLING_WINDOWS = [3, 6]


# =============================================================================
# FEATURE BUILDING FOR INFERENCE
# =============================================================================

def build_v3_features_for_inference(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """
    Build v3 features for inference, applying frozen encoders.
    
    This function replicates the exact feature engineering used during
    training, ensuring consistency between training and inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input data with columns: state, district, month_date, total_enrolment
        (and optionally historical values for lag computation).
    encoders : dict[str, LabelEncoder]
        Pre-fitted label encoders from the frozen v3 model.
        
    Returns
    -------
    pd.DataFrame
        Feature matrix ready for model.predict().
        
    Notes
    -----
    - For proper lag/rolling features, the input DataFrame should contain
      sufficient historical data (at least max(lags) months of history).
    - If historical data is not available, lag features will be NaN and
      will be filled with -999 (the training convention).
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
        lags=V3_LAGS,
        date_col=DATE_COL,
    )
    
    # 2. Rolling features (per-district, leakage-safe with shift)
    df = add_rolling_features(
        df,
        target_col=TARGET_COL,
        group_cols=GROUP_COLS,
        windows=V3_ROLLING_WINDOWS,
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
    holiday_calendar = build_holiday_calendar(
        start=str(df[DATE_COL].min().date()),
        end=str(df[DATE_COL].max().date()),
    )
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
    
    # Drop non-feature columns
    drop_cols = [TARGET_COL, DATE_COL]
    
    # Encode categorical columns using frozen encoders
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in drop_cols]
    
    for col in cat_cols:
        le = encoders.get(col)
        if le is not None:
            # Handle unseen categories
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            # Unknown column, drop or encode with -1
            logger.warning(f"No encoder found for column '{col}', using -1")
            df[col] = -1
    
    # Drop non-feature columns
    X = df.drop(columns=drop_cols, errors="ignore")
    
    # Handle NaN values (from lag/rolling features)
    X = X.fillna(-999)
    
    return X


# =============================================================================
# PHASE-4 v3 MODEL CLASS
# =============================================================================

class Phase4V3Model:
    """
    Production inference wrapper for the Phase-4 v3 baseline model.
    
    This class provides a clean API for loading the frozen v3 baseline
    model and making predictions on new district-month data.
    
    Attributes
    ----------
    model : xgb.XGBRegressor | None
        The loaded XGBoost model (None until load() is called).
    encoders : dict[str, LabelEncoder] | None
        The label encoders for categorical features.
    params : dict | None
        The model hyperparameters.
    metrics : dict | None
        The CV evaluation metrics.
    is_loaded : bool
        Whether the model has been loaded.
        
    Examples
    --------
    >>> model = Phase4V3Model()
    >>> model.load()
    >>> 
    >>> # Prepare new data (must have same schema as training data)
    >>> df_new = pd.read_csv("new_district_month_data.csv")
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(df_new)
    >>> print(predictions)
    """
    
    def __init__(self):
        """Initialize the model wrapper."""
        self.model: xgb.XGBRegressor | None = None
        self.encoders: dict[str, LabelEncoder] | None = None
        self.params: dict | None = None
        self.metrics: dict | None = None
        self.is_loaded: bool = False
    
    def load(self) -> None:
        """
        Load the frozen v3 baseline model and encoders.
        
        Raises
        ------
        FileNotFoundError
            If the frozen model files do not exist.
        """
        # Check if files exist
        if not PHASE4_V3_FINAL.final_model.exists():
            raise FileNotFoundError(
                f"v3 baseline model not found: {PHASE4_V3_FINAL.final_model}\n"
                "Run 'python -m src.freeze_phase4_v3_baseline' first to create it."
            )
        
        if not PHASE4_V3_FINAL.final_encoders.exists():
            raise FileNotFoundError(
                f"v3 baseline encoders not found: {PHASE4_V3_FINAL.final_encoders}\n"
                "Run 'python -m src.freeze_phase4_v3_baseline' first to create it."
            )
        
        # Load model
        self.model = joblib.load(PHASE4_V3_FINAL.final_model)
        logger.info(f"Loaded model from: {PHASE4_V3_FINAL.final_model}")
        
        # Load encoders
        self.encoders = joblib.load(PHASE4_V3_FINAL.final_encoders)
        logger.info(f"Loaded encoders from: {PHASE4_V3_FINAL.final_encoders}")
        
        # Load params (optional)
        if PHASE4_V3_FINAL.final_params.exists():
            with open(PHASE4_V3_FINAL.final_params, "r") as f:
                self.params = json.load(f)
        
        # Load metrics (optional)
        if PHASE4_V3_FINAL.final_metrics.exists():
            with open(PHASE4_V3_FINAL.final_metrics, "r") as f:
                self.metrics = json.load(f)
        
        self.is_loaded = True
        logger.info("v3 baseline model loaded successfully")
    
    def predict(self, df_raw: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new district-month data.
        
        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input data with columns matching the training schema:
            - state: State name
            - district: District name
            - month_date: Month date (YYYY-MM-DD or similar)
            - total_enrolment: Historical enrolment values (for lag features)
            
        Returns
        -------
        pd.Series
            Predictions for total_enrolment, indexed like df_raw.
            
        Raises
        ------
        RuntimeError
            If the model has not been loaded yet.
            
        Notes
        -----
        For accurate predictions, include sufficient historical data
        (at least 6 months) for proper lag and rolling feature computation.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Build features
        X = build_v3_features_for_inference(df_raw, self.encoders)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df_raw.index, name="predicted_enrolment")
    
    def get_cv_metrics(self) -> dict | None:
        """
        Get the cross-validation metrics from training.
        
        Returns
        -------
        dict | None
            CV metrics dictionary, or None if not available.
        """
        if self.metrics is None:
            return None
        return self.metrics.get("cv_metrics")
    
    def get_feature_names(self) -> list[str] | None:
        """
        Get the feature names used by the model.
        
        Returns
        -------
        list[str] | None
            List of feature names, or None if not available.
        """
        if self.metrics is None:
            return None
        return self.metrics.get("feature_names")
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"Phase4V3Model(status={status})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_v3_model() -> Phase4V3Model:
    """
    Convenience function to load and return the v3 baseline model.
    
    Returns
    -------
    Phase4V3Model
        Loaded model ready for predictions.
        
    Examples
    --------
    >>> from src.phase4_v3_inference import load_v3_model
    >>> model = load_v3_model()
    >>> predictions = model.predict(df_new)
    """
    model = Phase4V3Model()
    model.load()
    return model


def predict_with_v3(df_raw: pd.DataFrame) -> pd.Series:
    """
    One-liner to make predictions with the v3 baseline model.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input data (same format as training data).
        
    Returns
    -------
    pd.Series
        Predictions for total_enrolment.
        
    Examples
    --------
    >>> from src.phase4_v3_inference import predict_with_v3
    >>> predictions = predict_with_v3(df_new)
    """
    model = load_v3_model()
    return model.predict(df_raw)


# =============================================================================
# MODULE INFO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("PHASE-4 v3 INFERENCE API")
    print("=" * 70)
    print()
    print("Frozen v3 Baseline Paths:")
    print(f"  Model:    {PHASE4_V3_FINAL.final_model}")
    print(f"  Encoders: {PHASE4_V3_FINAL.final_encoders}")
    print(f"  Metrics:  {PHASE4_V3_FINAL.final_metrics}")
    print(f"  Params:   {PHASE4_V3_FINAL.final_params}")
    print()
    
    # Check if model exists
    if PHASE4_V3_FINAL.final_model.exists():
        print("Status: ✅ v3 baseline model is available")
        print()
        
        # Load and show info
        model = Phase4V3Model()
        model.load()
        
        cv_metrics = model.get_cv_metrics()
        if cv_metrics:
            print("CV Performance:")
            print(f"  R²:  {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
            print(f"  MAE: {cv_metrics['mae_mean']:.2f} ± {cv_metrics['mae_std']:.2f}")
        
        feature_names = model.get_feature_names()
        if feature_names:
            print(f"\nFeatures: {len(feature_names)} total")
    else:
        print("Status: ❌ v3 baseline model NOT found")
        print()
        print("Run the following to create it:")
        print("  python -m src.freeze_phase4_v3_baseline")
    
    print()
    print("Usage:")
    print("  from src.phase4_v3_inference import Phase4V3Model")
    print("  model = Phase4V3Model()")
    print("  model.load()")
    print("  predictions = model.predict(df_new)")
    print("=" * 70)
