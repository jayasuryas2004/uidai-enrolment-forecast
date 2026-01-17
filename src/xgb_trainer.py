"""
xgb_trainer.py
==============

Production-style module for training regularized XGBoost regression models
with early stopping on time-series data.

This module provides:
    1. train_xgb_with_early_stopping() - Train XGBRegressor with regularization + early stopping
    2. evaluate_model() - Compute regression metrics on any split
    3. log_training_summary() - Print formatted training results

Dependencies:
    - ctx_v3_features.py (build_ctx_v3_features, time_based_split)
    - xgboost, sklearn, pandas, numpy

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,  # L2 regularization
    "reg_alpha": 0.0,   # L1 regularization
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50


# =============================================================================
# Preprocessing: Categorical Encoding
# =============================================================================

def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Dict[str, LabelEncoder]]:
    """
    Encode categorical (object) columns using LabelEncoder.

    Fits encoders on the union of train/val/test to handle unseen categories.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    X_test : pd.DataFrame | None
        Test features (optional).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Dict[str, LabelEncoder]]
        Encoded X_train, X_val, X_test (or None), and dict of fitted encoders.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy() if X_test is not None else None

    encoders = {}
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        
        # Fit on all data to handle any category in val/test
        all_values = pd.concat([
            X_train[col].astype(str),
            X_val[col].astype(str),
        ])
        if X_test is not None:
            all_values = pd.concat([all_values, X_test[col].astype(str)])
        
        le.fit(all_values.unique())
        
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        if X_test is not None:
            X_test[col] = le.transform(X_test[col].astype(str))
        
        encoders[col] = le

    return X_train, X_val, X_test, encoders


# =============================================================================
# Part 1: Training Function
# =============================================================================

def train_xgb_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    verbose: bool = False,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Train a regularized XGBRegressor with early stopping on a validation set.

    This function trains an XGBoost model using the training set and monitors
    performance on the validation set. Training stops early if validation
    performance doesn't improve for `early_stopping_rounds` consecutive rounds.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values.
    X_val : pd.DataFrame
        Validation features (used for early stopping).
    y_val : pd.Series
        Validation target values.
    params : Dict[str, Any] | None, default None
        XGBoost hyperparameters. If None, uses DEFAULT_XGB_PARAMS.
        Any provided params will override defaults.
    early_stopping_rounds : int, default 50
        Number of rounds without improvement before stopping.
    verbose : bool, default False
        If True, print training progress.

    Returns
    -------
    Tuple[XGBRegressor, Dict[str, float]]
        - Fitted XGBRegressor model
        - Metrics dict with keys:
            - 'train_r2', 'train_mae', 'train_rmse'
            - 'val_r2', 'val_mae', 'val_rmse'
            - 'best_iteration': Number of trees in best model

    Examples
    --------
    >>> model, metrics = train_xgb_with_early_stopping(X_train, y_train, X_val, y_val)
    >>> print(f"Val R²: {metrics['val_r2']:.4f}, Val MAE: {metrics['val_mae']:.2f}")
    """
    # Merge default params with any user-provided overrides
    final_params = DEFAULT_XGB_PARAMS.copy()
    if params is not None:
        final_params.update(params)

    # Create model with early stopping
    model = XGBRegressor(
        **final_params,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
    )

    # Fit model with validation set for early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=verbose,
    )

    # Get best iteration (trees used by best model)
    best_iteration = model.best_iteration

    # Compute predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Compute metrics
    metrics = {
        # Training metrics
        "train_r2": r2_score(y_train, y_train_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        # Validation metrics
        "val_r2": r2_score(y_val, y_val_pred),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        # Model info
        "best_iteration": best_iteration,
        "n_estimators_used": best_iteration,
    }

    return model, metrics


def evaluate_model(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
) -> Dict[str, float]:
    """
    Evaluate a fitted XGBoost model on a given split.

    Parameters
    ----------
    model : XGBRegressor
        Fitted XGBoost model.
    X : pd.DataFrame
        Features for prediction.
    y : pd.Series
        True target values.
    split_name : str, default "test"
        Name prefix for metric keys (e.g., "test" -> "test_r2").

    Returns
    -------
    Dict[str, float]
        Metrics dict with keys: {split_name}_r2, {split_name}_mae, {split_name}_rmse
    """
    y_pred = model.predict(X)

    return {
        f"{split_name}_r2": r2_score(y, y_pred),
        f"{split_name}_mae": mean_absolute_error(y, y_pred),
        f"{split_name}_rmse": np.sqrt(mean_squared_error(y, y_pred)),
    }


def log_training_summary(
    metrics: Dict[str, float],
    params: Dict[str, Any] | None = None,
) -> None:
    """
    Print a formatted summary of training results.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics dict from train_xgb_with_early_stopping or evaluate_model.
    params : Dict[str, Any] | None
        Optional hyperparameters to display.
    """
    print("\n" + "=" * 60)
    print("XGBoost Training Summary")
    print("=" * 60)

    if params:
        print("\nHyperparameters:")
        for k, v in params.items():
            if k not in ["objective", "random_state", "n_jobs"]:
                print(f"  {k}: {v}")

    if "best_iteration" in metrics:
        print(f"\nEarly stopping: Best iteration = {int(metrics['best_iteration'])}")

    # Group metrics by split
    splits = set()
    for key in metrics.keys():
        if "_r2" in key or "_mae" in key or "_rmse" in key:
            split = key.rsplit("_", 1)[0]
            splits.add(split)

    for split in sorted(splits):
        print(f"\n{split.upper()} Metrics:")
        print("-" * 30)
        for metric in ["r2", "mae", "rmse"]:
            key = f"{split}_{metric}"
            if key in metrics:
                val = metrics[key]
                if metric == "r2":
                    print(f"  R²:   {val:.4f}")
                elif metric == "mae":
                    print(f"  MAE:  {val:.2f}")
                elif metric == "rmse":
                    print(f"  RMSE: {val:.2f}")

    print("\n" + "=" * 60)


# =============================================================================
# Part 2: Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the XGBoost training pipeline.

    This example:
    1. Creates synthetic time-series data
    2. Builds ctx_v3 features
    3. Creates time-based splits
    4. Trains XGBoost with early stopping
    5. Evaluates on test set
    """
    from datetime import timedelta

    # Import feature engineering helpers
    from ctx_v3_features import build_ctx_v3_features, time_based_split

    # -------------------------------------------------------------------------
    # Create synthetic example data (same pattern as ctx_v3_features.py)
    # -------------------------------------------------------------------------
    np.random.seed(42)

    n_days = 365 * 3  # 3 years of daily data
    states = ["CA", "TX", "NY"]
    segments = ["A", "B"]

    records = []
    base_date = datetime(2020, 1, 1)

    for state in states:
        for segment in segments:
            for day in range(n_days):
                date = base_date + timedelta(days=day)
                # Synthetic target with trend, seasonality, and noise
                y = (
                    100
                    + day * 0.05  # linear trend
                    + 20 * np.sin(2 * np.pi * day / 365)  # yearly seasonality
                    + 5 * np.sin(2 * np.pi * day / 7)  # weekly seasonality
                    + np.random.normal(0, 10)  # noise
                )
                records.append({
                    "date": date,
                    "state": state,
                    "segment": segment,
                    "y": max(0, y),
                })

    df = pd.DataFrame(records)
    print("Raw data shape:", df.shape)
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())

    # -------------------------------------------------------------------------
    # 1) Build ctx_v3 features
    # -------------------------------------------------------------------------
    print("\nBuilding ctx_v3 features...")
    df_ctx = build_ctx_v3_features(
        df,
        date_col="date",
        target_col="y",
        group_cols=["state", "segment"],
    )
    print("After feature engineering:", df_ctx.shape)

    # -------------------------------------------------------------------------
    # 2) Define split dates
    # -------------------------------------------------------------------------
    train_end = datetime(2021, 6, 30)
    val_end = datetime(2021, 12, 31)

    print(f"\nSplit dates:")
    print(f"  Train: <= {train_end.date()}")
    print(f"  Val:   {train_end.date()} < date <= {val_end.date()}")
    print(f"  Test:  > {val_end.date()}")

    # -------------------------------------------------------------------------
    # 3) Create time-based splits
    # -------------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        df_ctx,
        date_col="date",
        target_col="y",
        train_end=train_end,
        val_end=val_end,
    )

    print(f"\nSplit sizes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # -------------------------------------------------------------------------
    # 4) Encode categorical columns for XGBoost
    # -------------------------------------------------------------------------
    print("\nEncoding categorical columns...")
    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categoricals(
        X_train, X_val, X_test
    )
    print(f"  Encoded columns: {list(encoders.keys())}")

    # -------------------------------------------------------------------------
    # 5) Train XGBoost with early stopping
    # -------------------------------------------------------------------------
    print("\nTraining XGBoost with early stopping...")

    # Custom params (optional - these override defaults)
    custom_params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
    }

    model, metrics = train_xgb_with_early_stopping(
        X_train=X_train_enc,
        y_train=y_train,
        X_val=X_val_enc,
        y_val=y_val,
        params=custom_params,
        early_stopping_rounds=50,
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # 6) Evaluate on test set
    # -------------------------------------------------------------------------
    if len(X_test_enc) > 0:
        test_metrics = evaluate_model(model, X_test_enc, y_test, split_name="test")
        metrics.update(test_metrics)

    # -------------------------------------------------------------------------
    # 7) Print summary
    # -------------------------------------------------------------------------
    log_training_summary(metrics, params=custom_params)

    print("\n✅ XGBoost training pipeline complete!")
    print(f"   Model has {model.best_iteration} trees (early stopped from 2000)")
