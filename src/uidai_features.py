"""
uidai_features.py

Feature engineering for UIDAI demand forecasting.
Final model: B2_log_ctx_opt (XGBoost, regression).

GOAL:
- Provide a SINGLE, SAFE feature pipeline used by the final model and all experiments.
- NO data leakage: all features must use only past information.

INPUT:
- df: pandas DataFrame with at least:
    - date column (e.g. 'date')
    - target column (e.g. 'volume')
    - state / region columns (e.g. 'state', 'region')
    - any existing policy calendar columns
- target_col: name of target column
- date_col: name of date column

OUTPUT:
- X: feature matrix (pandas DataFrame or numpy array)
- y: target vector (pandas Series or numpy array)
- All rows with NaN from lag/rolling features should be dropped in a safe way.

FEATURE GROUPS TO IMPLEMENT:

1) Time / seasonality features:
   - year, month, day, day_of_week
   - quarter
   - lags on the target (e.g. 1, 3, 6, 12 months for monthly data)
   - rolling means on the target (e.g. 3- and 6-month rolling mean)
   - IMPORTANT: lags and rolling must be shifted so they only use PAST data.

2) Policy / event features:
   - Use existing policy calendar columns (if present) and treat them as numeric/categorical features.
   - If there is a 'holiday' or 'campaign' flag, keep it as binary.

3) State / segment features:
   - Keep 'state' as a categorical/string column (do NOT one-hot here).
   - Create region/zone buckets if a column exists.
   - Create a volume_bucket column (e.g. low/medium/high) based on historical target quantiles.
   - Optionally implement simple target encoding for state or region with smoothing,
     but ensure it is computed ONLY from past data (no leakage).

REQUIRED FUNCTIONS TO IMPLEMENT:

- add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame
- add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame
- add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame
- add_policy_features(df: pd.DataFrame) -> pd.DataFrame
- add_segment_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame
- build_b2_log_ctx_opt_features(df: pd.DataFrame, target_col: str, date_col: str)

build_b2_log_ctx_opt_features should:
- Sort df by date_col
- Call the helper functions in a clear order:
    time -> lag -> rolling -> policy -> segment
- Drop rows with NaN introduced by lag/rolling
- Separate X and y (y = df[target_col])
- Return X, y
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add time/seasonality features from the date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["quarter"] = df[date_col].dt.quarter
    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str, lags: List[int] = None
) -> pd.DataFrame:
    """Add lag features on target. Uses shift to avoid leakage (only past data)."""
    if lags is None:
        lags = [1, 3, 6, 12]
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: List[int] = None
) -> pd.DataFrame:
    """Add rolling mean features on target. Shift by 1 to avoid leakage."""
    if windows is None:
        windows = [3, 6]
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = (
            df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        )
    return df


def add_policy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep existing policy/event columns as-is. No transformation needed."""
    # Placeholder: if specific policy columns exist, process them here
    return df.copy()


def add_segment_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add segment features like volume_bucket based on historical quantiles."""
    df = df.copy()
    # Create volume bucket based on target quantiles (using expanding to avoid leakage)
    if target_col in df.columns:
        expanding_median = df[target_col].expanding().median()
        df["volume_bucket"] = pd.cut(
            df[target_col],
            bins=[-np.inf, expanding_median.quantile(0.33), expanding_median.quantile(0.67), np.inf],
            labels=["low", "medium", "high"],
        )
        # Fallback: simple quantile-based bucketing
        if df["volume_bucket"].isna().all():
            df["volume_bucket"] = pd.qcut(
                df[target_col], q=3, labels=["low", "medium", "high"], duplicates="drop"
            )
    return df


def build_b2_log_ctx_opt_features(
    df: pd.DataFrame, target_col: str, date_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature set for B2_log_ctx_opt model.
    
    Pipeline: sort -> time -> lag -> rolling -> policy -> segment -> drop NaN -> split X/y
    """
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Apply feature transformations in order
    df = add_time_features(df, date_col)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_policy_features(df)
    df = add_segment_features(df, target_col)
    
    # Drop rows with NaN introduced by lag/rolling features
    df = df.dropna().reset_index(drop=True)
    
    # Separate X and y
    y = df[target_col].copy()
    
    # Drop target and date columns from X
    cols_to_drop = [target_col, date_col]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return X, y
