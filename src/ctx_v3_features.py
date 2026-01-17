"""
ctx_v3_features.py
==================

Production-style module for time-series feature engineering and time-based splitting.

This module provides:
    1. build_ctx_v3_features() - Create lag, rolling, and calendar features
    2. time_based_split() - Create train/val/test splits respecting temporal order

No model training is included here; this is purely data wrangling and feature engineering.

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Part 1: ctx_v3 Feature Engineering
# =============================================================================

def build_ctx_v3_features(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "y",
    group_cols: list[str] | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build ctx_v3 features for time-series forecasting.

    Creates lag features, rolling statistics, and calendar features from
    a time-series DataFrame. Features are computed per group to avoid
    data leakage across groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a date column and target column.
    date_col : str, default "date"
        Name of the datetime column.
    target_col : str, default "y"
        Name of the target variable column.
    group_cols : list[str] | None, default None
        Columns to group by when computing lag/rolling features.
        If None, defaults to ["state", "segment"] if they exist in df.
    drop_na : bool, default True
        If True, drop rows with NaN values introduced by lag/rolling operations.
        If False, keep NaN values (caller must handle them).

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus new feature columns:
        - y_lag_1, y_lag_7, y_lag_30: Lag features of target
        - y_roll7_mean, y_roll30_mean: Rolling mean features
        - dow: Day of week (0=Monday, 6=Sunday)
        - month: Month (1-12)
        - is_weekend: 1 if Saturday/Sunday, else 0

    Notes
    -----
    - Index is reset to ensure clean sequential indexing.
    - Rows with insufficient history for lag/rolling are dropped by default
      to avoid training on incomplete feature vectors.
    - The maximum lag is 30, so up to 30 initial rows per group may be dropped.

    Examples
    --------
    >>> df_features = build_ctx_v3_features(df, date_col="date", target_col="sales")
    >>> print(df_features.columns.tolist())
    """
    # Validate required columns exist
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Work on a copy to avoid modifying the original
    df = df.copy()

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Determine group columns
    if group_cols is None:
        # Default to ["state", "segment"] if they exist
        default_groups = ["state", "segment"]
        group_cols = [col for col in default_groups if col in df.columns]

    # If no group columns, create a dummy group for consistent processing
    if not group_cols:
        df["_dummy_group"] = 1
        group_cols = ["_dummy_group"]
        remove_dummy = True
    else:
        remove_dummy = False

    # Sort by group columns + date for correct temporal ordering
    sort_cols = group_cols + [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Calendar features (computed globally, no grouping needed)
    # -------------------------------------------------------------------------
    df["dow"] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)

    # -------------------------------------------------------------------------
    # Lag and rolling features (computed per group)
    # -------------------------------------------------------------------------
    lag_periods = [1, 7, 30]
    roll_windows = [7, 30]

    # Lag features (per group to avoid cross-group leakage)
    for lag in lag_periods:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    # Rolling mean features (shifted by 1 to prevent leakage)
    for window in roll_windows:
        # Compute rolling mean on the shifted series
        df[f"{target_col}_roll{window}_mean"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=window).mean())
        )

    # -------------------------------------------------------------------------
    # Handle NaN values from lag/rolling operations
    # -------------------------------------------------------------------------
    if drop_na:
        # Drop rows that have NaN in any of the new feature columns
        feature_cols = (
            [f"{target_col}_lag_{lag}" for lag in lag_periods]
            + [f"{target_col}_roll{window}_mean" for window in roll_windows]
        )
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Remove dummy group column if we added it
    if remove_dummy:
        df = df.drop(columns=["_dummy_group"])

    return df


# =============================================================================
# Part 2: Time-Based Splitting
# =============================================================================

def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    train_end: datetime,
    val_end: datetime,
    feature_cols: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split DataFrame into train/validation/test sets based on temporal boundaries.

    This function performs a strict time-based split with no shuffling,
    ensuring that the model is trained on past data and evaluated on future data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and target.
    date_col : str
        Name of the datetime column used for splitting.
    target_col : str
        Name of the target variable column.
    train_end : datetime
        End date for training set (inclusive). Rows with date <= train_end go to train.
    val_end : datetime
        End date for validation set (inclusive). Rows with train_end < date <= val_end
        go to validation. Rows with date > val_end go to test.
    feature_cols : list[str] | None, default None
        Specific feature columns to include in X. If None, all columns except
        date_col and target_col are used as features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        (X_train, y_train, X_val, y_val, X_test, y_test)

    Raises
    ------
    ValueError
        If date_col or target_col not in DataFrame.
        If any split results in zero rows.

    Examples
    --------
    >>> from datetime import datetime
    >>> X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
    ...     df, "date", "y", datetime(2021, 12, 31), datetime(2022, 12, 31)
    ... )
    """
    # Validate required columns
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Work on a copy
    df = df.copy()

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Convert boundaries to pandas Timestamp for consistent comparison
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    # Sort by date (no shuffling!)
    df = df.sort_values(date_col).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Create temporal splits
    # -------------------------------------------------------------------------
    train_mask = df[date_col] <= train_end
    val_mask = (df[date_col] > train_end) & (df[date_col] <= val_end)
    test_mask = df[date_col] > val_end

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Validate splits are non-empty
    if len(train_df) == 0:
        raise ValueError(f"Training set is empty. Check train_end={train_end}")
    if len(val_df) == 0:
        raise ValueError(f"Validation set is empty. Check val_end={val_end}")
    if len(test_df) == 0:
        raise ValueError(f"Test set is empty. All data is before val_end={val_end}")

    # -------------------------------------------------------------------------
    # Separate features and target
    # -------------------------------------------------------------------------
    if feature_cols is None:
        # Use all columns except date and target as features
        feature_cols = [
            col for col in df.columns if col not in [date_col, target_col]
        ]

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)

    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df[target_col].reset_index(drop=True)

    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# Part 3: Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the ctx_v3 feature pipeline.
    
    This block creates synthetic data to show how the functions work.
    In production, `df` would be loaded from your data source.
    """
    from datetime import datetime, timedelta

    # -------------------------------------------------------------------------
    # Create synthetic example data
    # -------------------------------------------------------------------------
    np.random.seed(42)
    
    n_days = 365 * 3  # 3 years of data
    states = ["CA", "TX", "NY"]
    segments = ["A", "B"]
    
    records = []
    base_date = datetime(2020, 1, 1)
    
    for state in states:
        for segment in segments:
            for day in range(n_days):
                date = base_date + timedelta(days=day)
                # Synthetic target with trend and seasonality
                y = (
                    100
                    + day * 0.05  # trend
                    + 20 * np.sin(2 * np.pi * day / 365)  # yearly seasonality
                    + 5 * np.sin(2 * np.pi * day / 7)  # weekly seasonality
                    + np.random.normal(0, 10)  # noise
                )
                records.append({
                    "date": date,
                    "state": state,
                    "segment": segment,
                    "y": max(0, y),  # ensure non-negative
                })
    
    df = pd.DataFrame(records)
    print("Raw data shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())

    # -------------------------------------------------------------------------
    # Build ctx_v3 features
    # -------------------------------------------------------------------------
    df_ctx = build_ctx_v3_features(
        df,
        date_col="date",
        target_col="y",
        group_cols=["state", "segment"],
    )
    print("\nAfter feature engineering:")
    print("Shape:", df_ctx.shape)
    print("New columns:", [c for c in df_ctx.columns if c not in df.columns])

    # -------------------------------------------------------------------------
    # Create time-based splits
    # -------------------------------------------------------------------------
    train_end = datetime(2021, 6, 30)
    val_end = datetime(2021, 12, 31)

    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        df_ctx,
        date_col="date",
        target_col="y",
        train_end=train_end,
        val_end=val_end,
    )

    print("\n" + "=" * 50)
    print("Time-Based Split Results")
    print("=" * 50)
    print(f"Train end: {train_end.date()}")
    print(f"Val end:   {val_end.date()}")
    print()
    print("Shapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print()
    print("Feature columns:", X_train.columns.tolist()[:5], "...")
    print("\n✅ ctx_v3 feature pipeline complete!")
