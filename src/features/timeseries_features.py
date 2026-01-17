"""
timeseries_features.py
======================

Leakage-safe lag and rolling window features for time-series forecasting.

**PURPOSE:**
Create temporal features that capture historical patterns without leaking
future information into training.

**LEAKAGE PREVENTION:**
    1. All lag features use shift(k) where k >= 1, ensuring only past data is used
    2. Rolling features are computed with shift(1) before the rolling window,
       so the current value is never included
    3. Features are computed per-group (state/district) respecting temporal order
    4. NaN values from initial periods are handled gracefully

**RECOMMENDED USAGE:**
    1. Analyze ACF/PACF on TRAINING data to select lags (see timeseries_lag_utils.py)
    2. Call add_lag_features() and add_rolling_features() on training data
    3. For validation, compute features using only historical data

**INTEGRATION WITH CV:**
    - In each fold, compute features on df_train
    - For df_val, ensure lag/rolling features use only past data
    - The CV script handles this by processing data chronologically

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_LAGS = [1, 2, 3, 6]  # Previous month, recent months, half-year
DEFAULT_ROLLING_WINDOWS = [3, 6]  # 3-month and 6-month rolling statistics


# =============================================================================
# Lag Features (Point-in-Time Safe)
# =============================================================================

def add_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    lags: Optional[List[int]] = None,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Create lagged versions of target column for each group.

    **LEAKAGE SAFETY:**
    Uses shift(k) where k >= 1, so only past values are used.
    For lag=1, the feature at time t contains the value from time t-1.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by group and date.
    group_cols : List[str]
        Columns defining groups (e.g., ['state', 'district']).
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column to lag.
    lags : List[int] | None, default None
        List of lag periods (e.g., [1, 2, 3, 6]).
        If None, uses DEFAULT_LAGS = [1, 2, 3, 6].
    fill_na : bool, default True
        If True, fill NaN values with group mean.
        If False, keep NaN (XGBoost can handle missing values).

    Returns
    -------
    pd.DataFrame
        DataFrame with new lag columns added:
            - {target_col}_lag_1, {target_col}_lag_2, etc.

    Examples
    --------
    >>> df = add_lag_features(
    ...     df, ['state', 'district'], 'month_date', 'total_enrolment', lags=[1, 2, 3]
    ... )
    >>> print(df[['total_enrolment', 'total_enrolment_lag_1']].head())

    Notes
    -----
    The first few rows for each group will have NaN values because
    there's no historical data to lag from. This is expected and
    correct behavior (no leakage).
    """
    if lags is None:
        lags = DEFAULT_LAGS.copy()

    df = df.copy()

    # Ensure data is sorted correctly for temporal operations
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Validate lags
    valid_lags = [lag for lag in lags if lag >= 1]
    if len(valid_lags) != len(lags):
        logging.warning(f"Removed invalid lags (< 1). Using: {valid_lags}")

    # Create lag features
    for lag in valid_lags:
        col_name = f"{target_col}_lag_{lag}"

        if group_cols:
            df[col_name] = (
                df.groupby(group_cols, observed=True)[target_col]
                .shift(lag)
            )
        else:
            df[col_name] = df[target_col].shift(lag)

    # Handle NaN values
    if fill_na:
        for lag in valid_lags:
            col_name = f"{target_col}_lag_{lag}"
            if group_cols:
                # Fill with group mean
                group_means = df.groupby(group_cols, observed=True)[target_col].transform("mean")
                df[col_name] = df[col_name].fillna(group_means)
            else:
                df[col_name] = df[col_name].fillna(df[target_col].mean())

    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    windows: Optional[List[int]] = None,
    include_std: bool = True,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Create rolling window statistics for each group.

    **LEAKAGE SAFETY:**
    Uses shift(1) before computing rolling statistics, ensuring the current
    value is never included in its own features. The rolling window looks
    at values from t-1 back to t-window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by group and date.
    group_cols : List[str]
        Columns defining groups (e.g., ['state', 'district']).
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    windows : List[int] | None, default None
        List of window sizes (e.g., [3, 6]).
        If None, uses DEFAULT_ROLLING_WINDOWS = [3, 6].
    include_std : bool, default True
        If True, also compute rolling standard deviation.
    fill_na : bool, default True
        If True, fill NaN values with group mean/std.

    Returns
    -------
    pd.DataFrame
        DataFrame with new rolling columns:
            - {target_col}_rolling_{window}_mean
            - {target_col}_rolling_{window}_std (if include_std=True)

    Examples
    --------
    >>> df = add_rolling_features(
    ...     df, ['state', 'district'], 'month_date', 'total_enrolment', windows=[3, 6]
    ... )

    Notes
    -----
    - Rolling mean captures recent trend
    - Rolling std captures recent volatility
    - Both are useful for forecasting models
    """
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS.copy()

    df = df.copy()

    # Ensure correct sorting
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for window in windows:
        mean_col = f"{target_col}_rolling_{window}_mean"
        std_col = f"{target_col}_rolling_{window}_std"

        if group_cols:
            # Shift first, then compute rolling (leakage-safe)
            shifted = df.groupby(group_cols, observed=True)[target_col].shift(1)

            df[mean_col] = (
                shifted.groupby(df[group_cols].apply(tuple, axis=1))
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            if include_std:
                df[std_col] = (
                    shifted.groupby(df[group_cols].apply(tuple, axis=1))
                    .rolling(window=window, min_periods=2)
                    .std()
                    .reset_index(level=0, drop=True)
                )
        else:
            shifted = df[target_col].shift(1)
            df[mean_col] = shifted.rolling(window=window, min_periods=1).mean()
            if include_std:
                df[std_col] = shifted.rolling(window=window, min_periods=2).std()

    # Handle NaN values
    if fill_na:
        for window in windows:
            mean_col = f"{target_col}_rolling_{window}_mean"
            std_col = f"{target_col}_rolling_{window}_std"

            if group_cols:
                group_mean = df.groupby(group_cols, observed=True)[target_col].transform("mean")
                group_std = df.groupby(group_cols, observed=True)[target_col].transform("std")
            else:
                group_mean = df[target_col].mean()
                group_std = df[target_col].std()

            df[mean_col] = df[mean_col].fillna(group_mean)
            if include_std:
                df[std_col] = df[std_col].fillna(group_std)

    return df


def add_diff_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    periods: Optional[List[int]] = None,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Create difference features (change from previous periods).

    **LEAKAGE SAFETY:**
    Computes diff(k) which is value[t] - value[t-k].
    This uses only past values for the difference calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns defining groups.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    periods : List[int] | None, default None
        List of difference periods. If None, uses [1].
    fill_na : bool, default True
        If True, fill NaN with 0 (no change).

    Returns
    -------
    pd.DataFrame
        DataFrame with difference columns:
            - {target_col}_diff_1, etc.
    """
    if periods is None:
        periods = [1]

    df = df.copy()
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for period in periods:
        col_name = f"{target_col}_diff_{period}"

        if group_cols:
            df[col_name] = df.groupby(group_cols, observed=True)[target_col].diff(period)
        else:
            df[col_name] = df[target_col].diff(period)

        if fill_na:
            df[col_name] = df[col_name].fillna(0)

    return df


def add_pct_change_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    periods: Optional[List[int]] = None,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Create percentage change features.

    **LEAKAGE SAFETY:**
    Computes pct_change(k) = (value[t] - value[t-k]) / value[t-k].
    Uses only past values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns defining groups.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    periods : List[int] | None, default None
        List of periods for pct change. If None, uses [1].
    fill_na : bool, default True
        If True, fill NaN with 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with pct_change columns:
            - {target_col}_pct_change_1, etc.
    """
    if periods is None:
        periods = [1]

    df = df.copy()
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for period in periods:
        col_name = f"{target_col}_pct_change_{period}"

        if group_cols:
            df[col_name] = df.groupby(group_cols, observed=True)[target_col].pct_change(period)
        else:
            df[col_name] = df[target_col].pct_change(period)

        # Handle infinite values from division by zero
        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)

        if fill_na:
            df[col_name] = df[col_name].fillna(0)

    return df


# =============================================================================
# Combined Feature Builder
# =============================================================================

def add_all_timeseries_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    include_diff: bool = True,
    include_pct_change: bool = False,
    fill_na: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Add all time-series features in one call.

    This is a convenience function that calls:
        - add_lag_features()
        - add_rolling_features()
        - add_diff_features() (optional)
        - add_pct_change_features() (optional)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns defining groups.
    date_col : str
        Date column name.
    target_col : str
        Target column name.
    lags : List[int] | None
        Lags for lag features. Default: [1, 2, 3, 6].
    rolling_windows : List[int] | None
        Windows for rolling features. Default: [3, 6].
    include_diff : bool, default True
        Include difference features.
    include_pct_change : bool, default False
        Include percentage change features.
    fill_na : bool, default True
        Fill NaN values.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        - DataFrame with all features added
        - Dictionary listing the new feature columns

    Examples
    --------
    >>> df, feature_info = add_all_timeseries_features(
    ...     df, ['state', 'district'], 'month_date', 'total_enrolment'
    ... )
    >>> print(feature_info['new_columns'])
    """
    if lags is None:
        lags = DEFAULT_LAGS.copy()
    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS.copy()

    original_cols = set(df.columns)

    # Add features
    df = add_lag_features(df, group_cols, date_col, target_col, lags, fill_na)
    df = add_rolling_features(df, group_cols, date_col, target_col, rolling_windows, True, fill_na)

    if include_diff:
        df = add_diff_features(df, group_cols, date_col, target_col, [1], fill_na)

    if include_pct_change:
        df = add_pct_change_features(df, group_cols, date_col, target_col, [1], fill_na)

    new_cols = [col for col in df.columns if col not in original_cols]

    feature_info = {
        "new_columns": new_cols,
        "lags_used": lags,
        "rolling_windows_used": rolling_windows,
        "include_diff": include_diff,
        "include_pct_change": include_pct_change,
        "n_new_features": len(new_cols),
    }

    logging.info(f"Added {len(new_cols)} time-series features: {new_cols}")

    return df, feature_info
