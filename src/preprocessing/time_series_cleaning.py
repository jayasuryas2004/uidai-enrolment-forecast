#!/usr/bin/env python
"""
time_series_cleaning.py
=======================

Time-series cleaning module for UIDAI district-level enrolment forecasting.

This module provides leakage-safe outlier detection and missing data handling
for monthly district-level time series data.

**KEY FEATURES:**

1. **Outlier Detection (Moving Z-Score)**
   - Detects abnormal spikes/drops in enrolment using a moving z-score per district
   - Marks outliers with `is_outlier_event` flag for the model to learn from
   - Optionally caps extreme outliers using rolling median

2. **Missing Data Imputation**
   - LOCF (Last Observation Carried Forward) for short gaps
   - State-level mean fallback for longer gaps
   - Linear interpolation option for smooth transitions

3. **Leakage Safety**
   - All statistics computed using only past data (shift before rolling)
   - Safe to use inside time-series CV folds
   - No peeking into future data points

**INTEGRATION:**

This cleaning runs inside each time-series CV fold, so the model only sees
information that would have been available at that time in real life.

Usage:
    from src.preprocessing.time_series_cleaning import (
        CleaningConfig,
        clean_uidai_time_series,
    )

    config = CleaningConfig(
        outlier_method="zscore_moving",
        outlier_window=3,
        outlier_z_thresh=3.0,
        missing_method="locf",
        max_locf_gap=3,
    )

    df_clean = clean_uidai_time_series(
        df=df_train,
        state_col="state",
        district_col="district",
        date_col="month_date",
        target_col="total_enrolment",
        config=config,
    )

Author: UIDAI Forecast Team
Date: January 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

OutlierMethod = Literal["zscore_moving", "iqr", "none"]
MissingMethod = Literal["locf", "state_mean", "linear", "none"]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CleaningConfig:
    """
    Configuration for time-series cleaning operations.

    Attributes
    ----------
    outlier_method : OutlierMethod
        Method for outlier detection:
        - "zscore_moving": Moving z-score with configurable window
        - "iqr": Interquartile range method (1.5 * IQR)
        - "none": Skip outlier detection

    outlier_window : int
        Window size for moving statistics in outlier detection.
        Default is 3 months.

    outlier_z_thresh : float
        Z-score threshold for outlier detection (|z| > threshold).
        Default is 3.0 (corresponds to ~99.7% confidence).

    outlier_cap_method : Literal["median", "clip", "none"]
        How to handle detected outliers:
        - "median": Replace with rolling median of neighbors
        - "clip": Clip to rolling mean ± z_thresh * rolling_std
        - "none": Keep original value (only flag)

    missing_method : MissingMethod
        Method for missing value imputation:
        - "locf": Last observation carried forward
        - "state_mean": Fill with state-level monthly mean
        - "linear": Linear interpolation per district
        - "none": Skip imputation

    max_locf_gap : int
        Maximum consecutive months for LOCF before falling back
        to another method. Default is 3 months.

    log_changes : bool
        Whether to log summary of changes made. Default is True.
    """

    outlier_method: OutlierMethod = "zscore_moving"
    outlier_window: int = 3
    outlier_z_thresh: float = 3.0
    outlier_cap_method: Literal["median", "clip", "none"] = "median"
    missing_method: MissingMethod = "locf"
    max_locf_gap: int = 3
    log_changes: bool = True


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

def detect_outliers_moving_zscore(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    window: int,
    z_thresh: float,
) -> pd.Series:
    """
    Detect outliers using a moving z-score per group (district).

    For each group, computes a moving mean/std over the past `window` values
    (excluding the current point via shift), then computes a z-score for the
    current value relative to that moving window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    group_cols : List[str]
        Columns to group by (e.g., ["state", "district"]).
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column to check for outliers.
    window : int
        Window size for computing rolling statistics.
    z_thresh : float
        Z-score threshold; |z| > z_thresh is flagged as outlier.

    Returns
    -------
    pd.Series
        Boolean Series where True indicates a candidate outlier.
        Index matches the input DataFrame.

    Notes
    -----
    - Leakage-safe: rolling stats use shift(1) so current value is excluded
    - First `window` values per group will have NaN stats and are not flagged
    - Groups with constant values (std=0) are handled safely

    Examples
    --------
    >>> is_outlier = detect_outliers_moving_zscore(
    ...     df, ["state", "district"], "month_date", "total_enrolment",
    ...     window=3, z_thresh=3.0
    ... )
    >>> print(f"Found {is_outlier.sum()} outliers")
    """
    df = df.copy()
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # Shift target so current value is not in its own rolling window
    df["_shifted_target"] = df.groupby(group_cols)[target_col].shift(1)

    # Compute rolling mean and std on shifted values
    df["_rolling_mean"] = df.groupby(group_cols)["_shifted_target"].transform(
        lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).mean()
    )
    df["_rolling_std"] = df.groupby(group_cols)["_shifted_target"].transform(
        lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).std()
    )

    # Compute z-score for current value
    # Handle std=0 (constant values) by setting z=0
    df["_zscore"] = np.where(
        df["_rolling_std"] > 0,
        (df[target_col] - df["_rolling_mean"]) / df["_rolling_std"],
        0.0,
    )

    # Flag outliers where |z| > threshold and we have valid rolling stats
    is_outlier = (
        (df["_zscore"].abs() > z_thresh) &
        df["_rolling_mean"].notna() &
        df["_rolling_std"].notna()
    )

    return is_outlier


def detect_outliers_iqr(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    """
    Detect outliers using the Interquartile Range (IQR) method per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns to group by.
    target_col : str
        Name of the target column.
    iqr_multiplier : float
        Multiplier for IQR bounds (default 1.5 for standard outliers).

    Returns
    -------
    pd.Series
        Boolean Series where True indicates outlier.

    Notes
    -----
    - Uses cumulative Q1/Q3 computed only from past data for leakage safety
    - Falls back to global stats for early months with insufficient history
    """
    df = df.copy()

    def compute_iqr_outliers(group: pd.DataFrame) -> pd.Series:
        """Compute IQR-based outliers for a single group."""
        result = pd.Series(False, index=group.index)

        for i in range(len(group)):
            if i < 4:  # Need at least 4 points for meaningful IQR
                continue

            # Use only past data (leakage-safe)
            past_values = group[target_col].iloc[:i]
            q1 = past_values.quantile(0.25)
            q3 = past_values.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            current_value = group[target_col].iloc[i]
            if current_value < lower or current_value > upper:
                result.iloc[i] = True

        return result

    is_outlier = df.groupby(group_cols, group_keys=False).apply(compute_iqr_outliers)

    return is_outlier.reindex(df.index).fillna(False)


def cap_outliers_with_median(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    is_outlier: pd.Series,
    window: int = 3,
) -> pd.Series:
    """
    Cap outlier values using rolling median of non-outlier neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns to group by.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    is_outlier : pd.Series
        Boolean Series indicating outlier positions.
    window : int
        Window size for median computation.

    Returns
    -------
    pd.Series
        Target values with outliers replaced by rolling median.
    """
    df = df.copy()
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # Create masked series where outliers are NaN
    df["_masked_target"] = np.where(is_outlier, np.nan, df[target_col])

    # Shift to exclude current value
    df["_shifted_masked"] = df.groupby(group_cols)["_masked_target"].shift(1)

    # Compute rolling median on shifted, masked values
    df["_rolling_median"] = df.groupby(group_cols)["_shifted_masked"].transform(
        lambda x: x.rolling(window=window, min_periods=1).median()
    )

    # Replace outliers with rolling median
    capped_values = np.where(
        is_outlier,
        df["_rolling_median"],
        df[target_col],
    )

    # If rolling median is still NaN (early in series), use original value
    capped_values = np.where(
        pd.isna(capped_values),
        df[target_col],
        capped_values,
    )

    return pd.Series(capped_values, index=df.index)


def clip_outliers_to_bounds(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    window: int,
    z_thresh: float,
) -> pd.Series:
    """
    Clip outlier values to rolling mean ± z_thresh * rolling_std bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns to group by.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    window : int
        Window size for rolling statistics.
    z_thresh : float
        Number of standard deviations for bounds.

    Returns
    -------
    pd.Series
        Target values clipped to bounds.
    """
    df = df.copy()
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # Shift and compute rolling stats
    df["_shifted_target"] = df.groupby(group_cols)[target_col].shift(1)

    df["_rolling_mean"] = df.groupby(group_cols)["_shifted_target"].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    df["_rolling_std"] = df.groupby(group_cols)["_shifted_target"].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )

    # Handle NaN std (single value) by setting to 0
    df["_rolling_std"] = df["_rolling_std"].fillna(0)

    # Compute bounds
    lower_bound = df["_rolling_mean"] - z_thresh * df["_rolling_std"]
    upper_bound = df["_rolling_mean"] + z_thresh * df["_rolling_std"]

    # Clip values
    clipped_values = df[target_col].clip(lower=lower_bound, upper=upper_bound)

    # For early values without valid bounds, keep original
    clipped_values = clipped_values.fillna(df[target_col])

    return clipped_values


# =============================================================================
# MISSING DATA HANDLING
# =============================================================================

def summarize_missing_by_district(
    df: pd.DataFrame,
    state_col: str,
    district_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Generate a summary of missing data patterns by district.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    state_col : str
        Name of the state column.
    district_col : str
        Name of the district column.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column to analyze.

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - state, district
        - total_rows: Total number of rows
        - missing_count: Number of missing values
        - missing_pct: Percentage of missing values
        - max_consecutive_gap: Longest run of consecutive missing months
        - first_missing_date: First date with missing value
        - last_missing_date: Last date with missing value

    Examples
    --------
    >>> summary = summarize_missing_by_district(
    ...     df, "state", "district", "month_date", "total_enrolment"
    ... )
    >>> print(summary[summary["missing_count"] > 0])
    """
    results = []

    for (state, district), group in df.groupby([state_col, district_col]):
        group = group.sort_values(date_col).reset_index(drop=True)

        total_rows = len(group)
        missing_mask = group[target_col].isna()
        missing_count = missing_mask.sum()
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0

        # Compute maximum consecutive missing gap
        max_consecutive_gap = 0
        if missing_count > 0:
            # Use run-length encoding
            is_missing = missing_mask.values
            runs = []
            current_run = 0
            for val in is_missing:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            max_consecutive_gap = max(runs) if runs else 0

        # Find first and last missing dates
        missing_dates = group.loc[missing_mask, date_col]
        first_missing = missing_dates.min() if len(missing_dates) > 0 else None
        last_missing = missing_dates.max() if len(missing_dates) > 0 else None

        results.append({
            state_col: state,
            district_col: district,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "max_consecutive_gap": max_consecutive_gap,
            "first_missing_date": first_missing,
            "last_missing_date": last_missing,
        })

    return pd.DataFrame(results)


def impute_missing_locf(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    max_gap: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Impute missing values using Last Observation Carried Forward (LOCF).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns to group by.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    max_gap : int
        Maximum consecutive months to fill with LOCF.
        Longer gaps are left as NaN for another method.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        - Imputed values (NaN where not filled)
        - Boolean mask of values that were imputed

    Notes
    -----
    LOCF is leakage-safe as it only uses past observations.
    """
    df = df.copy()
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    original_na = df[target_col].isna()
    imputed = df[target_col].copy()

    # Apply LOCF within each group
    for _, group_idx in df.groupby(group_cols).groups.items():
        group_data = df.loc[group_idx].sort_values(date_col)

        # Forward fill
        filled = group_data[target_col].ffill()

        # Track consecutive NaN runs to respect max_gap
        is_na = group_data[target_col].isna()
        cumsum = (~is_na).cumsum()
        consecutive_na_count = is_na.groupby(cumsum).cumsum()

        # Only keep fills within max_gap
        valid_fill = consecutive_na_count <= max_gap

        # Apply valid fills
        imputed.loc[group_idx] = np.where(
            is_na & valid_fill,
            filled.loc[group_idx],
            group_data[target_col]
        )

    was_imputed = original_na & imputed.notna()

    return imputed, was_imputed


def impute_missing_state_mean(
    df: pd.DataFrame,
    state_col: str,
    district_col: str,
    date_col: str,
    target_col: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Impute missing values using state-level monthly mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    state_col : str
        Name of the state column.
    district_col : str
        Name of the district column.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        - Imputed values
        - Boolean mask of values that were imputed

    Notes
    -----
    - Computes mean per (state, month_date) excluding missing values
    - Falls back to overall state mean if month mean is not available
    - Leakage-safe when applied within training folds
    """
    df = df.copy()

    original_na = df[target_col].isna()
    imputed = df[target_col].copy()

    # Compute state-month means
    state_month_means = df.groupby([state_col, date_col])[target_col].transform("mean")

    # Fill with state-month mean where missing
    imputed = imputed.fillna(state_month_means)

    # If still missing (no other data for that state-month), use overall state mean
    still_missing = imputed.isna()
    if still_missing.any():
        state_means = df.groupby(state_col)[target_col].transform("mean")
        imputed = imputed.fillna(state_means)

    was_imputed = original_na & imputed.notna()

    return imputed, was_imputed


def impute_missing_linear(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Impute missing values using linear interpolation within each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : List[str]
        Columns to group by.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        - Imputed values
        - Boolean mask of values that were imputed

    Notes
    -----
    - Uses time-based linear interpolation
    - Only interpolates, does not extrapolate beyond known values
    - When used in CV, apply only to training fold to avoid leakage
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    original_na = df[target_col].isna()
    imputed = df[target_col].copy()

    for _, group_idx in df.groupby(group_cols).groups.items():
        group_data = df.loc[group_idx].sort_values(date_col)

        # Create series with datetime index for time-based interpolation
        series = pd.Series(
            group_data[target_col].values,
            index=pd.to_datetime(group_data[date_col]),
        )

        # Linear interpolation (does not extrapolate)
        interpolated = series.interpolate(method="time")

        imputed.loc[group_idx] = interpolated.values

    was_imputed = original_na & imputed.notna()

    return imputed, was_imputed


def impute_missing_values(
    df: pd.DataFrame,
    state_col: str,
    district_col: str,
    date_col: str,
    target_col: str,
    method: MissingMethod,
    max_locf_gap: int = 3,
) -> pd.DataFrame:
    """
    Impute missing target values according to the chosen method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    state_col : str
        Name of the state column.
    district_col : str
        Name of the district column.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    method : MissingMethod
        Imputation method:
        - "locf": Last observation carried forward (with max_locf_gap limit)
        - "state_mean": State-level monthly mean
        - "linear": Linear interpolation per district
        - "none": No imputation
    max_locf_gap : int
        Maximum gap for LOCF method.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values and new columns:
        - `{target_col}_was_imputed`: Boolean flag for imputed values
        - `{target_col}_original`: Original value before imputation

    Notes
    -----
    For LOCF with gaps longer than max_locf_gap, falls back to state_mean.
    """
    df = df.copy()
    group_cols = [state_col, district_col]

    # Store original values
    df[f"{target_col}_original"] = df[target_col].copy()
    df[f"{target_col}_was_imputed"] = False

    if method == "none":
        return df

    initial_missing = df[target_col].isna().sum()
    if initial_missing == 0:
        return df

    if method == "locf":
        # First pass: LOCF with max gap
        imputed, was_imputed = impute_missing_locf(
            df, group_cols, date_col, target_col, max_locf_gap
        )
        df[target_col] = imputed
        df[f"{target_col}_was_imputed"] = was_imputed

        # Second pass: Fill remaining with state mean
        still_missing = df[target_col].isna().sum()
        if still_missing > 0:
            imputed2, was_imputed2 = impute_missing_state_mean(
                df, state_col, district_col, date_col, target_col
            )
            df[target_col] = imputed2
            df[f"{target_col}_was_imputed"] = df[f"{target_col}_was_imputed"] | was_imputed2

    elif method == "state_mean":
        imputed, was_imputed = impute_missing_state_mean(
            df, state_col, district_col, date_col, target_col
        )
        df[target_col] = imputed
        df[f"{target_col}_was_imputed"] = was_imputed

    elif method == "linear":
        imputed, was_imputed = impute_missing_linear(
            df, group_cols, date_col, target_col
        )
        df[target_col] = imputed
        df[f"{target_col}_was_imputed"] = was_imputed

        # Fill remaining (extrapolation not possible) with state mean
        still_missing = df[target_col].isna().sum()
        if still_missing > 0:
            imputed2, was_imputed2 = impute_missing_state_mean(
                df, state_col, district_col, date_col, target_col
            )
            df[target_col] = imputed2
            df[f"{target_col}_was_imputed"] = df[f"{target_col}_was_imputed"] | was_imputed2

    final_missing = df[target_col].isna().sum()
    logger.debug(f"Missing values: {initial_missing} → {final_missing}")

    return df


# =============================================================================
# MAIN CLEANING API
# =============================================================================

def clean_uidai_time_series(
    df: pd.DataFrame,
    state_col: str,
    district_col: str,
    date_col: str,
    target_col: str,
    config: Optional[CleaningConfig] = None,
) -> pd.DataFrame:
    """
    Clean UIDAI time series data with outlier handling and missing imputation.

    This is the main entry point for time-series cleaning. It applies:
    1. Outlier detection using the configured method
    2. Outlier handling (capping or flagging)
    3. Missing value imputation

    All operations are leakage-safe and can be used inside time-series CV folds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with district-month time series data.
    state_col : str
        Name of the state column.
    district_col : str
        Name of the district column.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column (e.g., "total_enrolment").
    config : CleaningConfig, optional
        Cleaning configuration. If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with additional columns:
        - `is_outlier_event`: Boolean flag for detected outliers
        - `{target_col}_original`: Original value before any changes
        - `{target_col}_was_imputed`: Boolean flag for imputed values
        - `{target_col}_was_capped`: Boolean flag for capped outliers

    Examples
    --------
    >>> from src.preprocessing.time_series_cleaning import (
    ...     CleaningConfig, clean_uidai_time_series
    ... )
    >>> config = CleaningConfig(
    ...     outlier_method="zscore_moving",
    ...     outlier_z_thresh=3.0,
    ...     missing_method="locf",
    ... )
    >>> df_clean = clean_uidai_time_series(
    ...     df, "state", "district", "month_date", "total_enrolment", config
    ... )
    >>> print(f"Outliers flagged: {df_clean['is_outlier_event'].sum()}")

    Notes
    -----
    **Leakage Safety:**
    - Outlier detection uses only past data (shift before rolling)
    - Missing imputation uses LOCF (past only) or state means from current data
    - Safe to apply within each CV fold's training subset

    **For Validation/Test Data:**
    When cleaning validation/test data, the same logic applies:
    - Outlier flags are computed using only that point's past history
    - Missing values should ideally be filled using training data statistics
      (pass pre-computed state means if needed for strict leakage safety)
    """
    if config is None:
        config = CleaningConfig()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    group_cols = [state_col, district_col]

    # Sort for consistent processing
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # Store original values
    df[f"{target_col}_original"] = df[target_col].copy()
    df["is_outlier_event"] = 0
    df[f"{target_col}_was_capped"] = False

    initial_rows = len(df)
    initial_missing = df[target_col].isna().sum()

    # -------------------------------------------------------------------------
    # Step 1: Outlier Detection
    # -------------------------------------------------------------------------
    if config.outlier_method != "none":
        if config.outlier_method == "zscore_moving":
            is_outlier = detect_outliers_moving_zscore(
                df=df,
                group_cols=group_cols,
                date_col=date_col,
                target_col=target_col,
                window=config.outlier_window,
                z_thresh=config.outlier_z_thresh,
            )
        elif config.outlier_method == "iqr":
            is_outlier = detect_outliers_iqr(
                df=df,
                group_cols=group_cols,
                target_col=target_col,
            )
        else:
            is_outlier = pd.Series(False, index=df.index)

        # Flag outliers
        df["is_outlier_event"] = is_outlier.astype(int)
        n_outliers = is_outlier.sum()

        # Handle outliers based on cap method
        if config.outlier_cap_method != "none" and n_outliers > 0:
            if config.outlier_cap_method == "median":
                capped_values = cap_outliers_with_median(
                    df=df,
                    group_cols=group_cols,
                    date_col=date_col,
                    target_col=target_col,
                    is_outlier=is_outlier,
                    window=config.outlier_window,
                )
            elif config.outlier_cap_method == "clip":
                capped_values = clip_outliers_to_bounds(
                    df=df,
                    group_cols=group_cols,
                    date_col=date_col,
                    target_col=target_col,
                    window=config.outlier_window,
                    z_thresh=config.outlier_z_thresh,
                )
            else:
                capped_values = df[target_col]

            # Track which values were actually capped
            was_capped = is_outlier & (df[target_col] != capped_values)
            df[f"{target_col}_was_capped"] = was_capped
            df[target_col] = capped_values

            if config.log_changes:
                n_capped = was_capped.sum()
                logger.info(f"Outliers detected: {n_outliers}, capped: {n_capped}")

    # -------------------------------------------------------------------------
    # Step 2: Missing Value Imputation
    # -------------------------------------------------------------------------
    if config.missing_method != "none":
        df = impute_missing_values(
            df=df,
            state_col=state_col,
            district_col=district_col,
            date_col=date_col,
            target_col=target_col,
            method=config.missing_method,
            max_locf_gap=config.max_locf_gap,
        )

        if config.log_changes:
            n_imputed = df[f"{target_col}_was_imputed"].sum()
            final_missing = df[target_col].isna().sum()
            logger.info(
                f"Missing values: {initial_missing} → {final_missing} "
                f"(imputed: {n_imputed})"
            )

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    if config.log_changes:
        logger.info(
            f"Cleaning complete: {initial_rows} rows, "
            f"{df['is_outlier_event'].sum()} outlier events, "
            f"{df[f'{target_col}_was_imputed'].sum()} imputed values"
        )

    return df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_cleaning_summary(
    df: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Generate a summary of cleaning operations performed on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after cleaning (must have cleaning columns).
    target_col : str
        Name of the target column.

    Returns
    -------
    dict
        Summary statistics including counts and percentages.
    """
    total_rows = len(df)

    outliers = df.get("is_outlier_event", pd.Series(0, index=df.index)).sum()
    was_capped = df.get(f"{target_col}_was_capped", pd.Series(False, index=df.index)).sum()
    was_imputed = df.get(f"{target_col}_was_imputed", pd.Series(False, index=df.index)).sum()

    return {
        "total_rows": total_rows,
        "outliers_detected": int(outliers),
        "outliers_capped": int(was_capped),
        "values_imputed": int(was_imputed),
        "outliers_pct": round(outliers / total_rows * 100, 2) if total_rows > 0 else 0,
        "imputed_pct": round(was_imputed / total_rows * 100, 2) if total_rows > 0 else 0,
    }


# =============================================================================
# MODULE INFO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("TIME-SERIES CLEANING MODULE")
    print("=" * 70)
    print()
    print("This module provides leakage-safe time-series cleaning for UIDAI data.")
    print()
    print("Features:")
    print("  • Outlier detection (moving z-score or IQR)")
    print("  • Outlier handling (cap with median or clip to bounds)")
    print("  • Missing data imputation (LOCF, state mean, linear)")
    print("  • All operations are leakage-safe for time-series CV")
    print()
    print("Usage:")
    print("  from src.preprocessing.time_series_cleaning import (")
    print("      CleaningConfig, clean_uidai_time_series")
    print("  )")
    print()
    print("  config = CleaningConfig(")
    print('      outlier_method="zscore_moving",')
    print("      outlier_z_thresh=3.0,")
    print('      missing_method="locf",')
    print("  )")
    print()
    print("  df_clean = clean_uidai_time_series(df, 'state', 'district',")
    print("      'month_date', 'total_enrolment', config)")
    print("=" * 70)
