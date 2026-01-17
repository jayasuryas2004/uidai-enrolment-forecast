"""
phase4_feature_enhancements.py
==============================

Phase-4 Feature Enhancements: Additional feature engineering for UIDAI forecasting.

This module provides production-style feature engineering functions to improve
model performance beyond the baseline ctx_v3 features:

    1. add_calendar_features() - Richer calendar/event features (quarter, FY, festivals)
    2. add_group_aggregates() - Group-level statistics and rolling aggregates
    3. apply_log_transform() / invert_log_transform() - Target transformation helpers

**USAGE:**
These functions are designed to be called BEFORE the Phase-4 training pipeline.
They operate on DataFrames and do NOT include any model training.

    from src.phase4_feature_enhancements import (
        add_calendar_features,
        add_group_aggregates,
        apply_log_transform,
    )

    # Enhance features
    df = add_calendar_features(df, date_col="month_date")
    df = add_group_aggregates(df, target_col="total_enrolment")

    # Optional: transform target
    y_log, offset = apply_log_transform(df["total_enrolment"])

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Part 1: Calendar / Event Features
# =============================================================================

# Placeholder list of festival months (India-specific)
# Note: This is a simplified approximation. Major festivals include:
#   - August: Raksha Bandhan, Independence Day
#   - September: Ganesh Chaturthi
#   - October: Durga Puja, Dussehra
#   - November: Diwali, Bhai Dooj
# These months typically see increased Aadhaar activity for new registrations.
FESTIVAL_MONTHS: List[int] = [8, 9, 10, 11]


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "month_date",
) -> pd.DataFrame:
    """
    Add richer calendar features for monthly UIDAI data.

    Features added:
    - year: Calendar year (e.g., 2025)
    - month: Month number (1-12)
    - quarter: Calendar quarter (1-4)
    - financial_year: Indian financial year (April-March)
    - is_financial_year_start: True if month is April (start of FY)
    - is_financial_year_end: True if month is March (end of FY)
    - is_festival_month: True if month is in festival season (Aug-Nov)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date column.
    date_col : str, default "month_date"
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional calendar feature columns.

    Notes
    -----
    Indian Financial Year runs April to March:
        - FY 2024-2025: April 2024 to March 2025
        - For April-December, financial_year = calendar_year
        - For January-March, financial_year = calendar_year - 1

    Examples
    --------
    >>> df = add_calendar_features(df, date_col="month_date")
    >>> print(df[["month_date", "financial_year", "is_festival_month"]].head())
    """
    df = df.copy()

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Basic calendar features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter

    # Indian Financial Year (April-March)
    # For months 4-12: FY = year (e.g., Oct 2025 -> FY 2025-26, we store 2025)
    # For months 1-3: FY = year - 1 (e.g., Feb 2026 -> FY 2025-26, we store 2025)
    df["financial_year"] = np.where(
        df["month"] >= 4,
        df["year"],
        df["year"] - 1
    )

    # Financial year boundaries
    df["is_financial_year_start"] = (df["month"] == 4).astype(int)
    df["is_financial_year_end"] = (df["month"] == 3).astype(int)

    # Festival season indicator (placeholder - Aug to Nov)
    df["is_festival_month"] = df["month"].isin(FESTIVAL_MONTHS).astype(int)

    return df


# =============================================================================
# Part 2: Group-Level Aggregates
# =============================================================================

def add_group_aggregates(
    df: pd.DataFrame,
    target_col: str = "total_enrolment",
    date_col: str = "month_date",
    group_cols: Optional[List[str]] = None,
    fill_na_with_group_mean: bool = False,
) -> pd.DataFrame:
    """
    Add group-level aggregate features for UIDAI enrolment forecasting.

    Features added per group (state, district):
    - group_long_term_mean: Historical mean of target for the group
    - group_long_term_std: Historical std of target for the group
    - rolling_3_mean: Rolling mean over last 3 periods (shifted, no leakage)
    - rolling_6_mean: Rolling mean over last 6 periods (shifted, no leakage)
    - ratio_to_state_mean: Ratio of target to state-level mean at same date

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with target and group columns.
    target_col : str, default "total_enrolment"
        Name of the target column.
    date_col : str, default "month_date"
        Name of the date column for sorting.
    group_cols : List[str] | None, default None
        Columns to group by. If None, defaults to ["state", "district"].
    fill_na_with_group_mean : bool, default False
        If True, fill NaN values from rolling with group long-term mean.
        If False, keep NaN (XGBoost can handle missing values).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional aggregate feature columns.

    Notes
    -----
    - Rolling features are SHIFTED by 1 period to prevent data leakage.
    - The rolling window uses min_periods=1 to compute partial means early on.
    - ratio_to_state_mean requires a 'state' column; skipped if not present.

    Examples
    --------
    >>> df = add_group_aggregates(df, target_col="total_enrolment")
    >>> print(df[["state", "district", "rolling_3_mean", "ratio_to_state_mean"]].head())
    """
    df = df.copy()

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Determine group columns
    if group_cols is None:
        default_groups = ["state", "district"]
        group_cols = [col for col in default_groups if col in df.columns]

    if not group_cols:
        # No grouping columns found, use entire dataset as one group
        group_cols = []

    # Sort by group + date for correct temporal ordering
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Long-term group statistics
    # -------------------------------------------------------------------------
    if group_cols:
        # Compute group-level long-term mean and std
        group_stats = df.groupby(group_cols, observed=True)[target_col].agg(
            ["mean", "std"]
        ).reset_index()
        group_stats.columns = group_cols + ["group_long_term_mean", "group_long_term_std"]

        # Merge back to original DataFrame
        df = df.merge(group_stats, on=group_cols, how="left")
    else:
        # Global stats if no grouping
        df["group_long_term_mean"] = df[target_col].mean()
        df["group_long_term_std"] = df[target_col].std()

    # -------------------------------------------------------------------------
    # Rolling aggregates (per group, shifted to prevent leakage)
    # -------------------------------------------------------------------------
    if group_cols:
        # Rolling 3-period mean (shifted by 1)
        df["rolling_3_mean"] = (
            df.groupby(group_cols, observed=True)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )

        # Rolling 6-period mean (shifted by 1)
        df["rolling_6_mean"] = (
            df.groupby(group_cols, observed=True)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=6, min_periods=1).mean())
        )
    else:
        # Global rolling if no grouping
        df["rolling_3_mean"] = df[target_col].shift(1).rolling(window=3, min_periods=1).mean()
        df["rolling_6_mean"] = df[target_col].shift(1).rolling(window=6, min_periods=1).mean()

    # -------------------------------------------------------------------------
    # Ratio to state-level mean (if state column exists)
    # -------------------------------------------------------------------------
    if "state" in df.columns:
        # Compute state-level mean for each date
        state_date_mean = df.groupby(["state", date_col], observed=True)[target_col].transform("mean")
        
        # Ratio of individual value to state mean
        # Avoid division by zero
        df["ratio_to_state_mean"] = np.where(
            state_date_mean > 0,
            df[target_col] / state_date_mean,
            1.0  # Default ratio if state mean is 0
        )
    else:
        # Skip ratio feature if state column not present
        df["ratio_to_state_mean"] = 1.0

    # -------------------------------------------------------------------------
    # Handle NaN values from rolling
    # -------------------------------------------------------------------------
    if fill_na_with_group_mean:
        # Fill NaN rolling values with group long-term mean
        df["rolling_3_mean"] = df["rolling_3_mean"].fillna(df["group_long_term_mean"])
        df["rolling_6_mean"] = df["rolling_6_mean"].fillna(df["group_long_term_mean"])
    # else: keep NaN - XGBoost handles missing values natively

    return df


# =============================================================================
# Part 3: Target Transformation Helpers
# =============================================================================

def apply_log_transform(
    y: pd.Series,
    offset: float = 1.0,
) -> Tuple[pd.Series, float]:
    """
    Safely apply log transform to a target series.

    The log transform can help with:
    - Stabilizing variance in skewed distributions
    - Making multiplicative relationships additive
    - Reducing the impact of outliers

    Parameters
    ----------
    y : pd.Series
        Target series to transform.
    offset : float, default 1.0
        Value added before taking log to handle zeros.
        Must be >= 0. Choose offset so that min(y + offset) > 0.

    Returns
    -------
    Tuple[pd.Series, float]
        - y_log: log-transformed series = log(y + offset)
        - offset: the offset used (for later inversion)

    Raises
    ------
    ValueError
        If offset is negative or if (y + offset) contains non-positive values.

    Examples
    --------
    >>> y_log, offset = apply_log_transform(df["total_enrolment"], offset=1.0)
    >>> # Later, to invert:
    >>> y_original = invert_log_transform(y_log, offset)
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")

    y_shifted = y + offset

    if (y_shifted <= 0).any():
        min_val = y_shifted.min()
        raise ValueError(
            f"All values must be positive after adding offset. "
            f"min(y + {offset}) = {min_val}. Increase offset."
        )

    y_log = np.log(y_shifted)

    return y_log, offset


def invert_log_transform(
    y_log: pd.Series,
    offset: float,
) -> pd.Series:
    """
    Invert a log transform applied with apply_log_transform.

    Parameters
    ----------
    y_log : pd.Series
        Log-transformed series.
    offset : float
        The offset that was used in apply_log_transform.

    Returns
    -------
    pd.Series
        Original-scale series = exp(y_log) - offset

    Examples
    --------
    >>> y_log, offset = apply_log_transform(y, offset=1.0)
    >>> y_pred_log = model.predict(X)  # predictions in log scale
    >>> y_pred = invert_log_transform(pd.Series(y_pred_log), offset)
    """
    return np.exp(y_log) - offset


def apply_sqrt_transform(
    y: pd.Series,
) -> pd.Series:
    """
    Apply square root transform to a target series.

    Alternative to log transform, less aggressive for variance stabilization.

    Parameters
    ----------
    y : pd.Series
        Target series (must be non-negative).

    Returns
    -------
    pd.Series
        sqrt-transformed series

    Raises
    ------
    ValueError
        If y contains negative values.
    """
    if (y < 0).any():
        raise ValueError("Cannot apply sqrt to negative values.")

    return np.sqrt(y)


def invert_sqrt_transform(
    y_sqrt: pd.Series,
) -> pd.Series:
    """
    Invert a sqrt transform.

    Parameters
    ----------
    y_sqrt : pd.Series
        Sqrt-transformed series.

    Returns
    -------
    pd.Series
        Original-scale series = y_sqrt ** 2
    """
    return y_sqrt ** 2


# =============================================================================
# Part 4: Convenience Function to Apply All Enhancements
# =============================================================================

def enhance_features(
    df: pd.DataFrame,
    date_col: str = "month_date",
    target_col: str = "total_enrolment",
    group_cols: Optional[List[str]] = None,
    add_calendar: bool = True,
    add_aggregates: bool = True,
) -> pd.DataFrame:
    """
    Apply all Phase-4 feature enhancements in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str, default "month_date"
        Name of the date column.
    target_col : str, default "total_enrolment"
        Name of the target column.
    group_cols : List[str] | None, default None
        Columns to group by for aggregates.
    add_calendar : bool, default True
        If True, add calendar features.
    add_aggregates : bool, default True
        If True, add group-level aggregates.

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with all new features.

    Examples
    --------
    >>> df_enhanced = enhance_features(df)
    >>> print(df_enhanced.columns.tolist())
    """
    df = df.copy()

    if add_calendar:
        df = add_calendar_features(df, date_col=date_col)

    if add_aggregates:
        df = add_group_aggregates(
            df,
            target_col=target_col,
            date_col=date_col,
            group_cols=group_cols,
        )

    return df


# =============================================================================
# Part 5: Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Demonstrate Phase-4 feature enhancements with synthetic data.
    """
    from datetime import datetime, timedelta

    print("=" * 60)
    print("Phase-4 Feature Enhancements Demo")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Create synthetic DataFrame
    # -------------------------------------------------------------------------
    np.random.seed(42)

    states = ["Maharashtra", "Gujarat"]
    districts = ["District_A", "District_B"]
    base_date = datetime(2025, 4, 1)  # Start of FY 2025-26

    records = []
    for state in states:
        for district in districts:
            for month_offset in range(9):  # 9 months of data
                date = base_date + timedelta(days=month_offset * 30)
                enrolment = int(100 + np.random.normal(50, 20))
                records.append({
                    "month_date": date,
                    "state": state,
                    "district": district,
                    "total_enrolment": max(0, enrolment),
                })

    df = pd.DataFrame(records)
    print("\nOriginal DataFrame:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # -------------------------------------------------------------------------
    # Apply calendar features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Adding calendar features...")
    df_cal = add_calendar_features(df, date_col="month_date")

    new_cols = [c for c in df_cal.columns if c not in df.columns]
    print(f"New columns: {new_cols}")
    print(df_cal[["month_date", "quarter", "financial_year", "is_festival_month"]].head(10))

    # -------------------------------------------------------------------------
    # Apply group aggregates
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Adding group aggregates...")
    df_agg = add_group_aggregates(
        df_cal,
        target_col="total_enrolment",
        group_cols=["state", "district"],
    )

    new_cols = [c for c in df_agg.columns if c not in df_cal.columns]
    print(f"New columns: {new_cols}")
    print(df_agg[["state", "district", "rolling_3_mean", "rolling_6_mean", "ratio_to_state_mean"]].head(10))

    # -------------------------------------------------------------------------
    # Demonstrate log transform
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Testing log transform...")

    y = df_agg["total_enrolment"]
    print(f"Original y: min={y.min()}, max={y.max()}, mean={y.mean():.2f}")

    y_log, offset = apply_log_transform(y, offset=1.0)
    print(f"Log-transformed: min={y_log.min():.2f}, max={y_log.max():.2f}, mean={y_log.mean():.2f}")

    y_inverted = invert_log_transform(y_log, offset)
    print(f"Inverted: min={y_inverted.min():.2f}, max={y_inverted.max():.2f}")

    # Verify round-trip
    max_error = (y - y_inverted).abs().max()
    print(f"Max round-trip error: {max_error:.2e}")

    # -------------------------------------------------------------------------
    # Use convenience function
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Using enhance_features() convenience function...")

    df_enhanced = enhance_features(df)
    print(f"Enhanced shape: {df_enhanced.shape}")
    print(f"All columns: {df_enhanced.columns.tolist()}")

    print("\n" + "=" * 60)
    print("✅ Phase-4 feature enhancements demo complete!")
    print("=" * 60)
