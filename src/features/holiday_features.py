"""
holiday_features.py
===================

Holiday and festival dummy features for Indian time-series forecasting.

**PURPOSE:**
Capture seasonal patterns related to Indian holidays, festivals, and events
that affect Aadhaar enrolment patterns.

**FEATURES CREATED:**
    - is_festival_peak: Major festival months (Diwali/Dussehra season)
    - is_exam_month: School/college exam periods
    - is_budget_month: Union Budget month (policy announcements)
    - is_fy_start: Financial year start (April)
    - is_fy_end: Financial year end (March)
    - is_monsoon: Monsoon season (may affect enrolment camps)
    - is_harvest: Harvest season (rural activity patterns)

**LEAKAGE SAFETY:**
    These features depend ONLY on the calendar date, not on any target values.
    They are exogenous features that can be safely computed for any date.

**TYPICAL PATTERNS:**
    - Festival months (Oct-Nov): Increased enrolment activity
    - Exam months (Mar-Apr): Students getting Aadhaar for exams
    - FY end (March): Government drives to meet targets
    - Monsoon (Jul-Sep): Potentially lower activity in rural areas

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Indian Holiday/Season Configuration
# =============================================================================

# Major festival months in India
# Oct-Nov: Dussehra, Durga Puja, Diwali, Bhai Dooj
# Jan: Pongal, Makar Sankranti, Republic Day
# Aug: Independence Day, Raksha Bandhan, Janmashtami
FESTIVAL_PEAK_MONTHS = [1, 8, 10, 11]

# Exam season months
# Mar-Apr: Board exams (10th, 12th), University exams
# Feb: Some competitive exams
EXAM_MONTHS = [2, 3, 4]

# Monsoon months (can affect field operations)
MONSOON_MONTHS = [7, 8, 9]

# Harvest months (rural activity patterns)
# Kharif harvest: Oct-Nov
# Rabi harvest: Mar-Apr
HARVEST_MONTHS = [3, 4, 10, 11]

# Union Budget month (policy announcements, typically Feb 1)
BUDGET_MONTH = 2


# =============================================================================
# Holiday Calendar Builder
# =============================================================================

def build_holiday_calendar(
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Create a DataFrame with monthly index and holiday/event features.

    Parameters
    ----------
    start : str
        Start date (e.g., '2020-01-01').
    end : str
        End date (e.g., '2026-12-31').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - month_date: Month-start dates
            - is_festival_peak: 1 if festival month
            - is_exam_month: 1 if exam month
            - is_budget_month: 1 if budget month
            - is_fy_start: 1 if April (FY start)
            - is_fy_end: 1 if March (FY end)
            - is_monsoon: 1 if monsoon month
            - is_harvest: 1 if harvest month

    Examples
    --------
    >>> calendar = build_holiday_calendar('2025-01-01', '2025-12-31')
    >>> print(calendar[['month_date', 'is_festival_peak', 'is_exam_month']])
    """
    # Create monthly date range
    dates = pd.date_range(start=start, end=end, freq="MS")

    calendar = pd.DataFrame({"month_date": dates})
    calendar["month"] = calendar["month_date"].dt.month

    # Festival peak months
    calendar["is_festival_peak"] = (
        calendar["month"].isin(FESTIVAL_PEAK_MONTHS).astype(int)
    )

    # Exam months
    calendar["is_exam_month"] = (
        calendar["month"].isin(EXAM_MONTHS).astype(int)
    )

    # Budget month
    calendar["is_budget_month"] = (
        calendar["month"] == BUDGET_MONTH
    ).astype(int)

    # Financial year boundaries
    calendar["is_fy_start"] = (calendar["month"] == 4).astype(int)
    calendar["is_fy_end"] = (calendar["month"] == 3).astype(int)

    # Monsoon season
    calendar["is_monsoon"] = (
        calendar["month"].isin(MONSOON_MONTHS).astype(int)
    )

    # Harvest season
    calendar["is_harvest"] = (
        calendar["month"].isin(HARVEST_MONTHS).astype(int)
    )

    # Drop helper column
    calendar = calendar.drop(columns=["month"])

    return calendar


def add_holiday_features(
    df: pd.DataFrame,
    date_col: str,
    holiday_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join holiday calendar features onto the main dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe with a date column.
    date_col : str
        Name of the date column in df.
    holiday_df : pd.DataFrame | None
        Holiday calendar from build_holiday_calendar().
        If None, builds one covering the date range in df.

    Returns
    -------
    pd.DataFrame
        DataFrame with holiday features added.

    Examples
    --------
    >>> df = add_holiday_features(df, 'month_date')
    >>> print(df[['month_date', 'is_festival_peak']].head())
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Build calendar if not provided
    if holiday_df is None:
        start = df[date_col].min().strftime("%Y-%m-%d")
        end = df[date_col].max().strftime("%Y-%m-%d")
        holiday_df = build_holiday_calendar(start, end)

    # Normalize dates for joining (use month start)
    df["_join_date"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    holiday_df = holiday_df.copy()
    holiday_df["_join_date"] = pd.to_datetime(holiday_df["month_date"])

    # Merge on normalized date
    holiday_cols = [c for c in holiday_df.columns if c != "month_date"]
    df = df.merge(holiday_df[holiday_cols], on="_join_date", how="left")

    # Fill any missing values with 0
    for col in holiday_cols:
        if col != "_join_date" and col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Clean up
    df = df.drop(columns=["_join_date"])

    return df


# =============================================================================
# Additional Event Features
# =============================================================================

def add_quarter_features(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Add quarter-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with quarter features:
            - quarter: 1-4
            - is_q1, is_q2, is_q3, is_q4: One-hot encoded quarters
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["quarter"] = df[date_col].dt.quarter
    df["is_q1"] = (df["quarter"] == 1).astype(int)
    df["is_q2"] = (df["quarter"] == 2).astype(int)
    df["is_q3"] = (df["quarter"] == 3).astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)

    return df


def add_month_sin_cos(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Add cyclical month encoding using sin/cos transformation.

    This captures the cyclical nature of months (December is close to January)
    which linear encoding (1-12) fails to represent.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
            - month_sin: sin(2π * month / 12)
            - month_cos: cos(2π * month / 12)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    month = df[date_col].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def print_holiday_calendar_summary(calendar: pd.DataFrame) -> None:
    """
    Print a formatted summary of the holiday calendar.

    Parameters
    ----------
    calendar : pd.DataFrame
        Output from build_holiday_calendar().
    """
    print("\n" + "=" * 70)
    print("INDIAN HOLIDAY/EVENT CALENDAR")
    print("=" * 70)

    feature_cols = [c for c in calendar.columns if c.startswith("is_")]

    print(f"\n{'Month':<12}", end="")
    for col in feature_cols:
        print(f"{col[3:]:>12}", end="")  # Remove 'is_' prefix
    print()
    print("-" * 70)

    for _, row in calendar.iterrows():
        month_str = row["month_date"].strftime("%Y-%m")
        print(f"{month_str:<12}", end="")
        for col in feature_cols:
            val = "✓" if row[col] == 1 else ""
            print(f"{val:>12}", end="")
        print()

    print("=" * 70 + "\n")
