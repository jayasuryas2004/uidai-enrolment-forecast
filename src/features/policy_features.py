"""
policy_features.py
==================

Policy and intervention dummy features for UIDAI forecasting.

**PURPOSE:**
Capture the impact of government policies, enrolment drives, and other
interventions that affect Aadhaar enrolment patterns.

**LEAKAGE SAFETY:**
These features depend ONLY on the calendar date, not on target values.
They represent known historical events or policy phases.

**TYPICAL POLICY PATTERNS:**
    - Enrolment drives during specific months
    - Policy phase changes (expansion phases)
    - Campaign months with boosted activity
    - Post-COVID recovery patterns

**IMPORTANT NOTE:**
In a real production system, these dates should be obtained from official
UIDAI announcements or internal records. The dates used here are
hypothetical examples for the hackathon context.

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Policy Phase Configuration
# =============================================================================

# Hypothetical policy phases for demonstration
# In production, replace with actual UIDAI policy dates
POLICY_PHASES: Dict[str, Tuple[str, str]] = {
    # Phase 1: Initial rollout period
    "phase_1": ("2025-01-01", "2025-06-30"),
    # Phase 2: Expansion period
    "phase_2": ("2025-07-01", "2025-09-30"),
    # Phase 3: Current operational period
    "phase_3": ("2025-10-01", "2026-12-31"),
}

# Campaign/drive months (hypothetical)
# These represent periods of increased government push for enrolments
CAMPAIGN_MONTHS: List[Tuple[str, str]] = [
    ("2025-04-01", "2025-04-30"),  # FY start drive
    ("2025-08-01", "2025-08-31"),  # Independence Day drive
    ("2025-10-01", "2025-10-31"),  # Festival season drive
]

# Special event months (e.g., policy announcements, system upgrades)
SPECIAL_EVENTS: Dict[str, str] = {
    "2025-02-01": "budget_announcement",
    "2025-08-15": "independence_day_target",
}


# =============================================================================
# Policy Feature Builder
# =============================================================================

def add_policy_phase_features(
    df: pd.DataFrame,
    date_col: str,
    policy_phases: Optional[Dict[str, Tuple[str, str]]] = None,
    campaign_periods: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Add dummy variables for known UIDAI policy phases and campaigns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date column.
    date_col : str
        Name of the date column.
    policy_phases : Dict[str, Tuple[str, str]] | None
        Dictionary mapping phase names to (start_date, end_date) tuples.
        If None, uses default POLICY_PHASES.
    campaign_periods : List[Tuple[str, str]] | None
        List of (start_date, end_date) tuples for campaign periods.
        If None, uses default CAMPAIGN_MONTHS.

    Returns
    -------
    pd.DataFrame
        DataFrame with policy feature columns:
            - is_{phase_name}: 1 if date is in that phase
            - is_campaign_month: 1 if date is in a campaign period
            - campaign_count: Cumulative count of campaigns (trend feature)

    Examples
    --------
    >>> df = add_policy_phase_features(df, 'month_date')
    >>> print(df[['month_date', 'is_phase_1', 'is_campaign_month']].head())

    Notes
    -----
    In production, replace the default dates with actual UIDAI policy dates
    from official sources.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Use defaults if not provided
    if policy_phases is None:
        policy_phases = POLICY_PHASES
    if campaign_periods is None:
        campaign_periods = CAMPAIGN_MONTHS

    # Add policy phase dummies
    for phase_name, (start, end) in policy_phases.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        col_name = f"is_{phase_name}"
        df[col_name] = (
            (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
        ).astype(int)

    # Add campaign month indicator
    df["is_campaign_month"] = 0
    for start, end in campaign_periods:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
        df.loc[mask, "is_campaign_month"] = 1

    # Add cumulative campaign count (trend feature)
    # This captures the idea that more campaigns have occurred over time
    df["campaign_count"] = 0
    for start, end in sorted(campaign_periods):
        end_dt = pd.to_datetime(end)
        mask = df[date_col] > end_dt
        df.loc[mask, "campaign_count"] += 1

    return df


def add_time_trend_features(
    df: pd.DataFrame,
    date_col: str,
    reference_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add time trend features for capturing secular trends.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.
    reference_date : str | None
        Reference date for computing time since.
        If None, uses the minimum date in the data.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
            - months_since_start: Months since reference date
            - months_since_start_sq: Squared (for non-linear trends)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if reference_date is None:
        ref_date = df[date_col].min()
    else:
        ref_date = pd.to_datetime(reference_date)

    # Compute months since reference
    df["months_since_start"] = (
        (df[date_col].dt.year - ref_date.year) * 12 +
        (df[date_col].dt.month - ref_date.month)
    )

    # Squared term for non-linear trends
    df["months_since_start_sq"] = df["months_since_start"] ** 2

    return df


def add_intervention_dummy(
    df: pd.DataFrame,
    date_col: str,
    intervention_date: str,
    feature_name: str = "post_intervention",
    include_interaction_with_trend: bool = False,
) -> pd.DataFrame:
    """
    Add a step-function dummy for a specific intervention/policy change.

    This is useful for interrupted time-series analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.
    intervention_date : str
        Date of the intervention (e.g., '2025-07-01').
    feature_name : str, default 'post_intervention'
        Name for the intervention dummy column.
    include_interaction_with_trend : bool, default False
        If True, also add an interaction term with time trend
        (to capture change in slope after intervention).

    Returns
    -------
    pd.DataFrame
        DataFrame with intervention dummy column(s).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    intervention_dt = pd.to_datetime(intervention_date)

    # Step function: 0 before intervention, 1 after
    df[feature_name] = (df[date_col] >= intervention_dt).astype(int)

    if include_interaction_with_trend:
        # Add time since intervention (0 before, increasing after)
        df[f"{feature_name}_time"] = (
            df[date_col] - intervention_dt
        ).dt.days / 30  # Convert to months
        df[f"{feature_name}_time"] = df[f"{feature_name}_time"].clip(lower=0)

    return df


def add_covid_recovery_features(
    df: pd.DataFrame,
    date_col: str,
    covid_start: str = "2020-03-01",
    recovery_start: str = "2021-06-01",
    full_recovery: str = "2022-01-01",
) -> pd.DataFrame:
    """
    Add features capturing COVID-19 impact and recovery patterns.

    This is relevant for Aadhaar data that spans the pandemic period.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column name.
    covid_start : str
        Start of COVID impact (lockdown).
    recovery_start : str
        Start of recovery period.
    full_recovery : str
        Approximate date of full recovery.

    Returns
    -------
    pd.DataFrame
        DataFrame with COVID-related features:
            - is_covid_impact: 1 during COVID impact period
            - is_covid_recovery: 1 during recovery period
            - post_covid: 1 after full recovery
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    covid_start_dt = pd.to_datetime(covid_start)
    recovery_start_dt = pd.to_datetime(recovery_start)
    full_recovery_dt = pd.to_datetime(full_recovery)

    # COVID impact period
    df["is_covid_impact"] = (
        (df[date_col] >= covid_start_dt) & (df[date_col] < recovery_start_dt)
    ).astype(int)

    # Recovery period
    df["is_covid_recovery"] = (
        (df[date_col] >= recovery_start_dt) & (df[date_col] < full_recovery_dt)
    ).astype(int)

    # Post-COVID
    df["post_covid"] = (df[date_col] >= full_recovery_dt).astype(int)

    return df


def print_policy_summary(df: pd.DataFrame, date_col: str) -> None:
    """
    Print a summary of policy features in the data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with policy features.
    date_col : str
        Date column name.
    """
    print("\n" + "=" * 60)
    print("POLICY/INTERVENTION FEATURE SUMMARY")
    print("=" * 60)

    policy_cols = [c for c in df.columns if c.startswith("is_phase_") or c.startswith("is_campaign")]

    for col in policy_cols:
        n_active = df[col].sum()
        pct = 100 * n_active / len(df)
        print(f"{col}: {n_active:,} rows ({pct:.1f}%)")

    print("=" * 60 + "\n")
