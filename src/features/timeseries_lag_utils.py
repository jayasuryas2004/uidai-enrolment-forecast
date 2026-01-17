"""
timeseries_lag_utils.py
=======================

ACF/PACF-guided lag selection utilities for time-series feature engineering.

**PURPOSE:**
Select optimal lag features based on autocorrelation analysis rather than
blindly adding many lags (which can cause overfitting).

**METHODOLOGY:**
    1. Compute ACF (Autocorrelation Function) and PACF (Partial ACF) per district
    2. Identify lags with significant correlation (above threshold)
    3. Select top-K lags to avoid dimensionality explosion
    4. Use a representative sample of districts to determine global lag set

**TYPICAL RESULTS FOR MONTHLY DATA:**
    - Lag 1: Previous month (strong autocorrelation)
    - Lag 2-3: Recent history
    - Lag 6: Half-year seasonality
    - Lag 12: Annual seasonality (if enough data)

**LEAKAGE SAFETY:**
    - ACF/PACF analysis should be done on TRAINING data only
    - The selected lags are then used to create features

Author: UIDAI Forecast Team
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import statsmodels; provide fallback if not available
try:
    from statsmodels.tsa.stattools import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning(
        "statsmodels not installed. ACF/PACF analysis unavailable. "
        "Using default lag selection instead."
    )


# =============================================================================
# Default Lag Configuration (fallback when statsmodels unavailable)
# =============================================================================

# These are sensible defaults for monthly enrolment data
DEFAULT_LAGS = [1, 2, 3, 6]  # Previous month, recent history, half-year


# =============================================================================
# ACF/PACF Computation
# =============================================================================

def compute_acf_pacf_for_district(
    df: pd.DataFrame,
    district_col: str,
    date_col: str,
    target_col: str,
    district_name: str,
    max_lag: int = 12,
) -> Dict[str, np.ndarray]:
    """
    Compute ACF and PACF for a single district's time series.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing all districts.
    district_col : str
        Name of the district column.
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column (e.g., 'total_enrolment').
    district_name : str
        Name of the specific district to analyze.
    max_lag : int, default 12
        Maximum lag to compute (should be < n_observations / 2).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with:
            - 'lags': array of lag indices [0, 1, 2, ..., max_lag]
            - 'acf': ACF values for each lag
            - 'pacf': PACF values for each lag

    Raises
    ------
    ValueError
        If district not found or insufficient data.

    Examples
    --------
    >>> result = compute_acf_pacf_for_district(
    ...     df, 'district', 'month_date', 'total_enrolment', 'Mumbai', max_lag=6
    ... )
    >>> print(result['acf'])  # ACF values at each lag
    """
    if not HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for ACF/PACF analysis. "
            "Install with: pip install statsmodels"
        )

    # Filter to specific district
    mask = df[district_col] == district_name
    if mask.sum() == 0:
        raise ValueError(f"District '{district_name}' not found in data")

    df_district = df[mask].copy()
    df_district = df_district.sort_values(date_col).reset_index(drop=True)

    # Extract time series
    series = df_district[target_col].values

    # Validate sufficient data
    n_obs = len(series)
    if n_obs < max_lag + 2:
        raise ValueError(
            f"District '{district_name}' has only {n_obs} observations, "
            f"need at least {max_lag + 2} for max_lag={max_lag}"
        )

    # Adjust max_lag if necessary
    effective_max_lag = min(max_lag, n_obs // 2 - 1)
    if effective_max_lag < 1:
        raise ValueError(f"Insufficient data for lag analysis: {n_obs} observations")

    # Compute ACF and PACF
    acf_values = acf(series, nlags=effective_max_lag, fft=True)
    pacf_values = pacf(series, nlags=effective_max_lag, method="ywm")

    return {
        "lags": np.arange(effective_max_lag + 1),
        "acf": acf_values,
        "pacf": pacf_values,
        "n_observations": n_obs,
        "district": district_name,
    }


def select_significant_lags(
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    max_lag: int,
    threshold: float = 0.2,
    top_k: int = 5,
) -> List[int]:
    """
    Select significant lags based on ACF/PACF magnitude.

    This function uses a simple heuristic:
        1. Consider lags >= 1 (skip lag 0 which is always 1.0)
        2. Combine ACF and PACF by taking max absolute value at each lag
        3. Keep lags where combined value > threshold
        4. Sort by magnitude and take top_k

    Parameters
    ----------
    acf_values : np.ndarray
        ACF values from compute_acf_pacf_for_district.
    pacf_values : np.ndarray
        PACF values from compute_acf_pacf_for_district.
    max_lag : int
        Maximum lag to consider.
    threshold : float, default 0.2
        Minimum correlation magnitude to consider significant.
        Common thresholds: 0.2 (moderate), 0.1 (lenient), 0.3 (strict).
    top_k : int, default 5
        Maximum number of lags to return.

    Returns
    -------
    List[int]
        Sorted list of selected lag values (e.g., [1, 2, 3, 6]).

    Notes
    -----
    The threshold of 0.2 is a practical choice. For n=50 observations,
    the 95% confidence interval is approximately ±0.28, so 0.2 is
    slightly lenient but reasonable for feature engineering.

    Examples
    --------
    >>> lags = select_significant_lags(acf, pacf, max_lag=12, threshold=0.2, top_k=5)
    >>> print(lags)  # e.g., [1, 2, 3, 6]
    """
    # Ensure we don't exceed available data
    effective_max = min(max_lag, len(acf_values) - 1, len(pacf_values) - 1)

    # Combine ACF and PACF: use max absolute value at each lag
    combined_importance = []
    for lag in range(1, effective_max + 1):  # Skip lag 0
        importance = max(abs(acf_values[lag]), abs(pacf_values[lag]))
        if importance >= threshold:
            combined_importance.append((lag, importance))

    # Sort by importance (descending) and take top_k
    combined_importance.sort(key=lambda x: x[1], reverse=True)
    selected_lags = [lag for lag, _ in combined_importance[:top_k]]

    # Sort lags in ascending order for consistency
    selected_lags.sort()

    # Always include lag 1 if not already present (previous month is always useful)
    if 1 not in selected_lags and len(selected_lags) < top_k:
        selected_lags = [1] + selected_lags
        selected_lags.sort()

    return selected_lags


def get_recommended_lags(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    target_col: str,
    max_lag: int = 12,
    sample_size: int = 5,
    threshold: float = 0.2,
    top_k: int = 5,
    random_state: int = 42,
) -> List[int]:
    """
    Analyze a sample of groups and return recommended lags for all.

    This function:
        1. Samples a few representative groups (e.g., districts)
        2. Computes ACF/PACF for each
        3. Aggregates significant lags across samples
        4. Returns a global lag set for feature engineering

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    group_cols : List[str]
        Columns defining groups (e.g., ['state', 'district']).
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    max_lag : int, default 12
        Maximum lag to analyze.
    sample_size : int, default 5
        Number of groups to sample for analysis.
    threshold : float, default 0.2
        Correlation threshold for significance.
    top_k : int, default 5
        Max lags to select per group.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    List[int]
        Sorted list of recommended lags (e.g., [1, 2, 3, 6]).

    Notes
    -----
    If statsmodels is not available or analysis fails, returns DEFAULT_LAGS.
    This ensures the pipeline doesn't break even without ACF/PACF analysis.

    Examples
    --------
    >>> lags = get_recommended_lags(
    ...     df, ['state', 'district'], 'month_date', 'total_enrolment'
    ... )
    >>> print(lags)  # e.g., [1, 2, 3, 6]
    """
    if not HAS_STATSMODELS:
        logging.info(f"statsmodels not available. Using default lags: {DEFAULT_LAGS}")
        return DEFAULT_LAGS.copy()

    # Get unique groups
    if len(group_cols) == 0:
        logging.warning("No group columns provided. Using default lags.")
        return DEFAULT_LAGS.copy()

    # Create group identifier
    df = df.copy()
    df["_group_id"] = df[group_cols].astype(str).agg("_".join, axis=1)
    unique_groups = df["_group_id"].unique()

    # Sample groups
    np.random.seed(random_state)
    n_sample = min(sample_size, len(unique_groups))
    sampled_groups = np.random.choice(unique_groups, size=n_sample, replace=False)

    # Collect lags from each sampled group
    all_lags: Dict[int, int] = {}  # lag -> count

    for group_id in sampled_groups:
        try:
            group_mask = df["_group_id"] == group_id
            group_df = df[group_mask].copy()

            if len(group_df) < max_lag + 2:
                continue

            # Get target series
            group_df = group_df.sort_values(date_col)
            series = group_df[target_col].values

            # Compute ACF/PACF
            effective_max = min(max_lag, len(series) // 2 - 1)
            if effective_max < 1:
                continue

            acf_vals = acf(series, nlags=effective_max, fft=True)
            pacf_vals = pacf(series, nlags=effective_max, method="ywm")

            # Select significant lags
            lags = select_significant_lags(
                acf_vals, pacf_vals, effective_max, threshold, top_k
            )

            # Count occurrences
            for lag in lags:
                all_lags[lag] = all_lags.get(lag, 0) + 1

        except Exception as e:
            logging.debug(f"Failed to analyze group {group_id}: {e}")
            continue

    if not all_lags:
        logging.warning("ACF/PACF analysis failed for all groups. Using default lags.")
        return DEFAULT_LAGS.copy()

    # Select lags that appear in at least 40% of sampled groups
    min_count = max(1, int(n_sample * 0.4))
    common_lags = [lag for lag, count in all_lags.items() if count >= min_count]

    # If too few common lags, take most frequent ones
    if len(common_lags) < 3:
        sorted_lags = sorted(all_lags.items(), key=lambda x: x[1], reverse=True)
        common_lags = [lag for lag, _ in sorted_lags[:top_k]]

    common_lags.sort()

    # Always ensure lag 1 is included
    if 1 not in common_lags:
        common_lags = [1] + common_lags

    logging.info(f"Recommended lags from ACF/PACF analysis: {common_lags}")
    return common_lags


def print_acf_pacf_summary(result: Dict[str, np.ndarray]) -> None:
    """
    Print a formatted summary of ACF/PACF analysis.

    Parameters
    ----------
    result : Dict
        Output from compute_acf_pacf_for_district.
    """
    print("\n" + "=" * 50)
    print(f"ACF/PACF ANALYSIS: {result.get('district', 'Unknown')}")
    print("=" * 50)
    print(f"Observations: {result.get('n_observations', 'N/A')}")
    print(f"\n{'Lag':<6} {'ACF':>10} {'PACF':>10} {'Significant':>12}")
    print("-" * 40)

    for lag in result["lags"]:
        if lag == 0:
            continue  # Skip lag 0
        acf_val = result["acf"][lag]
        pacf_val = result["pacf"][lag]
        significant = "✓" if abs(acf_val) > 0.2 or abs(pacf_val) > 0.2 else ""
        print(f"{lag:<6} {acf_val:>10.3f} {pacf_val:>10.3f} {significant:>12}")

    print("=" * 50 + "\n")
