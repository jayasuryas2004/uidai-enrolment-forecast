"""
time_series_cv.py
=================

Leakage-safe, expanding-window time-series cross-validation utilities.

**PURPOSE:**
Provide robust time-series cross-validation that:
    1. Respects temporal ordering (past → future, NO shuffling)
    2. Includes a gap between train and validation to prevent leakage
    3. Uses an expanding training window (more data over time)
    4. Produces realistic, unbiased model performance estimates

**WHY EXPANDING WINDOW + GAP?**
    - Expanding window: Mimics real deployment where we always have more
      historical data as time progresses.
    - Gap: Prevents leakage from lag/rolling features that might "see"
      validation data if computed naively.

**USAGE:**
    from src.validation.time_series_cv import generate_expanding_folds

    folds = generate_expanding_folds(
        df=df,
        date_col="month_date",
        n_folds=5,
        gap_months=1,
        min_train_months=6,
    )

    for train_mask, val_mask in folds:
        df_train = df[train_mask]
        df_val = df[val_mask]
        # ... train and evaluate

**CRITICAL LEAKAGE PREVENTION:**
    - This module only generates masks. Feature engineering must be done
      SEPARATELY per fold using only training data.
    - Never fit preprocessors (encoders, scalers) on validation data.
    - Never compute lag/rolling features using validation data.

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_expanding_folds(
    df: pd.DataFrame,
    date_col: str,
    n_folds: int = 5,
    gap_months: int = 1,
    min_train_months: int = 6,
    val_months: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate boolean train/val masks for expanding-window time-series CV with a gap.

    This function creates non-overlapping, time-ordered folds where:
        - Training data always precedes validation data
        - A gap of `gap_months` separates train and validation to prevent leakage
        - Training window expands as we move through folds
        - Validation window is fixed at `val_months`

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time-series data.
    date_col : str
        Name of the datetime column (must be datetime64 type).
    n_folds : int, default 5
        Number of cross-validation folds.
    gap_months : int, default 1
        Number of months between end of training and start of validation.
        This gap prevents information leakage from lag/rolling features.
    min_train_months : int, default 6
        Minimum number of months required in the first training fold.
    val_months : int, default 1
        Number of months in each validation fold.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_mask, val_mask) tuples, where each mask is a boolean
        array aligned with df's index.

    Raises
    ------
    ValueError
        If there are not enough months for the requested configuration.

    Examples
    --------
    >>> folds = generate_expanding_folds(
    ...     df, date_col="month_date", n_folds=4, gap_months=1, min_train_months=3
    ... )
    >>> for i, (train_mask, val_mask) in enumerate(folds):
    ...     print(f"Fold {i}: train={train_mask.sum()}, val={val_mask.sum()}")

    Notes
    -----
    IMPORTANT: This function only generates masks. You MUST:
        1. Build features on df[train_mask] ONLY (fit preprocessors here)
        2. Apply the same transformations to df[val_mask] (transform only)
        3. Never let validation data influence training features

    Timeline visualization for n_folds=3, gap_months=1, val_months=1:

        Months:  [1] [2] [3] [4] [5] [6] [7] [8] [9]
        Fold 1:  [TRAIN TRAIN TRAIN] GAP [VAL]
        Fold 2:  [TRAIN TRAIN TRAIN TRAIN] GAP [VAL]
        Fold 3:  [TRAIN TRAIN TRAIN TRAIN TRAIN] GAP [VAL]
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError(f"Column '{date_col}' must be datetime64 type")

    # Get sorted unique months
    unique_months = np.sort(df[date_col].unique())
    n_months = len(unique_months)

    # Calculate required months
    # For each fold after the first, we add 1 month to training
    # Required: min_train + gap + val for first fold, then +1 for each additional fold
    required_months = min_train_months + gap_months + val_months + (n_folds - 1)

    if n_months < required_months:
        raise ValueError(
            f"Not enough months for {n_folds} folds. "
            f"Have {n_months} months, need at least {required_months}. "
            f"(min_train={min_train_months}, gap={gap_months}, val={val_months})"
        )

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    # Start from the end and work backwards to determine fold boundaries
    # This ensures we use the most recent data for validation
    for fold_idx in range(n_folds):
        # Calculate indices from the end
        # For fold 0 (last chronologically): val ends at last month
        # For fold 1: val ends 1 month earlier, etc.
        val_end_idx = n_months - 1 - fold_idx
        val_start_idx = val_end_idx - val_months + 1

        # Training ends before the gap
        train_end_idx = val_start_idx - gap_months - 1

        # Training starts at the beginning (expanding window)
        train_start_idx = 0

        # Validate we have enough training months
        actual_train_months = train_end_idx - train_start_idx + 1
        if actual_train_months < min_train_months:
            # Skip this fold if not enough training data
            continue

        # Extract month values for this fold
        train_months = unique_months[train_start_idx : train_end_idx + 1]
        val_months_arr = unique_months[val_start_idx : val_end_idx + 1]

        # Create boolean masks
        train_mask = df[date_col].isin(train_months).values
        val_mask = df[date_col].isin(val_months_arr).values

        folds.append((train_mask, val_mask))

    # Reverse to get chronological order (earliest val first)
    folds = folds[::-1]

    return folds


def print_fold_summary(
    df: pd.DataFrame,
    date_col: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Print a human-readable summary of the fold configuration.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to generate folds.
    date_col : str
        Name of the date column.
    folds : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_mask, val_mask) from generate_expanding_folds.
    """
    print("\n" + "=" * 70)
    print("TIME-SERIES CROSS-VALIDATION FOLD SUMMARY")
    print("=" * 70)
    print(f"{'Fold':<6} {'Train Rows':>12} {'Train Months':>14} {'Val Rows':>10} {'Val Month':>12}")
    print("-" * 70)

    for i, (train_mask, val_mask) in enumerate(folds):
        train_dates = df.loc[train_mask, date_col]
        val_dates = df.loc[val_mask, date_col]

        train_months = train_dates.nunique()
        train_range = f"{train_dates.min().strftime('%Y-%m')}→{train_dates.max().strftime('%Y-%m')}"
        val_month = val_dates.iloc[0].strftime("%Y-%m") if len(val_dates) > 0 else "N/A"

        print(
            f"{i+1:<6} {train_mask.sum():>12,} {train_range:>14} "
            f"{val_mask.sum():>10,} {val_month:>12}"
        )

    print("-" * 70)
    print(f"Total folds: {len(folds)}")
    print("=" * 70 + "\n")


def validate_no_overlap(
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """
    Validate that train and val masks don't overlap within each fold.

    Parameters
    ----------
    folds : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_mask, val_mask).

    Returns
    -------
    bool
        True if no overlap exists in any fold.

    Raises
    ------
    ValueError
        If any overlap is detected.
    """
    for i, (train_mask, val_mask) in enumerate(folds):
        overlap = np.sum(train_mask & val_mask)
        if overlap > 0:
            raise ValueError(
                f"Fold {i}: Found {overlap} overlapping rows between train and val!"
            )
    return True


def validate_temporal_order(
    df: pd.DataFrame,
    date_col: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """
    Validate that training data always precedes validation data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    date_col : str
        Name of the date column.
    folds : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_mask, val_mask).

    Returns
    -------
    bool
        True if temporal order is respected in all folds.

    Raises
    ------
    ValueError
        If any fold violates temporal order.
    """
    for i, (train_mask, val_mask) in enumerate(folds):
        train_max = df.loc[train_mask, date_col].max()
        val_min = df.loc[val_mask, date_col].min()

        if train_max >= val_min:
            raise ValueError(
                f"Fold {i}: Training max date ({train_max}) >= "
                f"Validation min date ({val_min})!"
            )
    return True
