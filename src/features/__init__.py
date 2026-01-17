"""
src/features/__init__.py

Time-series feature engineering utilities for UIDAI forecasting.

This package provides leakage-safe feature engineering functions:
    - ACF/PACF-guided lag selection
    - Rolling statistics
    - Holiday/festival features
    - Policy/intervention features
"""

from .timeseries_lag_utils import (
    compute_acf_pacf_for_district,
    select_significant_lags,
    get_recommended_lags,
)
from .timeseries_features import (
    add_lag_features,
    add_rolling_features,
    add_all_timeseries_features,
)
from .holiday_features import (
    build_holiday_calendar,
    add_holiday_features,
)
from .policy_features import (
    add_policy_phase_features,
)

__all__ = [
    # Lag utilities
    "compute_acf_pacf_for_district",
    "select_significant_lags",
    "get_recommended_lags",
    # Timeseries features
    "add_lag_features",
    "add_rolling_features",
    "add_all_timeseries_features",
    # Holiday features
    "build_holiday_calendar",
    "add_holiday_features",
    # Policy features
    "add_policy_phase_features",
]
