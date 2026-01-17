"""
preprocessing
=============

Data preprocessing utilities for UIDAI time-series forecasting.

This package provides leakage-safe cleaning and preprocessing functions
for district-level monthly enrolment data.

Modules:
    time_series_cleaning: Outlier detection and missing data handling

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
        df=df,
        state_col="state",
        district_col="district",
        date_col="month_date",
        target_col="total_enrolment",
        config=config,
    )
"""

from src.preprocessing.time_series_cleaning import (
    CleaningConfig,
    OutlierMethod,
    MissingMethod,
    clean_uidai_time_series,
    detect_outliers_moving_zscore,
    impute_missing_values,
    summarize_missing_by_district,
)

__all__ = [
    "CleaningConfig",
    "OutlierMethod",
    "MissingMethod",
    "clean_uidai_time_series",
    "detect_outliers_moving_zscore",
    "impute_missing_values",
    "summarize_missing_by_district",
]
