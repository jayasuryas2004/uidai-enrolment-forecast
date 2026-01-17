"""
src/validation/__init__.py

Time-series validation utilities for UIDAI forecasting.
"""

from .time_series_cv import generate_expanding_folds

__all__ = ["generate_expanding_folds"]
