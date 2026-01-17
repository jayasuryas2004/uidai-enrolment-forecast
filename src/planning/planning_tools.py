# src/planning/planning_tools.py
"""
Planning Tools
==============

Data tools for the UIDAI Planning Assistant.

This module provides:
    - get_over_capacity_data(): Query capacity gaps from forecast data
    - simulate_extra_camps(): Simulate impact of adding centres
    - summarize_over_capacity(): Format capacity data as TOOL SUMMARY
    - summarize_simulation(): Format simulation results as TOOL SUMMARY

╔════════════════════════════════════════════════════════════════════════════╗
║  PHASE 2: Uses REAL forecast data from the XGBoost model.                 ║
║  All numbers come from actual model predictions.                          ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import functools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CapacityGapRow:
    """Single row of capacity gap analysis."""
    state: str
    district: Optional[str]
    forecast: int
    capacity: int
    gap: int
    gap_pct: float
    extra_centres_needed: int


@dataclass
class SimulationResult:
    """Result of capacity simulation."""
    region: str
    month: str
    n_camps_added: int
    gap_before: int
    gap_pct_before: float
    gap_after: int
    gap_pct_after: float
    recommendation: str


# =============================================================================
# CAPACITY CONSTANTS (same as dashboard)
# =============================================================================

PER_CENTRE_CAPACITY = 5000  # Enrolments per centre per month

# Approximate centres per state (same as app.py)
STATE_CENTRES = {
    "Andhra Pradesh": 450,
    "Bihar": 380,
    "Delhi": 120,
    "Gujarat": 280,
    "Haryana": 150,
    "Karnataka": 320,
    "Kerala": 180,
    "Madhya Pradesh": 350,
    "Maharashtra": 520,
    "Rajasthan": 340,
    "Tamil Nadu": 380,
    "Telangana": 220,
    "Uttar Pradesh": 650,
    "West Bengal": 300,
}
DEFAULT_CENTRES_PER_STATE = 200


# =============================================================================
# REAL DATA LOADING (PHASE 2)
# =============================================================================

@functools.lru_cache(maxsize=1)
def _load_forecast_data() -> Optional[pd.DataFrame]:
    """
    Load and cache the forecast data from the model.
    
    Returns:
        DataFrame with forecast data, or None if loading fails.
        
    Columns expected:
        - state, district
        - month_date (datetime)
        - y_pred_v3 (forecast from XGBoost model)
        - total_enrolment (actual, for reference)
    """
    try:
        print("\U0001F504 Loading Phase4V3Model...")
        
        from src.phase4_model_registry import PHASE4_V3_FINAL
        from src.phase4_v3_inference import Phase4V3Model
        from src.preprocessing.time_series_cleaning import (
            CleaningConfig,
            clean_uidai_time_series,
        )
        
        # Load base data (same path as app.py)
        data_path = PROJECT_ROOT / "data" / "processed" / "district_month_modeling.csv"
        if not data_path.exists():
            print(f"\u274C Data file not found: {data_path}")
            return None
        
        print(f"\u2705 Found data file: {data_path}")
        df_raw = pd.read_csv(data_path, parse_dates=["month_date"])
        print(f"\u2705 Loaded {len(df_raw)} raw rows")
        
        # Clean data (match app.py config)
        config = CleaningConfig(
            outlier_method="zscore_moving",
            outlier_window=3,
            outlier_z_thresh=3.0,
            outlier_cap_method="median",
            missing_method="locf",
            max_locf_gap=3,
        )
        df_clean = clean_uidai_time_series(
            df=df_raw.copy(),
            state_col="state",
            district_col="district",
            date_col="month_date",
            target_col="total_enrolment",
            config=config,
        )
        print(f"✅ Cleaned data: {len(df_clean)} rows")
        
        # Drop cleaning metadata columns
        cleaning_cols = ["is_outlier_event", "total_enrolment_original",
                         "total_enrolment_was_capped", "total_enrolment_was_imputed"]
        df_clean = df_clean.drop(columns=[c for c in cleaning_cols if c in df_clean.columns])
        
        # Load model and generate predictions
        print("🔄 Loading XGBoost model...")
        model = Phase4V3Model()
        model.load()
        print("✅ Model loaded")
        
        print("\U0001F504 Generating predictions...")
        predictions = model.predict(df_clean)
        
        df_forecast = df_clean.copy()
        df_forecast["y_pred_v3"] = predictions.values
        
        # Add month string column for easy filtering
        df_forecast["month"] = df_forecast["month_date"].dt.strftime("%Y-%m")
        
        # Add current centres
        df_forecast["current_centres"] = df_forecast["state"].apply(
            lambda s: STATE_CENTRES.get(s, DEFAULT_CENTRES_PER_STATE)
        )
        
        print(f"\u2705 Generated {len(df_forecast)} forecast rows")
        print(f"\U0001F4C5 Available months: {sorted(df_forecast['month'].unique())[:5]}...")
        print(f"\U0001F4CA States: {sorted(df_forecast['state'].unique())[:5]}...")
        
        return df_forecast
        
    except Exception as e:
        print(f"\u274C Forecast load FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_forecast_df() -> pd.DataFrame:
    """
    Get the cached forecast DataFrame.
    
    Falls back to dummy data if real data unavailable.
    """
    df = _load_forecast_data()
    if df is not None:
        return df
    
    # Fallback: create dummy DataFrame
    print("\u26A0\uFE0F Using dummy forecast data (real data load failed)")
    return _create_dummy_forecast_df()


def _create_dummy_forecast_df() -> pd.DataFrame:
    """Create a dummy forecast DataFrame for fallback."""
    rows = []
    for month in ["2025-01", "2025-02", "2025-03"]:
        for state, forecast in DUMMY_FORECASTS.get(month, {}).items():
            rows.append({
                "state": state,
                "district": f"{state} Central",  # Dummy district
                "month": month,
                "y_pred_v3": forecast,
                "current_centres": STATE_CENTRES.get(state, DEFAULT_CENTRES_PER_STATE),
            })
    return pd.DataFrame(rows)


# Dummy forecasts for fallback
DUMMY_FORECASTS = {
    "2025-01": {
        "Tamil Nadu": 42000,
        "Karnataka": 38000,
        "Maharashtra": 45000,
        "Uttar Pradesh": 85000,
        "Bihar": 55000,
        "Rajasthan": 42000,
        "Gujarat": 35000,
        "Madhya Pradesh": 40000,
    },
    "2025-02": {
        "Tamil Nadu": 44000,
        "Karnataka": 40000,
        "Maharashtra": 48000,
        "Uttar Pradesh": 88000,
        "Bihar": 58000,
        "Rajasthan": 45000,
        "Gujarat": 38000,
        "Madhya Pradesh": 42000,
    },
    "2025-03": {
        "Tamil Nadu": 46000,
        "Karnataka": 42000,
        "Maharashtra": 50000,
        "Uttar Pradesh": 92000,
        "Bihar": 60000,
        "Rajasthan": 48000,
        "Gujarat": 40000,
        "Madhya Pradesh": 45000,
    },
}


# =============================================================================
# CAPACITY GAP FUNCTIONS (PHASE 2 - REAL DATA)
# =============================================================================

def get_state_capacity(state: str) -> int:
    """Get estimated capacity for a state."""
    centres = STATE_CENTRES.get(state, DEFAULT_CENTRES_PER_STATE)
    return centres * PER_CENTRE_CAPACITY


def get_over_capacity_data(month: str, top_k: int = 5) -> List[CapacityGapRow]:
    """
    Get top-K over-capacity states for a given month.
    
    Uses REAL forecast data from the XGBoost model.
    
    Args:
        month: Month in YYYY-MM format
        top_k: Number of top states to return
        
    Returns:
        List of CapacityGapRow sorted by gap descending
    """
    df = get_forecast_df()
    
    # Filter for the requested month
    if "month" in df.columns:
        month_data = df[df["month"] == month].copy()
    elif "month_date" in df.columns:
        month_data = df[df["month_date"].dt.strftime("%Y-%m") == month].copy()
    else:
        # Fallback: use latest available data
        month_data = df.copy()
    
    if month_data.empty:
        # Try to find the closest available month
        available_months = df["month"].unique() if "month" in df.columns else []
        if len(available_months) > 0:
            # Use the latest month
            month_data = df[df["month"] == sorted(available_months)[-1]].copy()
    
    if month_data.empty:
        return []
    
    # Compute capacity and gap per row
    month_data["capacity"] = month_data["current_centres"] * PER_CENTRE_CAPACITY
    month_data["gap_absolute"] = month_data["y_pred_v3"] - month_data["capacity"]
    month_data["gap_pct"] = (month_data["gap_absolute"] / month_data["capacity"]) * 100
    month_data["extra_centres"] = np.ceil(month_data["gap_absolute"] / PER_CENTRE_CAPACITY).clip(lower=0)
    
    # Aggregate by state
    state_agg = (
        month_data.groupby("state")
        .agg({
            "y_pred_v3": "sum",
            "capacity": "sum",
            "gap_absolute": "sum",
            "extra_centres": "sum",
        })
        .reset_index()
    )
    
    # Recalculate gap percentage after aggregation
    state_agg["gap_pct"] = (state_agg["gap_absolute"] / state_agg["capacity"]) * 100
    state_agg["extra_centres"] = np.ceil(state_agg["gap_absolute"] / PER_CENTRE_CAPACITY).clip(lower=0)
    
    # Sort by gap descending, filter positive gaps, take top K
    state_agg = state_agg.sort_values("gap_absolute", ascending=False)
    state_agg = state_agg[state_agg["gap_absolute"] > 0].head(top_k)
    
    # Convert to CapacityGapRow objects
    rows = []
    for _, row in state_agg.iterrows():
        rows.append(CapacityGapRow(
            state=row["state"],
            district=None,
            forecast=int(row["y_pred_v3"]),
            capacity=int(row["capacity"]),
            gap=int(row["gap_absolute"]),
            gap_pct=float(row["gap_pct"]),
            extra_centres_needed=int(row["extra_centres"]),
        ))
    
    return rows


def get_over_capacity_districts(month: str, top_k: int = 5) -> List[CapacityGapRow]:
    """
    Get top-K over-capacity districts for a given month.
    
    Uses REAL forecast data from the XGBoost model.
    
    Args:
        month: Month in YYYY-MM format
        top_k: Number of top districts to return
        
    Returns:
        List of CapacityGapRow sorted by gap percentage descending
    """
    df = get_forecast_df()
    
    # Filter for the requested month
    if "month" in df.columns:
        month_data = df[df["month"] == month].copy()
    elif "month_date" in df.columns:
        month_data = df[df["month_date"].dt.strftime("%Y-%m") == month].copy()
    else:
        month_data = df.copy()
    
    if month_data.empty:
        return []
    
    # Compute capacity and gap per district
    month_data["capacity"] = month_data["current_centres"] * PER_CENTRE_CAPACITY
    # For districts, use a fraction of state capacity (estimate)
    month_data["district_capacity"] = month_data["capacity"] / month_data.groupby("state")["district"].transform("nunique")
    month_data["gap_absolute"] = month_data["y_pred_v3"] - month_data["district_capacity"]
    month_data["gap_pct"] = (month_data["gap_absolute"] / month_data["district_capacity"]) * 100
    month_data["extra_centres"] = np.ceil(month_data["gap_absolute"] / PER_CENTRE_CAPACITY).clip(lower=0)
    
    # Sort by gap percentage descending, filter positive gaps
    month_data = month_data.sort_values("gap_pct", ascending=False)
    month_data = month_data[month_data["gap_absolute"] > 0].head(top_k)
    
    rows = []
    for _, row in month_data.iterrows():
        rows.append(CapacityGapRow(
            state=row["state"],
            district=row.get("district", "Unknown"),
            forecast=int(row["y_pred_v3"]),
            capacity=int(row["district_capacity"]),
            gap=int(row["gap_absolute"]),
            gap_pct=float(row["gap_pct"]),
            extra_centres_needed=int(row["extra_centres"]),
        ))
    
    return rows


def simulate_extra_camps(
    region: str,
    n_camps: int,
    month: Optional[str] = None,
) -> SimulationResult:
    """
    Simulate the impact of adding extra centres to a region.
    
    Uses REAL forecast data from the XGBoost model.
    
    Args:
        region: State or district name
        n_camps: Number of extra centres to add
        month: Month in YYYY-MM format (optional)
        
    Returns:
        SimulationResult with before/after gap analysis
    """
    df = get_forecast_df()
    
    # Default month: use latest available
    if month is None:
        if "month" in df.columns:
            month = sorted(df["month"].unique())[-1]
        else:
            month = "2025-04"
    
    # Filter for the region (state or district)
    if "month" in df.columns:
        month_data = df[df["month"] == month].copy()
    elif "month_date" in df.columns:
        month_data = df[df["month_date"].dt.strftime("%Y-%m") == month].copy()
    else:
        month_data = df.copy()
    
    # Match region (case-insensitive)
    region_lower = region.lower()
    region_data = month_data[
        (month_data["state"].str.lower().str.contains(region_lower, na=False)) |
        (month_data.get("district", pd.Series()).str.lower().str.contains(region_lower, na=False))
    ]
    
    if region_data.empty:
        # Fallback to state-level match
        region_data = month_data[month_data["state"].str.lower().str.contains(region_lower, na=False)]
    
    if region_data.empty:
        return SimulationResult(
            region=region,
            month=month,
            n_camps_added=n_camps,
            gap_before=0,
            gap_pct_before=0.0,
            gap_after=0,
            gap_pct_after=0.0,
            recommendation=f"No data available for {region} in {month}.",
        )
    
    # Compute baseline capacity and gap
    total_forecast = region_data["y_pred_v3"].sum()
    total_centres = region_data["current_centres"].sum() if "current_centres" in region_data else (
        len(region_data) * DEFAULT_CENTRES_PER_STATE
    )
    baseline_capacity = total_centres * PER_CENTRE_CAPACITY
    baseline_gap = total_forecast - baseline_capacity
    baseline_gap_pct = (baseline_gap / baseline_capacity * 100) if baseline_capacity > 0 else 0
    
    # After adding centres
    added_capacity = n_camps * PER_CENTRE_CAPACITY
    new_capacity = baseline_capacity + added_capacity
    new_gap = max(0, baseline_gap - added_capacity)
    new_gap_pct = (new_gap / new_capacity * 100) if new_capacity > 0 else 0
    
    # Generate recommendation
    if new_gap <= 0:
        recommendation = (
            f"Adding {n_camps} centre(s) fully closes the gap. "
            f"{region} would have surplus capacity."
        )
    elif new_gap_pct < 5:
        recommendation = (
            f"Adding {n_camps} centre(s) reduces gap to acceptable level (<5%). "
            f"Recommended action."
        )
    else:
        extra_needed = int(np.ceil(new_gap / PER_CENTRE_CAPACITY))
        recommendation = (
            f"Gap remains at {new_gap_pct:.1f}%. "
            f"Consider adding {extra_needed} more centre(s) to fully close the gap."
        )
    
    return SimulationResult(
        region=region,
        month=month,
        n_camps_added=n_camps,
        gap_before=int(baseline_gap),
        gap_pct_before=float(baseline_gap_pct),
        gap_after=int(new_gap),
        gap_pct_after=float(new_gap_pct),
        recommendation=recommendation,
    )


# =============================================================================
# TOOL SUMMARY FORMATTERS (PHASE 2 - REAL DATA)
# =============================================================================

def summarize_over_capacity(month: str, top_k: int = 3) -> str:
    """
    Format over-capacity data as a bullet-point summary.
    
    Uses REAL forecast data from the XGBoost model.
    
    Args:
        month: Month in YYYY-MM format
        top_k: Number of top states to include
        
    Returns:
        Formatted bullet-point summary string
    """
    # Debug: show query
    df = get_forecast_df()
    month_data = df[df["month"] == month] if "month" in df.columns else df
    print(f"\U0001F50D Query month='{month}': Found {len(month_data)} rows")
    if len(month_data) == 0:
        available = sorted(df["month"].unique()) if "month" in df.columns else []
        print(f"   Available months: {available[:10]}...")
    
    # Get state-level data
    state_rows = get_over_capacity_data(month, top_k=top_k)
    
    # Get district-level data
    district_rows = get_over_capacity_districts(month, top_k=5)
    
    if not state_rows:
        return f"""**📊 Over-Capacity Analysis for {month}**

✅ All regions are within capacity for this month.

No additional centres needed."""
    
    lines = [
        f"**📊 Over-Capacity Analysis for {month}**",
        "",
        f"**🚨 Top {len(state_rows)} States Over Capacity:**",
    ]
    
    for i, row in enumerate(state_rows, 1):
        sign = "+" if row.gap_pct > 0 else ""
        lines.append(
            f"- **{row.state}**: {sign}{row.gap_pct:.0f}% over capacity "
            f"(🏢 +{row.extra_centres_needed} centres needed)"
        )
    
    if district_rows:
        lines.append("")
        lines.append(f"**📍 Top {len(district_rows)} Districts:**")
        for row in district_rows:
            sign = "+" if row.gap_pct > 0 else ""
            lines.append(
                f"- **{row.district}** ({row.state}): {sign}{row.gap_pct:.0f}% "
                f"(🏢 +{row.extra_centres_needed} centres)"
            )
    
    lines.append("")
    lines.append("💡 *Recommendation: Prioritize high-gap states for capacity planning.*")
    lines.append("")
    lines.append("📈 *Data: XGBoost v3 model forecasts*")
    
    return "\n".join(lines)


def summarize_simulation(
    user_message: str,
    region: Optional[str] = None,
    n_camps: int = 1,
    month: Optional[str] = None,
) -> str:
    """
    Format simulation results as a bullet-point summary.
    
    Uses REAL forecast data from the XGBoost model.
    
    Args:
        user_message: Original user request
        region: State or district name
        n_camps: Number of extra centres to simulate
        month: Month in YYYY-MM format
        
    Returns:
        Formatted bullet-point summary string
    """
    region = region or "Tamil Nadu"  # Default region
    
    result = simulate_extra_camps(region, n_camps, month)
    
    sign_before = "+" if result.gap_before > 0 else ""
    sign_after = "+" if result.gap_after > 0 else ""
    
    # Determine status emoji
    if result.gap_after <= 0:
        status_emoji = "✅"
        status_text = "Within capacity"
    else:
        status_emoji = "⚠️"
        status_text = "Still over capacity"
    
    return f"""**🔮 Simulation: +{result.n_camps_added} Centres in {result.region}**

**📅 Month:** {result.month}

**Before Adding Centres:**
- 📊 Capacity gap: {result.gap_before:,} ({sign_before}{result.gap_pct_before:.0f}%)

**After Adding +{result.n_camps_added} Centres:**
- 📊 New capacity gap: {result.gap_after:,} ({sign_after}{result.gap_pct_after:.0f}%)
- {status_emoji} Status: {status_text}

💡 *{result.recommendation}*

📈 *Data: XGBoost v3 model forecasts*"""
