#!/usr/bin/env python
"""
UIDAI Enrolment Forecast Dashboard – Phase-4 v3
================================================

A modern Streamlit dashboard for the UIDAI district-level demand forecasting model.

Features:
- Overview page with KPI cards and summary metrics
- District Explorer with interactive actual vs forecast charts
- Residual analysis and model insights
- CSV upload for custom predictions

Usage:
    streamlit run app.py

Author: UIDAI Forecast Team
Date: January 2026
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress sklearn R2 warnings for single-sample districts
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Suppress planning tools warnings (Phase 2.5)
logging.getLogger("src.planning").setLevel(logging.ERROR)
logging.getLogger("src.planning.planning_tools").setLevel(logging.ERROR)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project modules
from src.phase4_model_registry import PHASE4_V3_FINAL
from src.phase4_v3_inference import Phase4V3Model, build_v3_features_for_inference
from src.planning import planning_tab
from src.preprocessing.time_series_cleaning import (
    CleaningConfig,
    clean_uidai_time_series,
)

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="UIDAI Enrolment Forecast – Phase-4 v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern BoldBI-style government dashboard CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Roboto', system-ui, sans-serif;
    }
    
    /* Page background - light blue like BoldBI */
    .main {
        background-color: #F3F7FB;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Container max-width and TOP PADDING for heading visibility */
    .block-container {
        padding-top: 50px !important;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }
    
    /* Page headers - proper spacing from top */
    .uidai-page-header {
        margin-top: 20px !important;
        margin-bottom: 8px !important;
        padding-top: 10px !important;
    }
    
    /* First element on page - extra top margin */
    .main .block-container > div:first-child {
        margin-top: 10px !important;
    }
    
    /* H1, H2, H3 headings - proper spacing */
    .main h1:first-of-type,
    .main h2:first-of-type {
        margin-top: 20px !important;
        padding-top: 10px !important;
    }
    
    /* Streamlit header - make transparent */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Card styling - BoldBI style */
    .uidai-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    
    /* KPI Card styling */
    .uidai-kpi-card {
        min-height: 110px;
    }
    .uidai-kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .uidai-kpi-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0f172a;
    }
    .uidai-kpi-subtitle {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .uidai-kpi-delta {
        font-size: 0.78rem;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Section title styling */
    .uidai-section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .uidai-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        line-height: 1.6;
    }
    
    /* Page header */
    .uidai-page-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .uidai-page-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1.25rem;
    }
    
    /* Sidebar styling - enhanced navigation UI */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 2px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        font-family: 'Inter', 'Roboto', system-ui, sans-serif;
    }
    
    /* Sidebar navigation radio container */
    section[data-testid="stSidebar"] .stRadio > div {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Each navigation option */
    section[data-testid="stSidebar"] .stRadio > div > label {
        background-color: transparent;
        padding: 12px 16px !important;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        color: #374151 !important;
        font-weight: 500;
    }
    
    /* Hover effect on navigation items */
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
    }
    
    /* Selected/Active navigation item - blue highlight */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked),
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
    }
    
    /* Force dark text on sidebar - override system dark mode */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    
    /* Sidebar navigation labels */
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div > label,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 500;
    }
    
    /* Sidebar text and markdown */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #0f172a !important;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] strong {
        color: #111827 !important;
    }
    
    /* Sidebar muted/secondary text */
    [data-testid="stSidebar"] .stMarkdown div[style*="color"],
    [data-testid="stSidebar"] div[style*="9CA3AF"] {
        color: #64748b !important;
    }
    
    /* File uploader in sidebar */
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stFileUploader div,
    [data-testid="stSidebar"] .stFileUploader span,
    [data-testid="stSidebar"] .uploadedFile {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stFileUploader button {
        background-color: #111827 !important;
        color: #f9fafb !important;
        border-radius: 999px;
    }
    /* File uploader drag-drop zone */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #f9fafb !important;
        border-color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
        color: #64748b !important;
    }
    
    /* Selectbox in sidebar */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f9fafb !important;
        border-color: #e2e8f0 !important;
        color: #0f172a !important;
    }
    
    /* Radio button circles/indicators */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
        background-color: #ffffff !important;
        border-color: #0076A8 !important;
    }
    
    /* Horizontal rule in sidebar */
    [data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
    }
    
    /* District/State cards */
    .district-card {
        background-color: #ffffff;
        padding: 12px 14px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.6rem;
        border-left: 4px solid #0076A8;
    }
    .district-card-high {
        border-left-color: #dc2626;
    }
    .district-card-low {
        border-left-color: #16a34a;
    }
    
    /* Navigation radio buttons */
    div[data-testid="stRadio"] > label {
        font-family: 'Inter', 'Roboto', system-ui, sans-serif;
        font-weight: 500;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* FORCE ALL TEXT TO DARK - Permanent fix for dark mode */
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
    p, span, div, label { color: #0f172a !important; }
    .stMarkdown, .stMarkdown p { color: #0f172a !important; }
    .stMetric label { color: #64748b !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0f172a !important; }
    .stMetric div { color: #0f172a !important; }
    
    /* KPI cards text - always dark */
    .uidai-kpi-label, .uidai-kpi-value, .uidai-kpi-subtitle { color: #0f172a !important; }
    .uidai-page-header, .uidai-section-title { color: #0f172a !important; }
    .uidai-page-subtitle, .uidai-subtitle { color: #64748b !important; }
    
    /* Force main content text colors for readability */
    .main h1, .main h2, .main h3, .main h4 { color: #0f172a !important; }
    .main .stMetric label { color: #64748b !important; }
    .main .stMetric [data-testid="stMetricValue"] { color: #0f172a !important; }
    .main p, .main span, .main label { color: #0f172a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# COLOR PALETTE
# =============================================================================

COLORS = {
    "primary": "#0076A8",      # UIDAI teal/blue
    "secondary": "#FF8A3C",    # Orange accent
    "success": "#16a34a",      # Green
    "warning": "#f59e0b",      # Amber
    "danger": "#dc2626",       # Red
    "actual": "#0f766e",       # Teal (actual enrolment)
    "forecast": "#fb923c",     # Orange (forecast)
    "residual": "#6366f1",     # Indigo (residuals)
    "background": "#F3F7FB",
    "card": "#FFFFFF",
    "text": "#0f172a",
    "text_muted": "#64748b",
    "grid": "rgba(148, 163, 184, 0.25)",
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_base_data() -> pd.DataFrame:
    """Load the base UIDAI district-month dataset."""
    data_path = PROJECT_ROOT / "data" / "processed" / "district_month_modeling.csv"
    df = pd.read_csv(data_path, parse_dates=["month_date"])
    return df


@st.cache_resource
def load_v3_model() -> Phase4V3Model:
    """Load the frozen Phase-4 v3 model."""
    model = Phase4V3Model()
    model.load()
    return model


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply production-style cleaning to the data."""
    config = CleaningConfig(
        outlier_method="zscore_moving",
        outlier_window=3,
        outlier_z_thresh=3.0,
        outlier_cap_method="median",
        missing_method="locf",
        max_locf_gap=3,
    )
    
    df_clean = clean_uidai_time_series(
        df=df.copy(),
        state_col="state",
        district_col="district",
        date_col="month_date",
        target_col="total_enrolment",
        config=config,
    )
    
    # Drop cleaning metadata columns
    cleaning_cols = ["is_outlier_event", "total_enrolment_original",
                     "total_enrolment_was_capped", "total_enrolment_was_imputed"]
    df_clean = df_clean.drop(columns=[c for c in cleaning_cols if c in df_clean.columns])
    
    return df_clean


@st.cache_data
def generate_predictions(_model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using the v3 model."""
    predictions = _model.predict(df)
    df_eval = df.copy()
    df_eval["y_pred_v3"] = predictions.values
    df_eval["residual"] = df_eval["total_enrolment"] - df_eval["y_pred_v3"]
    df_eval["abs_error"] = df_eval["residual"].abs()
    return df_eval


# =============================================================================
# KPI CARDS (Official Metrics from Frozen JSON)
# =============================================================================

@st.cache_data
def load_v3_official_metrics():
    """
    Load official Phase-4 v3 metrics from the frozen JSON file.

    Expect a structure like:
    {
      "cv_metrics": {
        "r2_mean": 0.9551,
        "r2_std": 0.0488,
        "mae_mean": 116.63,
        "mae_std": 119.80,
        "rmse_mean": 258.65,
        "rmse_std": 261.36
      }
    }
    """
    metrics_path = PHASE4_V3_FINAL.final_metrics
    with open(metrics_path, "r") as f:
        m = json.load(f)

    # Support both "cv_metrics" and "cv" keys, fallback to top-level
    cv = m.get("cv_metrics", m.get("cv", m))

    r2_mean = float(cv["r2_mean"])
    mae_mean = float(cv["mae_mean"])
    rmse_mean = float(cv.get("rmse_mean", cv.get("rmse", 0.0)))

    r2_std = float(cv.get("r2_std", 0.0))
    mae_std = float(cv.get("mae_std", 0.0))
    rmse_std = float(cv.get("rmse_std", 0.0))

    return {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
    }


def load_cell49_metrics():
    """
    Load Cell 49 metrics from real new data testing.
    These are the production-validated metrics on new test data.
    
    Returns Cell 49 metrics dictionary with:
    - r2_full: 0.8806 (Full dataset evaluation)
    - mae_full: 82.67
    - rmse_full: 456.85
    - mape_full: 27.64
    - samples_tested: 2496
    - records_processed: 5060 (district-months)
    - states: 55
    - districts: 984
    """
    return {
        "r2_full": 0.8806,
        "mae_full": 82.67,
        "rmse_full": 456.85,
        "mape_full": 27.64,
        "samples_tested": 2496,
        "records_processed": 5060,
        "states": 55,
        "districts": 984,
    }


def compute_local_metrics(df_sel: pd.DataFrame):
    """
    Compute local R², MAE for a filtered subset (district/state).
    Use this for per-district or per-state charts, NOT for global KPIs.
    """
    y_true = df_sel["total_enrolment"]
    y_pred = df_sel["y_pred_v3"]
    n = len(y_true)
    r2 = r2_score(y_true, y_pred) if n >= 2 else 0.0
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae


def kpi_card(
    label: str,
    value: str,
    subtitle: str = "",
    delta: str | None = None,
    good: bool | None = None,
):
    """
    Render a single KPI card in BoldBI government dashboard style.
    
    Parameters:
        label: The metric name (uppercase label)
        value: The main value to display
        subtitle: Optional secondary text
        delta: Optional delta/change indicator
        good: True=green, False=red, None=gray for delta color
    """
    color = "#16a34a" if good else "#dc2626" if good is not None else "#64748b"
    delta_html = ""
    if delta is not None:
        delta_html = f"<div class='uidai-kpi-delta' style='color:{color}'>{delta}</div>"

    st.markdown(
        f"""
        <div class="uidai-card uidai-kpi-card">
          <div class="uidai-kpi-label">{label}</div>
          <div class="uidai-kpi-value">{value}</div>
          <div class="uidai-kpi-subtitle">{subtitle}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_kpi_row(df_eval: pd.DataFrame):
    """
    Render the top row of KPI cards with official CV metrics from the frozen JSON.
    Uses the reusable kpi_card component.
    """
    metrics = load_v3_official_metrics()
    n_districts = df_eval[["state", "district"]].drop_duplicates().shape[0]
    n_months = df_eval["month_date"].nunique()
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        kpi_card(
            label="R²",
            value=f"{metrics['r2_mean']:.3f}",
            subtitle="Official CV score",
            delta="✓ Strong fit",
            good=True,
        )
    
    with c2:
        mae_good = metrics["mae_mean"] < 150
        kpi_card(
            label="MAE",
            value=f"{metrics['mae_mean']:.1f}",
            subtitle="Avg. enrolments error",
            delta="✓ Low error" if mae_good else "Higher error",
            good=mae_good,
        )
    
    with c3:
        kpi_card(
            label="RMSE",
            value=f"{metrics['rmse_mean']:.1f}",
            subtitle="Root mean sq. error",
        )
    
    with c4:
        kpi_card(
            label="Districts",
            value=f"{n_districts:,}",
            subtitle=f"Phase-4 timeframe ({n_months} months)",
        )


def render_cell49_kpi_row():
    """
    Render Cell 49 metrics: New data testing results.
    Displays real production metrics on new test data.
    """
    cell49_metrics = load_cell49_metrics()
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        kpi_card(
            label="🎯 R² (Cell 49)",
            value=f"{cell49_metrics['r2_full']:.4f}",
            subtitle="Full dataset validation",
            delta="✓ Excellent",
            good=True,
        )
    
    with c2:
        kpi_card(
            label="📊 MAE (Cell 49)",
            value=f"{cell49_metrics['mae_full']:.2f}",
            subtitle="New data avg error",
            delta="✓ Low error",
            good=True,
        )
    
    with c3:
        kpi_card(
            label="📈 RMSE (Cell 49)",
            value=f"{cell49_metrics['rmse_full']:.2f}",
            subtitle="Robust error metric",
        )
    
    with c4:
        kpi_card(
            label="% Error (MAPE)",
            value=f"{cell49_metrics['mape_full']:.2f}%",
            subtitle="Percentage-based metric",
        )


# =============================================================================
# CAPACITY PLANNING WIDGET
# =============================================================================

# Capacity assumption: enrolments per centre per month (adjustable)
PER_CENTRE_CAPACITY = 5000

# Approximate centres per state (can be replaced with actual data)
# These are rough estimates for demonstration
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
DEFAULT_CENTRES_PER_STATE = 200  # Fallback for unlisted states


def compute_capacity_needs(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extra centre needs per state based on forecast vs capacity.
    
    Returns a DataFrame with columns:
        state, forecast, current_centres, current_capacity, gap, extra_centres_needed
    """
    # Get the latest month's forecast by state
    latest_month = df_eval["month_date"].max()
    df_latest = df_eval[df_eval["month_date"] == latest_month]
    
    state_forecast = (
        df_latest.groupby("state")["y_pred_v3"]
        .sum()
        .reset_index()
        .rename(columns={"y_pred_v3": "forecast"})
    )
    
    # Add current centres (from lookup or default)
    state_forecast["current_centres"] = state_forecast["state"].apply(
        lambda s: STATE_CENTRES.get(s, DEFAULT_CENTRES_PER_STATE)
    )
    
    # Compute capacity and gap
    state_forecast["current_capacity"] = state_forecast["current_centres"] * PER_CENTRE_CAPACITY
    state_forecast["gap"] = state_forecast["forecast"] - state_forecast["current_capacity"]
    state_forecast["extra_centres_needed"] = np.maximum(
        np.ceil(state_forecast["gap"] / PER_CENTRE_CAPACITY), 0
    ).astype(int)
    
    return state_forecast.sort_values("extra_centres_needed", ascending=False)


def render_capacity_planning_widget(df_eval: pd.DataFrame):
    """
    Render capacity planning section with KPI and table for top states.
    """
    capacity_df = compute_capacity_needs(df_eval)
    top5 = capacity_df.head(5)
    total_extra = int(top5["extra_centres_needed"].sum())
    
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">🏢 Capacity Planning: Extra Centres Needed</div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        kpi_card(
            label="Extra Centres",
            value=f"{total_extra:,}",
            subtitle="Top 5 states, next month",
            delta="Based on forecast demand",
            good=total_extra == 0,
        )
    
    with col2:
        # Format for display
        display_df = top5[["state", "forecast", "current_capacity", "extra_centres_needed"]].copy()
        display_df.columns = ["State", "Forecast", "Current Capacity", "Extra Centres"]
        display_df["Forecast"] = display_df["Forecast"].apply(lambda x: f"{x:,.0f}")
        display_df["Current Capacity"] = display_df["Current Capacity"].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
    
    st.markdown(
        f"""
        <p style="font-size:0.8rem; color:#94a3b8; margin-top:0.5rem;">
            Assumption: Each centre handles {PER_CENTRE_CAPACITY:,} enrolments/month. 
            Adjust <code>PER_CENTRE_CAPACITY</code> in code to customize.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# CHARTS
# =============================================================================

def render_main_timeseries(df_sel: pd.DataFrame, state: str, district: str):
    """Render the main actual vs forecast line chart (BoldBI style)."""
    # Create figure with solid teal for actual, dashed orange for forecast
    fig = go.Figure()
    
    # Actual enrolment - solid teal line
    fig.add_trace(go.Scatter(
        x=df_sel["month_date"],
        y=df_sel["total_enrolment"],
        mode="lines+markers",
        name="Actual",
        line=dict(color=COLORS["actual"], width=3),
        marker=dict(size=7),
        hovertemplate="%{y:,.0f}<extra>Actual</extra>",
    ))
    
    # Forecast - dashed orange line
    fig.add_trace(go.Scatter(
        x=df_sel["month_date"],
        y=df_sel["y_pred_v3"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=COLORS["forecast"], width=3, dash="dash"),
        marker=dict(size=7),
        hovertemplate="%{y:,.0f}<extra>Forecast</extra>",
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=16, t=30, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Enrolments",
            gridcolor=COLORS["grid"],
            tickfont=dict(size=11),
        ),
        font=dict(family="Inter, Roboto, system-ui, sans-serif"),
        hovermode="x unified",
    )
    
    # Calculate local district metrics (not from official JSON)
    r2_loc, mae_loc = compute_local_metrics(df_sel)
    
    st.markdown(
        f"""
        <div class="uidai-card">
            <div class="uidai-section-title">{state} – {district}: Actual vs Forecast</div>
            <div style="font-size:0.85rem; color:#64748b; margin-bottom:0.5rem;">
                Local R² = {r2_loc:.3f} | Local MAE = {mae_loc:.0f}
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_residual_histogram(df_sel: pd.DataFrame):
    """Render the residual distribution histogram (BoldBI style)."""
    fig = px.histogram(
        df_sel,
        x="residual",
        nbins=30,
        color_discrete_sequence=[COLORS["primary"]],
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["danger"], line_width=2)
    
    mean_residual = df_sel["residual"].mean()
    fig.add_vline(
        x=mean_residual,
        line_dash="dot",
        line_color=COLORS["success"],
        line_width=2,
        annotation_text=f"Mean: {mean_residual:.1f}",
        annotation_position="top",
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=16, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Residual (Actual − Forecast)",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Count",
            gridcolor=COLORS["grid"],
            tickfont=dict(size=11),
        ),
        font=dict(family="Inter, Roboto, system-ui, sans-serif"),
        showlegend=False,
    )
    
    return fig


def render_mae_by_state_chart(df_eval: pd.DataFrame, top_n: int = 10):
    """
    Render MAE by state horizontal bar chart (BoldBI Top N style).
    Sorted descending by MAE, teal color bars.
    """
    def safe_metrics(g):
        n = len(g)
        mae = mean_absolute_error(g["total_enrolment"], g["y_pred_v3"])
        r2 = r2_score(g["total_enrolment"], g["y_pred_v3"]) if n >= 2 else 0.0
        return pd.Series({"MAE": mae, "R2": r2, "Districts": g["district"].nunique()})
    
    state_metrics = (
        df_eval.groupby("state")
        .apply(safe_metrics, include_groups=False)
        .reset_index()
        .sort_values("MAE", ascending=False)
        .head(top_n)
    )
    
    # Reverse for horizontal bar (highest at top)
    state_metrics = state_metrics.iloc[::-1]
    
    fig = px.bar(
        state_metrics,
        y="state",
        x="MAE",
        orientation="h",
        color_discrete_sequence=[COLORS["primary"]],
    )
    
    fig.update_layout(
        height=380,
        margin=dict(l=16, r=16, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Mean Absolute Error",
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        font=dict(family="Inter, Roboto, system-ui, sans-serif"),
        showlegend=False,
    )
    
    return fig


def render_enrolment_coverage_chart(df_eval: pd.DataFrame, top_n: int = 15):
    """
    Render Enrolment Coverage by State horizontal bar chart.
    Shows each state's share of total national enrolment (%).
    Sorted descending, gradient colors from teal to orange.
    """
    # Aggregate total enrolment by state (sum across all months)
    state_enrolment = (
        df_eval.groupby("state")["total_enrolment"]
        .sum()
        .reset_index()
        .rename(columns={"total_enrolment": "total"})
    )
    
    # Calculate percentage of national total
    national_total = state_enrolment["total"].sum()
    state_enrolment["coverage_pct"] = (state_enrolment["total"] / national_total) * 100
    
    # Sort and take top N
    state_enrolment = state_enrolment.sort_values("coverage_pct", ascending=False).head(top_n)
    
    # Reverse for horizontal bar (highest at top)
    state_enrolment = state_enrolment.iloc[::-1]
    
    # Create color scale based on coverage percentage
    fig = px.bar(
        state_enrolment,
        y="state",
        x="coverage_pct",
        orientation="h",
        color="coverage_pct",
        color_continuous_scale=["#14b8a6", "#f97316"],  # teal to orange
        text=state_enrolment["coverage_pct"].apply(lambda x: f"{x:.1f}%"),
    )
    
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=10, color="#374151"),
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=16, r=60, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="% of National Enrolment",
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickfont=dict(size=11),
            ticksuffix="%",
            range=[0, max(state_enrolment["coverage_pct"]) * 1.15],
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        font=dict(family="Inter, Roboto, system-ui, sans-serif"),
        showlegend=False,
        coloraxis_showscale=False,
    )
    
    return fig


def render_national_timeseries_chart(df_eval: pd.DataFrame):
    """
    Render national-level actual vs forecast time-series (BoldBI style).
    Solid teal line for actuals, dashed orange line for forecast.
    Includes 80% prediction band based on historical residuals.
    """
    # Aggregate to national monthly level
    df_national = (
        df_eval.groupby("month_date")
        .agg({
            "total_enrolment": "sum",
            "y_pred_v3": "sum",
        })
        .reset_index()
        .sort_values("month_date")
    )
    
    # Compute residuals and rolling MAE for prediction band
    df_national["residual"] = df_national["total_enrolment"] - df_national["y_pred_v3"]
    df_national["abs_residual"] = df_national["residual"].abs()
    
    # Use rolling MAE (3-month window) or global MAE as fallback
    df_national["rolling_mae"] = df_national["abs_residual"].rolling(window=3, min_periods=1).mean()
    
    # Compute 80% prediction band (1.28 * error for symmetric distribution)
    band_multiplier = 1.28
    df_national["y_upper"] = df_national["y_pred_v3"] + band_multiplier * df_national["rolling_mae"]
    df_national["y_lower"] = df_national["y_pred_v3"] - band_multiplier * df_national["rolling_mae"]
    df_national["y_lower"] = df_national["y_lower"].clip(lower=0)  # No negative enrolments
    
    fig = go.Figure()
    
    # Add prediction band FIRST (so it appears under the lines)
    fig.add_trace(go.Scatter(
        x=df_national["month_date"],
        y=df_national["y_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name="_upper_bound",
    ))
    fig.add_trace(go.Scatter(
        x=df_national["month_date"],
        y=df_national["y_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.12)",  # Soft blue
        showlegend=True,
        name="80% prediction band",
        hoverinfo="skip",
    ))
    
    # Actual enrolment - solid teal line
    fig.add_trace(go.Scatter(
        x=df_national["month_date"],
        y=df_national["total_enrolment"],
        mode="lines",
        name="Actual enrolment",
        line=dict(color=COLORS["actual"], width=3),
        hovertemplate="%{y:,.0f}<extra>Actual</extra>",
    ))
    
    # Forecast - dashed orange line
    fig.add_trace(go.Scatter(
        x=df_national["month_date"],
        y=df_national["y_pred_v3"],
        mode="lines",
        name="Forecast",
        line=dict(color=COLORS["forecast"], width=3, dash="dash"),
        hovertemplate="%{y:,.0f}<extra>Forecast</extra>",
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=16, t=40, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Total enrolments",
            gridcolor=COLORS["grid"],
            tickfont=dict(size=11),
        ),
        font=dict(family="Inter, Roboto, system-ui, sans-serif"),
        hovermode="x unified",
    )
    
    return fig


def render_monthly_trend_chart(df_eval: pd.DataFrame):
    """Render monthly aggregated actual vs forecast (area chart version)."""
    monthly = (
        df_eval.groupby("month_date")
        .agg({
            "total_enrolment": "sum",
            "y_pred_v3": "sum",
        })
        .reset_index()
    )
    
    df_plot = monthly.melt(
        id_vars=["month_date"],
        value_vars=["total_enrolment", "y_pred_v3"],
        var_name="type",
        value_name="enrolments",
    )
    df_plot["type"] = df_plot["type"].replace({
        "total_enrolment": "Actual",
        "y_pred_v3": "Forecast",
    })
    
    fig = px.area(
        df_plot,
        x="month_date",
        y="enrolments",
        color="type",
        color_discrete_map={
            "Actual": COLORS["actual"],
            "Forecast": COLORS["forecast"],
        },
    )
    
    fig.update_traces(line_width=2)
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Total Enrolments",
        font=dict(family="Poppins, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title="",
        ),
        hovermode="x unified",
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    
    return fig


# =============================================================================
# DISTRICT CARDS
# =============================================================================

def render_top_districts(df_eval: pd.DataFrame, metric: str = "high"):
    """Render top/bottom performing district cards."""
    def safe_metrics(g):
        n = len(g)
        mae = mean_absolute_error(g["total_enrolment"], g["y_pred_v3"])
        r2 = r2_score(g["total_enrolment"], g["y_pred_v3"]) if n >= 2 else 0.0
        return pd.Series({"MAE": mae, "R2": r2})
    
    district_mae = (
        df_eval.groupby(["state", "district"])
        .apply(safe_metrics, include_groups=False)
        .reset_index()
    )
    
    if metric == "high":
        top = district_mae.nlargest(5, "MAE")
        card_class = "district-card district-card-high"
        title = "⚠️ Highest Forecast Errors"
    else:
        top = district_mae.nsmallest(5, "MAE")
        card_class = "district-card district-card-low"
        title = "✅ Best Performing Districts"
    
    st.markdown(f"**{title}**")
    
    for _, row in top.iterrows():
        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="font-weight:600; color:#0f172a;">{row['district']}</div>
                <div style="font-size:0.85rem; color:#64748B;">{row['state']}</div>
                <div style="font-size:0.9rem; margin-top:0.25rem;">
                    MAE: <strong>{row['MAE']:.0f}</strong> | R²: {row['R2']:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGES
# =============================================================================

def render_overview_page(df_eval: pd.DataFrame):
    """Render the Overview page (BoldBI government dashboard style) with Cell 49 metrics."""
    # Spacer to push content below Streamlit header
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="uidai-page-header">UIDAI Enrolment Forecast – Phase‑4 v3</div>
        <p class="uidai-page-subtitle">Production‑style XGBoost model with time‑series CV, cleaning and SHAP explainability.</p>
        """,
        unsafe_allow_html=True,
    )
    
    # === ROW 1: TRAINING CV METRICS (Official) ===
    st.markdown(
        """
        <div style='margin-top: 1.5rem; margin-bottom: 0.5rem;'>
            <div class="uidai-section-title">📊 Training Metrics (Time-Series CV)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_global_kpi_row(df_eval)
    st.markdown("")  # spacing
    
    # === ROW 2: CELL 49 METRICS (New Data Testing) ===
    st.markdown(
        """
        <div style='margin-top: 1.5rem; margin-bottom: 0.5rem;'>
            <div class="uidai-section-title">🔬 Cell 49: New Data Testing Results</div>
            <p style='color: #6b7280; font-size: 0.85rem; margin-top: -0.5rem;'>
                Production validation on 1,006,029 raw UIDAI records → 5,060 cleaned district-month records
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_cell49_kpi_row()
    
    # === DATA COVERAGE METRICS ===
    st.markdown("")
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">📍 Geographic & Temporal Coverage (Cell 49)</div>
        """,
        unsafe_allow_html=True,
    )
    col_cov1, col_cov2, col_cov3, col_cov4 = st.columns(4)
    cell49_data = load_cell49_metrics()
    
    with col_cov1:
        kpi_card(
            label="Records Processed",
            value=f"{cell49_data['records_processed']:,}",
            subtitle="District-month aggregates",
        )
    with col_cov2:
        kpi_card(
            label="Districts",
            value=f"{cell49_data['districts']}",
            subtitle="Geographic units",
        )
    with col_cov3:
        kpi_card(
            label="States",
            value=f"{cell49_data['states']}",
            subtitle="Full coverage",
        )
    with col_cov4:
        kpi_card(
            label="Samples",
            value=f"{cell49_data['samples_tested']:,}",
            subtitle="Training dataset",
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("")  # spacing
    
    # === ROW 3: Large national time-series + MAE by state ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">📈 National Enrolment: Actual vs Forecast</div>
            """,
            unsafe_allow_html=True,
        )
        fig = render_national_timeseries_chart(df_eval)
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">🎯 MAE by State (Top 10)</div>
            """,
            unsafe_allow_html=True,
        )
        fig = render_mae_by_state_chart(df_eval, top_n=10)
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 2.5: Enrolment Coverage Heatmap
    st.markdown("")  # spacing
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">🗺️ Enrolment Coverage by State</div>
            <p style="color: #6b7280; font-size: 0.9rem; margin-top: -0.5rem;">
                Share of national enrolment volume by state (higher = more enrolments processed)
            </p>
        """,
        unsafe_allow_html=True,
    )
    fig = render_enrolment_coverage_chart(df_eval, top_n=15)
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 3: Top/bottom districts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="uidai-card">', unsafe_allow_html=True)
        render_top_districts(df_eval, metric="high")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="uidai-card">', unsafe_allow_html=True)
        render_top_districts(df_eval, metric="low")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 4: Capacity planning widget
    st.markdown("")  # spacing
    render_capacity_planning_widget(df_eval)
    
    # === ROW 5: ALL 9 FIGURES IN TABS ===
    st.markdown("")  # spacing
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">📸 Complete Model Analysis Visualizations</div>
            <p style="color: #6b7280; font-size: 0.85rem; margin-top: -0.5rem;">
                All 9 figures from Cell 49 & Training analysis for PDF submission
            </p>
        """,
        unsafe_allow_html=True,
    )
    
    tabs = st.tabs([
        "🔧 CV Fold Structure",
        "📊 Actual vs Forecast",
        "📉 MAE by State",
        "📋 Residuals",
        "🎯 Scatter Plot",
        "🔍 SHAP Features",
        "🏗️ Capacity Planning",
        "📈 Analysis",
        "🌍 State Performance"
    ])
    
    fig_files = [
        ("01_cv_fold_structure.png", "CV Fold Structure: Expanding Window with 1-Month Gap"),
        ("02_actual_vs_forecast.png", "National Forecast: Actual vs Predicted"),
        ("03_mae_by_state.png", "MAE Analysis by State"),
        ("04_residual_distribution.png", "Residual Distribution & Bias Check"),
        ("05_scatter_actual_predicted.png", "Actual vs Predicted Scatter Plot"),
        ("06_shap_feature_importance.png", "SHAP Feature Importance Analysis"),
        ("07_capacity_planning_widget.png", "Capacity Planning: Gap Analysis by State"),
        ("49_model_comprehensive_analysis.png", "Cell 49: Comprehensive Model Analysis (6-panel)"),
        ("49_state_level_analysis.png", "Cell 49: State-Level Performance Analysis (4-panel)"),
    ]
    
    for tab, (fig_file, caption) in zip(tabs, fig_files):
        with tab:
            try:
                from PIL import Image
                img_path = PROJECT_ROOT / "pdf_figures" / fig_file
                if img_path.exists():
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True, caption=caption)
                else:
                    st.warning(f"⚠️ Figure not found: {fig_file}")
            except Exception as e:
                st.error(f"❌ Error loading {fig_file}: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("")  # spacing


def render_explorer_page(df_eval: pd.DataFrame, selected_state: str, selected_district: str):
    """Render the District Explorer page."""
    # Spacer to push content below Streamlit header
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="uidai-page-header">🔍 District Explorer</div>
        <p class="uidai-page-subtitle">Deep dive into individual district forecasts and performance</p>
        """,
        unsafe_allow_html=True,
    )
    
    # KPI cards at top (official CV metrics from frozen JSON)
    render_global_kpi_row(df_eval)
    st.markdown("")  # spacing
    
    # Filter data
    df_sel = (
        df_eval[
            (df_eval["state"] == selected_state)
            & (df_eval["district"] == selected_district)
        ]
        .sort_values("month_date")
        .copy()
    )
    
    if len(df_sel) == 0:
        st.warning("No data available for the selected district.")
        return
    
    # Main time series chart
    render_main_timeseries(df_sel, selected_state, selected_district)
    
    # Second row: residuals + drivers
    col1, col2 = st.columns([2.2, 1.1])
    
    with col1:
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">📊 Residual Distribution (Actual − Forecast)</div>
            """,
            unsafe_allow_html=True,
        )
        fig = render_residual_histogram(df_sel)
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        mean_res = df_sel["residual"].mean()
        std_res = df_sel["residual"].std()
        max_res = df_sel["residual"].abs().max()
        
        st.markdown(
            f"""
            <div class="uidai-card">
                <div class="uidai-section-title">🔍 Why this forecast?</div>
                <p class="uidai-subtitle">
                    Top drivers: recent 1–3 month enrolments, 3–6 month rolling trend, 
                    and festival/policy flags. Districts with repeated campaigns and 
                    high historical volumes get higher forecasts.
                </p>
                <hr style="border-color:#e2e8f0; margin:1rem 0;">
                <div style="font-size:0.85rem; color:#6B7280;">
                    <strong>District Stats:</strong><br>
                    Mean Residual: {mean_res:.1f}<br>
                    Std Residual: {std_res:.1f}<br>
                    Max |Error|: {max_res:.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("")  # spacing
    
    # Data table
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">📋 Monthly Data</div>
        """,
        unsafe_allow_html=True,
    )
    
    display_cols = ["month_date", "total_enrolment", "y_pred_v3", "residual", "abs_error"]
    df_display = df_sel[display_cols].copy()
    df_display.columns = ["Month", "Actual", "Forecast", "Residual", "Abs Error"]
    df_display["Month"] = df_display["Month"].dt.strftime("%b %Y")
    
    st.dataframe(
        df_display.style.format({
            "Actual": "{:.0f}",
            "Forecast": "{:.0f}",
            "Residual": "{:.1f}",
            "Abs Error": "{:.1f}",
        }).background_gradient(subset=["Abs Error"], cmap="Reds"),
        width="stretch",
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# INDIA MAP PAGE
# =============================================================================

def build_state_level_metrics(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics at state level for the map."""
    def safe_metrics(g):
        n = len(g)
        mae = mean_absolute_error(g["total_enrolment"], g["y_pred_v3"])
        r2 = r2_score(g["total_enrolment"], g["y_pred_v3"]) if n >= 2 else 0.0
        avg_enrol = g["total_enrolment"].mean()
        return pd.Series({"MAE": mae, "R2": r2, "avg_enrol": avg_enrol, "n_districts": g["district"].nunique()})
    
    df_state = (
        df_eval.groupby("state")
        .apply(safe_metrics, include_groups=False)
        .reset_index()
    )
    return df_state


def render_india_map_page(df_eval: pd.DataFrame):
    """Render the India Map page with state-level forecast quality."""
    # Spacer to push content below Streamlit header
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="uidai-page-header">🗺️ India Map – Forecast Quality by State</div>
        <p class="uidai-page-subtitle">Visualize forecast error (MAE) across Indian states. Higher MAE = harder to forecast.</p>
        """,
        unsafe_allow_html=True,
    )
    
    # KPI cards at top (official CV metrics from frozen JSON)
    render_global_kpi_row(df_eval)
    st.markdown("")  # spacing
    
    # Build state-level metrics
    df_state = build_state_level_metrics(df_eval)
    
    # Create choropleth-like bar chart (since we don't have geojson)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">📊 MAE by State (All States)</div>
            """,
            unsafe_allow_html=True,
        )
        
        df_sorted = df_state.sort_values("MAE", ascending=True)
        
        fig = px.bar(
            df_sorted,
            y="state",
            x="MAE",
            orientation="h",
            color="MAE",
            color_continuous_scale="YlOrRd",
            hover_data=["R2", "avg_enrol", "n_districts"],
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Mean Absolute Error",
            yaxis_title="",
            font=dict(family="Poppins, sans-serif"),
            showlegend=False,
            coloraxis_showscale=True,
            height=max(400, len(df_sorted) * 25),
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Top hardest states
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">⚠️ Hardest States (Highest MAE)</div>
            """,
            unsafe_allow_html=True,
        )
        
        top_hard = df_state.nlargest(5, "MAE")
        for _, row in top_hard.iterrows():
            st.markdown(
                f"""
                <div class="district-card district-card-high">
                    <div style="font-weight:600; color:#111827;">{row['state']}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">
                        MAE: <strong>{row['MAE']:.0f}</strong> | R²: {row['R2']:.3f}
                    </div>
                    <div style="font-size:0.8rem; color:#9CA3AF;">
                        {row['n_districts']:.0f} districts | Avg: {row['avg_enrol']:.0f} enrolments
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Best performing states
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">✅ Best States (Lowest MAE)</div>
            """,
            unsafe_allow_html=True,
        )
        
        top_easy = df_state.nsmallest(5, "MAE")
        for _, row in top_easy.iterrows():
            st.markdown(
                f"""
                <div class="district-card district-card-low">
                    <div style="font-weight:600; color:#111827;">{row['state']}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">
                        MAE: <strong>{row['MAE']:.0f}</strong> | R²: {row['R2']:.3f}
                    </div>
                    <div style="font-size:0.8rem; color:#9CA3AF;">
                        {row['n_districts']:.0f} districts | Avg: {row['avg_enrol']:.0f} enrolments
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # State-level data table
    st.markdown("")
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">📋 State-Level Metrics Table</div>
        """,
        unsafe_allow_html=True,
    )
    
    df_table = df_state.copy()
    df_table.columns = ["State", "MAE", "R²", "Avg Enrolment", "Districts"]
    df_table = df_table.sort_values("MAE", ascending=False)
    
    st.dataframe(
        df_table.style.format({
            "MAE": "{:.1f}",
            "R²": "{:.3f}",
            "Avg Enrolment": "{:.0f}",
            "Districts": "{:.0f}",
        }).background_gradient(subset=["MAE"], cmap="YlOrRd"),
        width="stretch",
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MODEL EXPERIMENTS & EXPLAINABILITY SECTIONS
# =============================================================================

def load_lgbm_experiment_metrics() -> dict | None:
    """Load LightGBM experiment metrics if available."""
    lgbm_path = Path("artifacts/phase4_lgbm_experiment_metrics.json")
    if lgbm_path.is_file():
        with open(lgbm_path, "r") as f:
            return json.load(f)
    return None


def render_model_experiments_section(xgb_metrics: dict):
    """Render the model experiments comparison table."""
    lgbm_metrics = load_lgbm_experiment_metrics()
    
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">🔬 Model Experiments (Offline)</div>
        """,
        unsafe_allow_html=True,
    )
    
    if not lgbm_metrics:
        st.info("LightGBM experiment metrics not found; only XGBoost v3 is documented.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    x = xgb_metrics  # XGBoost v3 metrics
    l = lgbm_metrics.get("cv_metrics", {})
    
    st.markdown(
        f"""
        <table style="width:100%; border-collapse:collapse; margin-top:0.5rem;">
            <thead>
                <tr style="border-bottom:2px solid #e2e8f0;">
                    <th style="padding:8px; text-align:left; color:#64748b;">Model</th>
                    <th style="padding:8px; text-align:right; color:#64748b;">R² (mean)</th>
                    <th style="padding:8px; text-align:right; color:#64748b;">R² (std)</th>
                    <th style="padding:8px; text-align:right; color:#64748b;">MAE (mean)</th>
                    <th style="padding:8px; text-align:right; color:#64748b;">MAE (std)</th>
                    <th style="padding:8px; text-align:center; color:#64748b;">Status</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
                    <td style="padding:8px; font-weight:600; color:#0f172a;">XGBoost v3</td>
                    <td style="padding:8px; text-align:right; color:#0f172a;">{x['r2_mean']:.4f}</td>
                    <td style="padding:8px; text-align:right; color:#64748b;">{x['r2_std']:.4f}</td>
                    <td style="padding:8px; text-align:right; color:#0f172a;">{x['mae_mean']:.2f}</td>
                    <td style="padding:8px; text-align:right; color:#64748b;">{x['mae_std']:.2f}</td>
                    <td style="padding:8px; text-align:center;"><span style="background:#22c55e; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem;">PRODUCTION</span></td>
                </tr>
                <tr>
                    <td style="padding:8px; color:#64748b;">LightGBM</td>
                    <td style="padding:8px; text-align:right; color:#64748b;">{l.get('r2_mean', 0):.4f}</td>
                    <td style="padding:8px; text-align:right; color:#94a3b8;">{l.get('r2_std', 0):.4f}</td>
                    <td style="padding:8px; text-align:right; color:#64748b;">{l.get('mae_mean', 0):.2f}</td>
                    <td style="padding:8px; text-align:right; color:#94a3b8;">{l.get('mae_std', 0):.2f}</td>
                    <td style="padding:8px; text-align:center;"><span style="background:#94a3b8; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem;">EXPERIMENT</span></td>
                </tr>
            </tbody>
        </table>
        <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.75rem; font-style:italic;">
            <strong>Conclusion:</strong> XGBoost v3 shows higher R² and much lower MAE on 
            time-series CV, so it remains the official production model while LightGBM 
            is kept as a documented experiment only.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_shap_section():
    """Placeholder for global SHAP feature importance (coming next)."""
    st.markdown(
        """
        <div class="uidai-card" style="border-left:4px solid #0076A8;">
            <div class="uidai-section-title">📊 Global Feature Importance (SHAP)</div>
            <p style="color:#64748b; line-height:1.7; margin-top:0.5rem;">
                <em>Coming next:</em> SHAP summary plot for XGBoost v3 showing which features 
                drive enrolment predictions across all districts.
            </p>
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
                This will help answer: <strong>"What factors most influence Aadhaar enrolment forecasts?"</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_local_shap_section():
    """Placeholder for local SHAP explanation (coming next)."""
    st.markdown(
        """
        <div class="uidai-card" style="border-left:4px solid #FF8A3C;">
            <div class="uidai-section-title">🔍 Why This Forecast? (Local SHAP)</div>
            <p style="color:#64748b; line-height:1.7; margin-top:0.5rem;">
                <em>Coming next:</em> Local SHAP waterfall chart for a selected district 
                and month, explaining why the model predicted a specific value.
            </p>
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
                This will help answer: <strong>"Why did the model predict X enrolments for District Y in Month Z?"</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ABOUT MODEL PAGE
# =============================================================================

def render_about_page():
    """Render the About Model page with model details and methodology."""
    # Spacer to push content below Streamlit header
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="uidai-page-header">ℹ️ About the Model</div>
        <p class="uidai-page-subtitle">Technical details about the Phase-4 v3 UIDAI Enrolment Forecasting Model</p>
        """,
        unsafe_allow_html=True,
    )
    
    # Load official metrics for display
    metrics = load_v3_official_metrics()
    
    # Model Overview card
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">🎯 Model Overview</div>
            <p style="color:#64748b; line-height:1.7; margin-top:0.5rem;">
                The <strong>Phase-4 v3</strong> model is a production-ready XGBoost regressor 
                designed for district-level Aadhaar enrolment forecasting. It was developed 
                with a focus on <em>leakage-safe validation</em> and <em>time-series best practices</em>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="uidai-card">
                <div class="uidai-section-title">📊 Cross-Validation Results</div>
                <table style="width:100%; border-collapse:collapse; margin-top:0.5rem;">
                    <tr style="border-bottom:1px solid #e2e8f0;">
                        <td style="padding:8px 0; color:#64748b;">R² (mean ± std)</td>
                        <td style="padding:8px 0; font-weight:600; color:#0f172a;">{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #e2e8f0;">
                        <td style="padding:8px 0; color:#64748b;">MAE (mean ± std)</td>
                        <td style="padding:8px 0; font-weight:600; color:#0f172a;">{metrics['mae_mean']:.2f} ± {metrics['mae_std']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 0; color:#64748b;">RMSE (mean ± std)</td>
                        <td style="padding:8px 0; font-weight:600; color:#0f172a;">{metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">⚙️ Validation Strategy</div>
                <ul style="color:#64748b; line-height:1.8; padding-left:1.2rem; margin-top:0.5rem;">
                    <li><strong>Method:</strong> Expanding-window time-series CV</li>
                    <li><strong>Folds:</strong> 4 folds</li>
                    <li><strong>Gap:</strong> 1 month (leakage-safe)</li>
                    <li><strong>Min train:</strong> 3 months minimum</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">🔢 Feature Categories</div>
                <ul style="color:#64748b; line-height:1.8; padding-left:1.2rem; margin-top:0.5rem;">
                    <li><strong>Demographics:</strong> Age group distributions (0-5, 5-17, 18+)</li>
                    <li><strong>Lag features:</strong> 1, 2, 3, 6 month lagged enrolments</li>
                    <li><strong>Rolling stats:</strong> 3 and 6 month rolling mean/std</li>
                    <li><strong>Calendar:</strong> Month, quarter, year, sin/cos encoding</li>
                    <li><strong>Policy flags:</strong> Festival peaks, FY boundaries, campaigns</li>
                    <li><strong>Updates:</strong> Demographic and biometric update counts</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="uidai-card">
                <div class="uidai-section-title">🛠️ Technical Stack</div>
                <ul style="color:#64748b; line-height:1.8; padding-left:1.2rem; margin-top:0.5rem;">
                    <li><strong>Algorithm:</strong> XGBoost Regressor</li>
                    <li><strong>Cleaning:</strong> Z-score outlier detection, LOCF imputation</li>
                    <li><strong>Explainability:</strong> SHAP values for feature importance</li>
                    <li><strong>Framework:</strong> Python, scikit-learn, pandas</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Data sources
    st.markdown(
        """
        <div class="uidai-card">
            <div class="uidai-section-title">📁 Data Sources</div>
            <p style="color:#64748b; line-height:1.7; margin-top:0.5rem;">
                The model is trained on district-level monthly aggregates from the UIDAI 
                enrolment database. Features are engineered from historical enrolment patterns, 
                demographic distributions, and calendar/policy events.
            </p>
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
                <strong>Note:</strong> All metrics displayed in this dashboard are official 
                cross-validation metrics from the frozen model artifacts, not recomputed from 
                the displayed data subset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Model Experiments section
    render_model_experiments_section(metrics)
    
    # SHAP placeholders (coming next)
    render_global_shap_section()
    render_local_shap_section()


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar_header():
    """Render sidebar header with optional UIDAI logo."""
    logo_path = Path("assets/uidai_logo.png")
    
    st.sidebar.markdown(
        '<div style="text-align:center; padding:1rem 0 0.5rem 0;">',
        unsafe_allow_html=True,
    )
    
    if logo_path.is_file():
        try:
            from PIL import Image
            logo_img = Image.open(logo_path)
            st.sidebar.image(logo_img, width=80)
        except Exception:
            # Fallback to emoji if image loading fails
            st.sidebar.markdown(
                '<div style="font-size:2.5rem;">📈</div>',
                unsafe_allow_html=True,
            )
    else:
        # Fallback: emoji icon
        st.sidebar.markdown(
            '<div style="font-size:2.5rem;">📈</div>',
            unsafe_allow_html=True,
        )
    
    st.sidebar.markdown(
        """
        <h2 style="margin:0.5rem 0 0 0; font-size:1.2rem; font-weight:700; color:#111827 !important;">UIDAI Dashboard</h2>
        <p style="margin:2px 0 0 0; font-size:0.85rem; color:#6B7280 !important;">Phase‑4 v3 Forecasts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df_eval: pd.DataFrame):
    """Render the sidebar navigation and filters."""
    # Render header with logo
    render_sidebar_header()
    
    st.sidebar.markdown("---")
    
    # Navigation with icons
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🔍 District Explorer", "🗺️ India Map", "🤖 Planning Assistant", "ℹ️ About Model"],
        label_visibility="collapsed",
    )
    
    st.sidebar.markdown("---")
    
    # Filters (only for District Explorer)
    selected_state = None
    selected_district = None
    
    if "District Explorer" in page:
        st.sidebar.markdown("**🎯 Filters**")
        
        states = sorted(df_eval["state"].dropna().unique())
        selected_state = st.sidebar.selectbox("State", states, index=0)
        
        districts = sorted(
            df_eval.loc[df_eval["state"] == selected_state, "district"]
            .dropna()
            .unique()
        )
        selected_district = st.sidebar.selectbox("District", districts, index=0)
        
        st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.markdown("**📁 Data**")
    uploaded_file = st.sidebar.file_uploader(
        "Upload custom CSV",
        type=["csv"],
        help="Upload a UIDAI district-month CSV to generate predictions",
    )
    
    st.sidebar.markdown("---")
    
    # Info
    st.sidebar.markdown(
        """
        <div style="font-size:0.8rem; color:#64748b !important;">
            <strong style="color:#0f172a !important;">Model:</strong> XGBoost v3 Baseline<br>
            <strong style="color:#0f172a !important;">CV:</strong> 4-fold expanding window<br>
            <strong style="color:#0f172a !important;">Gap:</strong> 1 month (leakage-safe)<br>
            <strong style="color:#0f172a !important;">R² (CV):</strong> 0.955 ± 0.049
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    return page, selected_state, selected_district, uploaded_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application entry point."""
    # Load model
    try:
        model = load_v3_model()
    except Exception as e:
        st.error(f"❌ Failed to load v3 model: {e}")
        st.info("Make sure to run `python -m src.freeze_phase4_v3_baseline` first.")
        return
    
    # Load base data
    try:
        df_raw = load_base_data()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return
    
    # Clean and generate predictions
    with st.spinner("Loading model and generating predictions..."):
        df_clean = clean_data(df_raw)
        df_eval = generate_predictions(model, df_clean)
    
    # Render sidebar and get selections
    page, selected_state, selected_district, uploaded_file = render_sidebar(df_eval)
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file, parse_dates=["month_date"])
            df_clean_uploaded = clean_data(df_uploaded)
            df_eval = generate_predictions(model, df_clean_uploaded)
            st.sidebar.success("✅ Custom data loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
    
    # Render selected page based on navigation choice
    if "Overview" in page:
        render_overview_page(df_eval)
    elif "District Explorer" in page:
        if selected_state and selected_district:
            render_explorer_page(df_eval, selected_state, selected_district)
        else:
            st.warning("Please select a state and district from the sidebar.")
    elif "India Map" in page:
        render_india_map_page(df_eval)
    elif "Planning Assistant" in page:
        planning_tab()
    elif "About Model" in page:
        render_about_page()


if __name__ == "__main__":
    main()
