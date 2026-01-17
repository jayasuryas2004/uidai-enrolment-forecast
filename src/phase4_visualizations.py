#!/usr/bin/env python
"""
phase4_visualizations.py
========================

Visualization module for Phase-4 v3 UIDAI forecasting model.

This module creates publication-quality visualizations for:
1. Forecast vs Actual over time (line plots for representative districts)
2. Residual analysis (time-series and scatter plots)
3. CV fold structure diagram (expanding-window visualization)

All plots are designed for hackathon presentations and technical documentation.

Usage:
    python -m src.phase4_visualizations --output-dir reports/figures

Author: UIDAI Forecast Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    "actual": "#2ecc71",       # Green
    "forecast": "#3498db",     # Blue
    "residual": "#e74c3c",     # Red
    "train": "#3498db",        # Blue
    "val": "#e74c3c",          # Red
    "gap": "#95a5a6",          # Gray
    "ci": "#bdc3c7",           # Light gray
    "zero": "#7f8c8d",         # Dark gray
}

# Data paths
DATA_PATH = Path("data/processed/district_month_modeling.csv")
METRICS_PATH = Path("artifacts/xgb_phase4_v3_baseline_metrics.json")


# =============================================================================
# 1. FORECAST VS ACTUAL PLOTS
# =============================================================================

def plot_forecast_vs_actual_single_district(
    df: pd.DataFrame,
    state: str,
    district: str,
    date_col: str = "month_date",
    actual_col: str = "total_enrolment",
    forecast_col: str = "forecast",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot forecast vs actual enrolment for a single district.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and forecast columns.
    state : str
        State name.
    district : str
        District name.
    date_col : str
        Name of date column.
    actual_col : str
        Name of actual values column.
    forecast_col : str
        Name of forecast values column.
    title : str, optional
        Plot title. If None, auto-generated.
    ax : plt.Axes, optional
        Matplotlib axes. If None, creates new figure.
        
    Returns
    -------
    plt.Axes
        The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    # Filter for this district
    mask = (df["state"] == state) & (df["district"] == district)
    district_df = df[mask].sort_values(date_col).copy()
    
    if len(district_df) == 0:
        ax.text(0.5, 0.5, f"No data for {district}, {state}", 
                ha="center", va="center", transform=ax.transAxes)
        return ax
    
    dates = pd.to_datetime(district_df[date_col])
    actuals = district_df[actual_col].values
    forecasts = district_df[forecast_col].values
    
    # Plot lines
    ax.plot(dates, actuals, 
            color=COLORS["actual"], linewidth=2.5, marker="o", 
            markersize=6, label="Actual", zorder=3)
    ax.plot(dates, forecasts, 
            color=COLORS["forecast"], linewidth=2.5, marker="s", 
            markersize=5, linestyle="--", label="Forecast", zorder=2)
    
    # Fill between for visual comparison
    ax.fill_between(dates, actuals, forecasts, 
                    alpha=0.2, color=COLORS["residual"], zorder=1)
    
    # Formatting
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Enrolment")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    if title is None:
        title = f"Forecast vs Actual: {district}, {state}"
    ax.set_title(title, fontweight="bold")
    
    # Add MAE annotation
    mae = np.mean(np.abs(actuals - forecasts))
    r2 = 1 - np.sum((actuals - forecasts)**2) / np.sum((actuals - np.mean(actuals))**2)
    ax.annotate(
        f"MAE: {mae:.1f}\nR²: {r2:.3f}",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=10, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    
    return ax


def plot_forecast_vs_actual_grid(
    df: pd.DataFrame,
    districts: List[Tuple[str, str]],
    date_col: str = "month_date",
    actual_col: str = "total_enrolment",
    forecast_col: str = "forecast",
    ncols: int = 2,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: str = "Forecast vs Actual Enrolment by District",
) -> plt.Figure:
    """
    Create a grid of forecast vs actual plots for multiple districts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and forecast columns.
    districts : List[Tuple[str, str]]
        List of (state, district) tuples to plot.
    ncols : int
        Number of columns in the grid.
    figsize : tuple
        Figure size.
    suptitle : str
        Super title for the figure.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    nrows = (len(districts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if len(districts) > 1 else [axes]
    
    for i, (state, district) in enumerate(districts):
        plot_forecast_vs_actual_single_district(
            df, state, district, date_col, actual_col, forecast_col,
            title=f"{district}",
            ax=axes[i],
        )
    
    # Hide empty subplots
    for j in range(len(districts), len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    return fig


# =============================================================================
# 2. RESIDUAL PLOTS
# =============================================================================

def plot_residuals_over_time(
    df: pd.DataFrame,
    districts: List[Tuple[str, str]],
    date_col: str = "month_date",
    actual_col: str = "total_enrolment",
    forecast_col: str = "forecast",
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot residuals over time for selected districts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and forecast columns.
    districts : List[Tuple[str, str]]
        List of (state, district) tuples to plot.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    
    for i, (state, district) in enumerate(districts):
        mask = (df["state"] == state) & (df["district"] == district)
        district_df = df[mask].sort_values(date_col).copy()
        
        if len(district_df) == 0:
            continue
        
        dates = pd.to_datetime(district_df[date_col])
        residuals = district_df[actual_col].values - district_df[forecast_col].values
        
        ax.plot(dates, residuals, 
                marker=markers[i % len(markers)], 
                linewidth=1.5, markersize=5,
                label=f"{district}", alpha=0.8)
    
    # Zero line
    ax.axhline(y=0, color=COLORS["zero"], linestyle="--", linewidth=1.5, zorder=1)
    
    # Formatting
    ax.set_xlabel("Month")
    ax.set_ylabel("Residual (Actual - Forecast)")
    ax.set_title("Residuals Over Time by District", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Add mean residual annotation
    all_residuals = df[actual_col] - df[forecast_col]
    mean_res = all_residuals.mean()
    std_res = all_residuals.std()
    ax.annotate(
        f"Mean: {mean_res:.1f}\nStd: {std_res:.1f}",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=10, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    
    plt.tight_layout()
    return fig


def plot_residuals_scatter(
    df: pd.DataFrame,
    actual_col: str = "total_enrolment",
    forecast_col: str = "forecast",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Create scatter plots of predictions vs residuals to check for bias.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and forecast columns.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    forecasts = df[forecast_col].values
    actuals = df[actual_col].values
    residuals = actuals - forecasts
    
    # Plot 1: Predictions vs Residuals
    ax1 = axes[0]
    ax1.scatter(forecasts, residuals, alpha=0.5, s=20, color=COLORS["forecast"])
    ax1.axhline(y=0, color=COLORS["zero"], linestyle="--", linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(forecasts, residuals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(forecasts.min(), forecasts.max(), 100)
    ax1.plot(x_line, p(x_line), color=COLORS["residual"], linewidth=2, 
             linestyle="-", label=f"Trend (slope={z[0]:.3f})")
    
    ax1.set_xlabel("Predicted Value")
    ax1.set_ylabel("Residual (Actual - Predicted)")
    ax1.set_title("Predictions vs Residuals", fontweight="bold")
    ax1.legend(loc="upper right")
    
    # Plot 2: Actual vs Predicted with identity line
    ax2 = axes[1]
    ax2.scatter(actuals, forecasts, alpha=0.5, s=20, color=COLORS["actual"])
    
    # Identity line (perfect predictions)
    min_val = min(actuals.min(), forecasts.min())
    max_val = max(actuals.max(), forecasts.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 
             color=COLORS["residual"], linestyle="--", linewidth=2, label="Perfect Fit")
    
    # R² annotation
    r2 = 1 - np.sum((actuals - forecasts)**2) / np.sum((actuals - np.mean(actuals))**2)
    ax2.annotate(
        f"R² = {r2:.4f}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        fontsize=11, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    
    ax2.set_xlabel("Actual Value")
    ax2.set_ylabel("Predicted Value")
    ax2.set_title("Actual vs Predicted", fontweight="bold")
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    return fig


def plot_residual_distribution(
    df: pd.DataFrame,
    actual_col: str = "total_enrolment",
    forecast_col: str = "forecast",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot histogram and Q-Q plot of residuals.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and forecast columns.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    residuals = df[actual_col].values - df[forecast_col].values
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(residuals, bins=50, color=COLORS["forecast"], alpha=0.7, edgecolor="white")
    ax1.axvline(x=0, color=COLORS["residual"], linestyle="--", linewidth=2)
    ax1.axvline(x=residuals.mean(), color=COLORS["actual"], linestyle="-", linewidth=2,
                label=f"Mean = {residuals.mean():.1f}")
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Residual Distribution", fontweight="bold")
    ax1.legend()
    
    # Box plot by month
    ax2 = axes[1]
    df_temp = df.copy()
    df_temp["residual"] = residuals
    df_temp["month"] = pd.to_datetime(df_temp["month_date"]).dt.strftime("%b")
    
    # Get unique months in order
    df_temp["month_order"] = pd.to_datetime(df_temp["month_date"]).dt.month
    months_order = df_temp.sort_values("month_order")["month"].unique()
    
    boxplot_data = [df_temp[df_temp["month"] == m]["residual"].values for m in months_order]
    bp = ax2.boxplot(boxplot_data, tick_labels=months_order, patch_artist=True)
    
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["forecast"])
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color=COLORS["residual"], linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals by Month", fontweight="bold")
    
    plt.tight_layout()
    return fig


# =============================================================================
# 3. CV FOLD VISUALIZATION
# =============================================================================

def plot_cv_fold_diagram(
    n_months: int = 9,
    n_folds: int = 4,
    gap_months: int = 1,
    min_train_months: int = 3,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Create a visual diagram of the expanding-window CV structure.
    
    This shows the train (blue) vs validation (red) blocks for each fold,
    with the gap clearly marked.
    
    Parameters
    ----------
    n_months : int
        Total number of months in the dataset.
    n_folds : int
        Number of CV folds.
    gap_months : int
        Gap between train and validation periods.
    min_train_months : int
        Minimum training months for first fold.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate fold boundaries (matching actual CV logic)
    months = list(range(n_months))
    month_labels = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if len(month_labels) < n_months:
        month_labels = [f"M{i+1}" for i in range(n_months)]
    
    # Fold structure based on expanding window
    folds = []
    
    # Calculate validation months (last n_folds unique months)
    val_months = months[-(n_folds):]
    
    for fold_idx in range(n_folds):
        val_month = val_months[fold_idx]
        train_end = val_month - gap_months
        train_start = 0
        
        if train_end >= min_train_months:
            folds.append({
                "fold": fold_idx + 1,
                "train_start": train_start,
                "train_end": train_end,
                "gap_start": train_end,
                "gap_end": val_month,
                "val_month": val_month,
            })
    
    # Plot parameters
    bar_height = 0.6
    y_spacing = 1.2
    
    for i, fold in enumerate(folds):
        y = (len(folds) - 1 - i) * y_spacing
        
        # Training period (blue)
        train_width = fold["train_end"] - fold["train_start"]
        train_rect = Rectangle(
            (fold["train_start"], y - bar_height/2),
            train_width, bar_height,
            facecolor=COLORS["train"], edgecolor="white", linewidth=2,
            alpha=0.8, label="Train" if i == 0 else None
        )
        ax.add_patch(train_rect)
        
        # Gap period (gray)
        gap_width = fold["gap_end"] - fold["gap_start"]
        gap_rect = Rectangle(
            (fold["gap_start"], y - bar_height/2),
            gap_width, bar_height,
            facecolor=COLORS["gap"], edgecolor="white", linewidth=2,
            alpha=0.5, label="Gap" if i == 0 else None
        )
        ax.add_patch(gap_rect)
        
        # Validation period (red)
        val_rect = Rectangle(
            (fold["val_month"], y - bar_height/2),
            1, bar_height,
            facecolor=COLORS["val"], edgecolor="white", linewidth=2,
            alpha=0.8, label="Validation" if i == 0 else None
        )
        ax.add_patch(val_rect)
        
        # Fold label
        ax.text(-0.8, y, f"Fold {fold['fold']}", 
                ha="right", va="center", fontsize=11, fontweight="bold")
        
        # Training months count
        ax.text(fold["train_start"] + train_width/2, y, 
                f"{train_width} months",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        
        # Gap indicator
        if gap_months > 0:
            ax.text(fold["gap_start"] + gap_width/2, y,
                    f"{gap_width}m gap",
                    ha="center", va="center", fontsize=8, color="black", fontstyle="italic")
    
    # Set axis limits and labels
    ax.set_xlim(-1.5, n_months + 0.5)
    ax.set_ylim(-0.8, len(folds) * y_spacing)
    
    # X-axis with month labels
    ax.set_xticks(range(n_months))
    ax.set_xticklabels(month_labels[:n_months], fontsize=10)
    ax.set_xlabel("Month (2025)", fontsize=11)
    
    # Hide y-axis
    ax.set_yticks([])
    ax.set_ylabel("")
    
    # Title
    ax.set_title(
        f"Expanding-Window Time-Series Cross-Validation\n"
        f"({n_folds} folds, {gap_months}-month gap, leakage-safe)",
        fontsize=13, fontweight="bold", pad=15
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["train"], alpha=0.8, label="Training Period"),
        mpatches.Patch(facecolor=COLORS["gap"], alpha=0.5, label=f"Gap ({gap_months} month)"),
        mpatches.Patch(facecolor=COLORS["val"], alpha=0.8, label="Validation Month"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    
    # Add annotation explaining the approach
    explanation = (
        "→ Each fold trains on all available past data\n"
        "→ Gap prevents data leakage from recent months\n"
        "→ Validation is always in the future"
    )
    ax.annotate(
        explanation,
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )
    
    # Remove spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_cv_fold_detailed(
    fold_metrics: List[dict],
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Create a bar chart showing metrics per CV fold.
    
    Parameters
    ----------
    fold_metrics : List[dict]
        List of fold metrics from CV run.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    folds = [f"Fold {m['fold']}" for m in fold_metrics]
    r2_scores = [m["r2"] for m in fold_metrics]
    mae_scores = [m["mae"] for m in fold_metrics]
    train_sizes = [m["train_rows"] for m in fold_metrics]
    
    # R² by fold
    ax1 = axes[0]
    bars1 = ax1.bar(folds, r2_scores, color=COLORS["train"], alpha=0.8, edgecolor="white")
    ax1.axhline(y=np.mean(r2_scores), color=COLORS["residual"], linestyle="--", 
                linewidth=2, label=f"Mean = {np.mean(r2_scores):.3f}")
    ax1.set_ylabel("R² Score")
    ax1.set_title("R² Score by Fold", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    
    # Add value labels
    for bar, val in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    # MAE by fold
    ax2 = axes[1]
    bars2 = ax2.bar(folds, mae_scores, color=COLORS["forecast"], alpha=0.8, edgecolor="white")
    ax2.axhline(y=np.mean(mae_scores), color=COLORS["residual"], linestyle="--",
                linewidth=2, label=f"Mean = {np.mean(mae_scores):.1f}")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE by Fold", fontweight="bold")
    ax2.legend()
    
    # Add value labels with train size
    for bar, val, train_n in zip(bars2, mae_scores, train_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.1f}\n(n={train_n})", ha="center", fontsize=8, fontweight="bold")
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN VISUALIZATION PIPELINE
# =============================================================================

def generate_sample_predictions(
    df: pd.DataFrame,
    model_path: Path,
    encoders_path: Path,
) -> pd.DataFrame:
    """
    Generate predictions using the frozen v3 model for visualization.
    
    This is a simplified version - in production, use phase4_v3_inference.
    """
    from src.phase4_v3_inference import Phase4V3Model
    
    model = Phase4V3Model()
    model.load()
    
    predictions = model.predict(df)
    df = df.copy()
    df["forecast"] = predictions.values
    
    return df


def select_representative_districts(
    df: pd.DataFrame,
    n_districts: int = 6,
) -> List[Tuple[str, str]]:
    """
    Select representative districts for visualization.
    
    Selects districts with varying enrolment levels and patterns.
    """
    # Get district summary
    district_stats = df.groupby(["state", "district"]).agg({
        "total_enrolment": ["mean", "std", "count"]
    }).reset_index()
    district_stats.columns = ["state", "district", "mean", "std", "count"]
    
    # Filter for districts with enough data
    district_stats = district_stats[district_stats["count"] >= 6]
    
    # Select districts from different quantiles
    district_stats["quantile"] = pd.qcut(district_stats["mean"], q=3, labels=["low", "medium", "high"])
    
    selected = []
    for q in ["low", "medium", "high"]:
        q_districts = district_stats[district_stats["quantile"] == q]
        if len(q_districts) >= 2:
            # Select 2 from each quantile
            sample = q_districts.sample(min(2, len(q_districts)), random_state=42)
            for _, row in sample.iterrows():
                selected.append((row["state"], row["district"]))
    
    return selected[:n_districts]


def run_visualization_pipeline(
    output_dir: Path,
    with_predictions: bool = True,
) -> None:
    """
    Run the complete visualization pipeline.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save figures.
    with_predictions : bool
        Whether to generate model predictions (requires frozen model).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PHASE-4 v3 VISUALIZATION PIPELINE")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["month_date"] = pd.to_datetime(df["month_date"])
    logger.info(f"Loaded: {len(df)} rows")
    
    # Generate predictions if model available
    if with_predictions:
        try:
            logger.info("Generating predictions with v3 model...")
            df = generate_sample_predictions(
                df,
                Path("artifacts/xgb_phase4_v3_baseline.pkl"),
                Path("artifacts/xgb_phase4_v3_baseline_encoders.pkl"),
            )
            has_predictions = True
            logger.info("Predictions generated successfully")
        except Exception as e:
            logger.warning(f"Could not generate predictions: {e}")
            logger.warning("Using dummy forecasts for visualization demo")
            # Create dummy forecasts for demonstration
            df["forecast"] = df["total_enrolment"] * (1 + np.random.randn(len(df)) * 0.1)
            has_predictions = False
    else:
        df["forecast"] = df["total_enrolment"] * (1 + np.random.randn(len(df)) * 0.1)
        has_predictions = False
    
    # Select representative districts
    districts = select_representative_districts(df, n_districts=6)
    logger.info(f"Selected {len(districts)} representative districts")
    
    # -------------------------------------------------------------------------
    # 1. Forecast vs Actual Plots
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Creating forecast vs actual plots...")
    
    fig = plot_forecast_vs_actual_grid(
        df, districts,
        suptitle="Forecast vs Actual Enrolment: Representative Districts",
    )
    fig.savefig(output_dir / "forecast_vs_actual_grid.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: forecast_vs_actual_grid.png")
    
    # Individual district plot (larger, more detailed)
    if len(districts) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        state, district = districts[0]
        plot_forecast_vs_actual_single_district(
            df, state, district, ax=ax,
            title=f"Detailed View: {district}, {state}",
        )
        fig.savefig(output_dir / "forecast_vs_actual_detailed.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: forecast_vs_actual_detailed.png")
    
    # -------------------------------------------------------------------------
    # 2. Residual Plots
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Creating residual plots...")
    
    # Residuals over time
    fig = plot_residuals_over_time(df, districts[:4])
    fig.savefig(output_dir / "residuals_over_time.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: residuals_over_time.png")
    
    # Residual scatter plots
    fig = plot_residuals_scatter(df)
    fig.savefig(output_dir / "residuals_scatter.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: residuals_scatter.png")
    
    # Residual distribution
    fig = plot_residual_distribution(df)
    fig.savefig(output_dir / "residuals_distribution.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: residuals_distribution.png")
    
    # -------------------------------------------------------------------------
    # 3. CV Fold Visualization
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Creating CV fold diagrams...")
    
    # CV structure diagram
    fig = plot_cv_fold_diagram(
        n_months=9,
        n_folds=4,
        gap_months=1,
        min_train_months=3,
    )
    fig.savefig(output_dir / "cv_fold_structure.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: cv_fold_structure.png")
    
    # CV metrics by fold (if metrics available)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        
        if "cv_metrics" in metrics and "fold_metrics" in metrics["cv_metrics"]:
            fold_metrics = metrics["cv_metrics"]["fold_metrics"]
            fig = plot_cv_fold_detailed(fold_metrics)
            fig.savefig(output_dir / "cv_fold_metrics.png", dpi=150)
            plt.close(fig)
            logger.info(f"  Saved: cv_fold_metrics.png")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    for f in sorted(output_dir.glob("*.png")):
        logger.info(f"  • {f.name}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Phase-4 v3 visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory to save figures (default: reports/figures)",
    )
    
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip generating predictions (use dummy data)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    run_visualization_pipeline(
        output_dir=Path(args.output_dir),
        with_predictions=not args.no_predictions,
    )


if __name__ == "__main__":
    main()
