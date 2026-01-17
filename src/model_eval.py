"""
UIDAI ASRIS - Model Evaluation Module

This module provides production-ready functions for:
- Evaluating model performance (MAE, RMSE, R²)
- Running time-series cross-validation
- Detecting data drift and distribution shifts
- Generating operational alerts and risk signals
- Extracting feature importance for explainability

All functions include input validation, type hints, and defensive checks
to ensure robustness in production environments.

Author: UIDAI ASRIS Team
Version: 1.1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
import logging

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Local imports
from .config import (
    PipelineConfig, DriftConfig, AlertConfig, 
    DEFAULT_CONFIG, PROJECT_ROOT
)


# =============================================================================
# Module Logger
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Model Evaluation
# =============================================================================
def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "evaluation",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and return metrics.
    
    Parameters
    ----------
    model : Pipeline
        Trained sklearn Pipeline.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True target values.
    dataset_name : str
        Name for logging.
    verbose : bool
        Print metrics.
        
    Returns
    -------
    dict
        Dictionary with mae, rmse, r2, n_samples metrics.
        
    Raises
    ------
    ValueError
        If X and y have mismatched lengths.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got X={len(X)}, y={len(y)}")
    
    if len(X) == 0:
        raise ValueError("Cannot evaluate on empty dataset")
    
    y_pred = model.predict(X)
    
    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
        "n_samples": len(y),
    }
    
    logger.info(f"{dataset_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    
    if verbose:
        print(f"📊 {dataset_name} Metrics:")
        print(f"   MAE:  {metrics['mae']:.2f}")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   R²:   {metrics['r2']:.4f}")
    
    return metrics


def run_cv_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    date_series: pd.Series,
    build_pipeline_fn: Callable[[], Pipeline],
    n_splits: int = None,
    verbose: bool = True,
    config: Union[PipelineConfig, None] = None,
) -> pd.DataFrame:
    """
    Run time-series cross-validation and return per-fold metrics.
    
    This function uses sklearn's TimeSeriesSplit to ensure proper
    temporal ordering (no data leakage from future to past).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (must be sorted by time).
    y : pd.Series
        Target variable.
    date_series : pd.Series
        Date column for reporting.
    build_pipeline_fn : callable
        Function that returns a fresh (unfitted) Pipeline.
    n_splits : int, optional
        Number of CV folds. Uses config default if not provided.
    verbose : bool
        Print progress.
    config : PipelineConfig, optional
        Configuration object.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fold, train_start, train_end, 
        val_start, val_end, n_train, n_val, mae, rmse, r2.
    """
    # Resolve configuration
    if config is None:
        config = DEFAULT_CONFIG
    
    if n_splits is None:
        n_splits = config.cv.n_splits
    
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    logger.info(f"Starting {n_splits}-fold time-series CV")
    
    if verbose:
        print("=" * 60)
        print(f"TIME-SERIES CROSS-VALIDATION ({n_splits} folds)")
        print("=" * 60)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        train_dates = date_series.iloc[train_idx]
        val_dates = date_series.iloc[val_idx]
        
        if verbose:
            print(f"\nFold {fold_idx}:")
            print(f"  Train: {train_dates.min()} → {train_dates.max()} ({len(train_idx):,} rows)")
            print(f"  Val:   {val_dates.min()} → {val_dates.max()} ({len(val_idx):,} rows)")
        
        # Build and fit fresh pipeline
        pipe = build_pipeline_fn()
        pipe.fit(X_train_fold, y_train_fold)
        
        # Predict and evaluate
        y_pred = pipe.predict(X_val_fold)
        mae = float(mean_absolute_error(y_val_fold, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
        r2 = float(r2_score(y_val_fold, y_pred))
        
        if verbose:
            print(f"  MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        
        logger.info(f"Fold {fold_idx}: MAE={mae:.2f}, R²={r2:.4f}")
        
        results.append({
            "fold": fold_idx,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "val_start": val_dates.min(),
            "val_end": val_dates.max(),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })
    
    if verbose:
        print("=" * 60)
    
    return pd.DataFrame(results)


# =============================================================================
# Drift Detection
# =============================================================================
def compute_feature_drift_over_time(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: List[str],
    freq: str = "M",
) -> pd.DataFrame:
    """
    Compute feature statistics (mean, std, count) over time periods.
    
    This is used to detect distribution shifts that may require retraining.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with date and feature columns.
    date_col : str
        Name of the date/timestamp column.
    feature_cols : list of str
        List of numeric feature columns to monitor.
    freq : str
        Frequency for grouping ('M' for month, 'W' for week).
        
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: period, feature, mean, std, count.
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_period"] = df[date_col].dt.to_period(freq)
    
    results = []
    for feature in feature_cols:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in DataFrame, skipping")
            continue
            
        stats = df.groupby("_period")[feature].agg(["mean", "std", "count"])
        for period, row in stats.iterrows():
            results.append({
                "period": str(period),
                "feature": feature,
                "mean": float(row["mean"]) if pd.notna(row["mean"]) else 0.0,
                "std": float(row["std"]) if pd.notna(row["std"]) else 0.0,
                "count": int(row["count"]),
            })
    
    return pd.DataFrame(results)


def detect_drift(
    drift_stats: pd.DataFrame,
    reference_periods: int = None,
    threshold_pct: float = None,
    config: Union[PipelineConfig, DriftConfig, None] = None,
) -> pd.DataFrame:
    """
    Detect drift by comparing each period against a reference baseline.
    
    A feature is flagged for drift if its mean value changes by more than
    threshold_pct compared to the reference period mean.
    
    Parameters
    ----------
    drift_stats : pd.DataFrame
        Output from compute_feature_drift_over_time().
    reference_periods : int, optional
        Number of initial periods to use as reference baseline.
    threshold_pct : float, optional
        Threshold for relative change (%) to flag as drift.
    config : PipelineConfig or DriftConfig, optional
        Configuration object for defaults.
        
    Returns
    -------
    pd.DataFrame
        Drift report with columns: period, feature, mean, ref_mean, 
        rel_change_pct, drift_flag.
    """
    # Resolve configuration
    if config is None:
        drift_config = DEFAULT_CONFIG.drift
    elif isinstance(config, PipelineConfig):
        drift_config = config.drift
    elif isinstance(config, DriftConfig):
        drift_config = config
    else:
        drift_config = DEFAULT_CONFIG.drift
    
    if reference_periods is None:
        reference_periods = drift_config.reference_periods
    if threshold_pct is None:
        threshold_pct = drift_config.threshold_pct
    
    results = []
    
    for feature in drift_stats["feature"].unique():
        feature_data = drift_stats[drift_stats["feature"] == feature].copy()
        feature_data = feature_data.sort_values("period")
        
        # Compute reference baseline from first N periods
        ref_periods = feature_data["period"].unique()[:reference_periods]
        ref_data = feature_data[feature_data["period"].isin(ref_periods)]
        ref_mean = ref_data["mean"].mean()
        
        for _, row in feature_data.iterrows():
            if ref_mean != 0:
                rel_change_pct = ((row["mean"] - ref_mean) / abs(ref_mean)) * 100
            else:
                rel_change_pct = 0.0 if row["mean"] == 0 else float('inf')
            
            drift_flag = abs(rel_change_pct) > threshold_pct
            
            results.append({
                "period": row["period"],
                "feature": feature,
                "mean": row["mean"],
                "ref_mean": ref_mean,
                "rel_change_pct": rel_change_pct,
                "drift_flag": drift_flag,
            })
    
    return pd.DataFrame(results)


def run_drift_checks(
    df: pd.DataFrame,
    date_col: str = None,
    feature_cols: List[str] = None,
    reference_periods: int = None,
    threshold_pct: float = None,
    verbose: bool = True,
    config: Union[PipelineConfig, DriftConfig, None] = None,
) -> Dict[str, Any]:
    """
    Run complete drift detection pipeline.
    
    This is the main entry point for drift monitoring. It:
    1. Computes feature statistics over time
    2. Compares to reference baseline
    3. Flags significant drift
    4. Generates summary report
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str, optional
        Name of the date column.
    feature_cols : list of str, optional
        Features to monitor. Uses config default if not provided.
    reference_periods : int, optional
        Number of reference periods for baseline.
    threshold_pct : float, optional
        Threshold for drift flagging.
    verbose : bool
        Print summary.
    config : PipelineConfig or DriftConfig, optional
        Configuration object.
        
    Returns
    -------
    dict
        Contains: drift_stats, drift_report, summary.
    """
    # Resolve configuration
    if config is None:
        pipe_config = DEFAULT_CONFIG
        drift_config = DEFAULT_CONFIG.drift
    elif isinstance(config, PipelineConfig):
        pipe_config = config
        drift_config = config.drift
    elif isinstance(config, DriftConfig):
        pipe_config = DEFAULT_CONFIG
        drift_config = config
    else:
        pipe_config = DEFAULT_CONFIG
        drift_config = DEFAULT_CONFIG.drift
    
    if date_col is None:
        date_col = pipe_config.data.date_col
    if feature_cols is None:
        feature_cols = drift_config.drift_features.copy()
    if reference_periods is None:
        reference_periods = drift_config.reference_periods
    if threshold_pct is None:
        threshold_pct = drift_config.threshold_pct
    
    # Filter to columns that exist
    available_features = [c for c in feature_cols if c in df.columns]
    
    if not available_features:
        logger.warning("No drift features found in DataFrame")
        return {
            "drift_stats": pd.DataFrame(),
            "drift_report": pd.DataFrame(),
            "summary": {"n_features_monitored": 0, "total_drift_flags": 0},
        }
    
    logger.info(f"Running drift checks on {len(available_features)} features")
    
    if verbose:
        print("=" * 60)
        print("DRIFT DETECTION")
        print("=" * 60)
        print(f"Monitoring {len(available_features)} features")
        print(f"Threshold: ±{threshold_pct}% change from baseline")
    
    # Compute drift stats
    drift_stats = compute_feature_drift_over_time(df, date_col, available_features)
    
    # Detect drift
    drift_report = detect_drift(
        drift_stats, 
        reference_periods=reference_periods,
        threshold_pct=threshold_pct,
        config=drift_config,
    )
    
    # Summary
    total_flags = int(drift_report["drift_flag"].sum())
    features_with_drift = int(drift_report[drift_report["drift_flag"]]["feature"].nunique())
    
    summary = {
        "n_features_monitored": len(available_features),
        "n_periods": drift_report["period"].nunique(),
        "total_drift_flags": total_flags,
        "features_with_drift": features_with_drift,
        "threshold_pct": threshold_pct,
    }
    
    logger.info(f"Drift check complete: {total_flags} flags across {features_with_drift} features")
    
    if verbose:
        print(f"  Total drift flags: {total_flags}")
        print(f"  Features with drift: {features_with_drift}")
        
        # Show recent drift
        if not drift_report.empty:
            latest_period = drift_report["period"].max()
            latest_drift = drift_report[
                (drift_report["period"] == latest_period) & 
                (drift_report["drift_flag"])
            ]
            if not latest_drift.empty:
                print(f"\n⚠️  Drift detected in latest period ({latest_period}):")
                for _, row in latest_drift.iterrows():
                    print(f"    {row['feature']}: {row['rel_change_pct']:+.1f}%")
            else:
                print(f"\n✓ No drift in latest period ({latest_period})")
        print("=" * 60)
    
    return {
        "drift_stats": drift_stats,
        "drift_report": drift_report,
        "summary": summary,
    }


# =============================================================================
# Alert Generation
# =============================================================================
def generate_alerts(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray = None,
    high_load_percentile: int = None,
    high_load_growth_threshold: float = None,
    underperform_percentile: int = None,
    underperform_error_threshold: float = None,
    lag_col: str = None,
    config: Union[PipelineConfig, AlertConfig, None] = None,
) -> pd.DataFrame:
    """
    Generate alert signals based on model predictions and errors.
    
    Two types of alerts are generated:
    1. High Load Alert: High predicted volume + high growth rate
    2. Underperforming Alert: High predicted volume + actual < predicted + high error
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with district/month info.
    y_pred : np.ndarray
        Model predictions.
    y_true : np.ndarray, optional
        Actual values (required for underperforming alerts).
    high_load_percentile : int, optional
        Percentile threshold for high load volume.
    high_load_growth_threshold : float, optional
        Growth rate threshold for high load.
    underperform_percentile : int, optional
        Percentile threshold for underperforming.
    underperform_error_threshold : float, optional
        Relative error threshold for underperforming.
    lag_col : str, optional
        Name of lag column for growth calculation.
    config : PipelineConfig or AlertConfig, optional
        Configuration object.
        
    Returns
    -------
    pd.DataFrame
        Alerts table with alert flags and types.
    """
    # Resolve configuration
    if config is None:
        alert_config = DEFAULT_CONFIG.alert
    elif isinstance(config, PipelineConfig):
        alert_config = config.alert
    elif isinstance(config, AlertConfig):
        alert_config = config
    else:
        alert_config = DEFAULT_CONFIG.alert
    
    if high_load_percentile is None:
        high_load_percentile = alert_config.high_load_volume_percentile
    if high_load_growth_threshold is None:
        high_load_growth_threshold = alert_config.high_load_growth_threshold
    if underperform_percentile is None:
        underperform_percentile = alert_config.underperform_volume_percentile
    if underperform_error_threshold is None:
        underperform_error_threshold = alert_config.underperform_rel_error_threshold
    if lag_col is None:
        lag_col = alert_config.lag_col
    
    epsilon = alert_config.epsilon
    
    # Validate inputs
    if len(df) != len(y_pred):
        raise ValueError(f"df and y_pred must have same length. Got df={len(df)}, y_pred={len(y_pred)}")
    
    logger.info(f"Generating alerts for {len(df)} records")
    
    # Build base alerts DataFrame
    id_cols = ["state", "district", "month_date"]
    available_cols = [c for c in id_cols if c in df.columns]
    
    if "year_month" in df.columns:
        available_cols.append("year_month")
    
    alerts = df[available_cols].copy()
    alerts["y_pred"] = y_pred
    
    if y_true is not None:
        alerts["y_true"] = y_true
        alerts["abs_error"] = np.abs(y_true - y_pred)
        alerts["rel_error"] = alerts["abs_error"] / (y_true + epsilon)
    
    # Growth rate
    if lag_col in df.columns:
        lag_values = df[lag_col].values
        alerts["growth_rate"] = (y_pred - lag_values) / (lag_values + epsilon)
    else:
        alerts["growth_rate"] = 0.0
        logger.warning(f"Lag column '{lag_col}' not found, growth rate set to 0")
    
    # High load alert
    y_pred_threshold = np.percentile(y_pred, high_load_percentile)
    alerts["high_load_alert"] = (
        (alerts["y_pred"] >= y_pred_threshold) & 
        (alerts["growth_rate"] > high_load_growth_threshold)
    ).astype(int)
    
    # Underperforming alert
    if y_true is not None:
        underperform_threshold = np.percentile(y_pred, underperform_percentile)
        alerts["underperforming_alert"] = (
            (alerts["y_pred"] >= underperform_threshold) &
            (y_true < y_pred) &
            (alerts["rel_error"] > underperform_error_threshold)
        ).astype(int)
    else:
        alerts["underperforming_alert"] = 0
    
    # Classify alert type
    def classify_alert(row):
        if row["high_load_alert"] == 1 and row["underperforming_alert"] == 1:
            return "both"
        elif row["high_load_alert"] == 1:
            return "high_load"
        elif row["underperforming_alert"] == 1:
            return "underperforming"
        else:
            return "none"
    
    alerts["alert_type"] = alerts.apply(classify_alert, axis=1)
    
    # Log summary
    alert_counts = alerts["alert_type"].value_counts()
    total_alerts = (alerts["alert_type"] != "none").sum()
    
    logger.info(f"Generated {total_alerts} alerts ({100*total_alerts/len(alerts):.1f}%)")
    
    return alerts


# =============================================================================
# Feature Importance
# =============================================================================
def get_feature_importance(
    model: Pipeline,
    feature_names: List[str] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract feature importance from trained pipeline.
    
    Parameters
    ----------
    model : Pipeline
        Trained sklearn Pipeline with XGBoost model.
    feature_names : list of str, optional
        Feature names for labeling.
    top_n : int
        Number of top features to return.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature, importance columns sorted by importance.
    """
    try:
        xgb_model = model.named_steps["model"]
        importances = xgb_model.feature_importances_
    except (KeyError, AttributeError) as e:
        logger.error(f"Could not extract feature importance: {e}")
        return pd.DataFrame(columns=["feature", "importance"])
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Handle case where feature names don't match
    n_features = len(importances)
    if len(feature_names) != n_features:
        logger.warning(
            f"Feature names length ({len(feature_names)}) doesn't match "
            f"importance length ({n_features}). Using generic names."
        )
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    importance_df = pd.DataFrame({
        "feature": feature_names[:n_features],
        "importance": importances,
    })
    
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    return importance_df.head(top_n)


# =============================================================================
# Retraining Decision
# =============================================================================
def should_retrain(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float] = None,
    drift_summary: Dict[str, Any] = None,
    config: Union[PipelineConfig, None] = None,
) -> Tuple[bool, List[str]]:
    """
    Determine if model retraining is recommended.
    
    Parameters
    ----------
    current_metrics : dict
        Current model metrics (must include 'mae' and/or 'r2').
    baseline_metrics : dict, optional
        Previous/baseline metrics for comparison.
    drift_summary : dict, optional
        Drift detection summary from run_drift_checks().
    config : PipelineConfig, optional
        Configuration object for thresholds.
        
    Returns
    -------
    should_retrain : bool
        True if retraining is recommended.
    reasons : list of str
        List of reasons triggering retraining.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    retrain_config = config.retraining
    reasons = []
    
    # Check R² threshold
    if "r2" in current_metrics:
        if current_metrics["r2"] < retrain_config.r2_min_threshold:
            reasons.append(
                f"R² ({current_metrics['r2']:.4f}) below minimum threshold "
                f"({retrain_config.r2_min_threshold})"
            )
    
    # Check MAE degradation
    if baseline_metrics and "mae" in current_metrics and "mae" in baseline_metrics:
        baseline_mae = baseline_metrics["mae"]
        current_mae = current_metrics["mae"]
        if baseline_mae > 0:
            mae_increase_pct = ((current_mae - baseline_mae) / baseline_mae) * 100
            if mae_increase_pct > retrain_config.mae_degradation_pct:
                reasons.append(
                    f"MAE increased by {mae_increase_pct:.1f}% "
                    f"(threshold: {retrain_config.mae_degradation_pct}%)"
                )
    
    # Check drift
    if drift_summary:
        n_drift = drift_summary.get("features_with_drift", 0)
        if n_drift >= retrain_config.drift_count_threshold:
            reasons.append(
                f"{n_drift} features show drift "
                f"(threshold: {retrain_config.drift_count_threshold})"
            )
    
    should_retrain_flag = len(reasons) > 0
    
    if should_retrain_flag:
        logger.warning(f"Retraining recommended: {reasons}")
    else:
        logger.info("No retraining needed")
    
    return should_retrain_flag, reasons


# =============================================================================
# Module Entry Point
# =============================================================================
if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("Model evaluation module loaded.")
    print("\nAvailable functions:")
    print("  - evaluate_model(model, X, y)")
    print("  - run_cv_evaluation(X, y, dates, build_fn)")
    print("  - run_drift_checks(df)")
    print("  - generate_alerts(df, y_pred, y_true)")
    print("  - get_feature_importance(model)")
    print("  - should_retrain(current_metrics, baseline, drift)")
    
    print(f"\nDefault drift threshold: {DEFAULT_CONFIG.drift.threshold_pct}%")
    print(f"Default alert thresholds:")
    print(f"  High load: top {100-DEFAULT_CONFIG.alert.high_load_volume_percentile}% + {DEFAULT_CONFIG.alert.high_load_growth_threshold:.0%} growth")
    print(f"  Underperform: top {100-DEFAULT_CONFIG.alert.underperform_volume_percentile}% + {DEFAULT_CONFIG.alert.underperform_rel_error_threshold:.0%} error")
