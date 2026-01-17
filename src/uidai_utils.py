"""
uidai_utils.py

Utility functions for UIDAI B2_log_ctx_opt (XGBoost) pipeline.

GOALS:
- Provide safe, reusable utilities for:
    - Time-based train/val/test split
    - Regression metrics (MAE, MAPE, RMSE, R2)
    - Safety analysis (error by group)
    - Saving metrics and safety outputs to disk

CONSTRAINTS:
- Time split must avoid leakage: train < val < test strictly by date.
- Safety analysis:
    - Inputs: y_true, y_pred, groups_df (DataFrame with grouping columns, e.g. state, volume_bucket)
    - For each group, compute:
        - MAE
        - MAPE
        - Optional RMSE
    - Compare each group MAE to global MAE.
    - Mark pass_flag = True if group MAE <= mae_factor_threshold * global_MAE
    - Compute overall safety_pct = (number of passing groups / total groups) * 100.

REQUIRED FUNCTIONS TO IMPLEMENT:

1) make_time_splits(df: pd.DataFrame, date_col: str,
                    train_end: str, val_end: str, test_end: str | None = None)
   - Sort df by date_col.
   - Train: df[date <= train_end]
   - Val:   df[(date > train_end) & (date <= val_end)]
   - Test:
        if test_end is not None:
            df[(date > val_end) & (date <= test_end)]
        else:
            df[date > val_end]
   - Return: train_df, val_df, test_df

2) compute_regression_metrics(y_true, y_pred)
   - Return a dict with keys:
       {
         "mae": ...,
         "mape": ...,
         "rmse": ...,
         "r2": ...
       }

3) compute_safety_report(
       y_true,
       y_pred,
       groups_df: pd.DataFrame,
       mae_factor_threshold: float = 1.5,
       mape_threshold: float | None = None
   )
   - Compute global MAE and MAPE.
   - Concatenate y_true, y_pred, groups_df into one DataFrame.
   - Group by all columns in groups_df.
   - For each group, compute MAE, MAPE, RMSE.
   - Determine pass_flag per group based on:
       - group_mae <= mae_factor_threshold * global_mae
       - and if mape_threshold is not None: group_mape <= mape_threshold
   - safety_pct = 100 * (number of groups with pass_flag=True / total groups)
   - Return:
       - safety_df: DataFrame with [group_cols..., mae, mape, rmse, pass_flag]
       - summary: dict with global metrics and safety_pct.

4) save_json(obj, path: Path)
   - Save obj as JSON to the given path.

5) save_metrics_and_safety(metrics_dict: dict,
                           safety_df: pd.DataFrame,
                           out_dir: Path)
   - Save metrics_dict to metrics.json in out_dir.
   - Save safety_df to safety_report.csv in out_dir.

IMPORTANT:
- Use sklearn.metrics:
    - mean_absolute_error
    - mean_squared_error (then sqrt for RMSE)
    - r2_score
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import json
from typing import Tuple, Optional


def make_time_splits(
    df: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_end: str,
    test_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/val/test splits with no overlap.
    
    Train: date <= train_end
    Val:   train_end < date <= val_end
    Test:  date > val_end (up to test_end if provided)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df[date_col] <= train_end_dt].copy()
    val_df = df[(df[date_col] > train_end_dt) & (df[date_col] <= val_end_dt)].copy()
    
    if test_end is not None:
        test_end_dt = pd.to_datetime(test_end)
        test_df = df[(df[date_col] > val_end_dt) & (df[date_col] <= test_end_dt)].copy()
    else:
        test_df = df[df[date_col] > val_end_dt].copy()
    
    return train_df, val_df, test_df


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, MAPE, RMSE, and R² for regression predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE: avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {"mae": float(mae), "mape": float(mape), "rmse": float(rmse), "r2": float(r2)}


def compute_safety_report(
    y_true,
    y_pred,
    groups_df: pd.DataFrame,
    mae_factor_threshold: float = 1.5,
    mape_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute safety report by groups.
    
    Returns:
        safety_df: DataFrame with group-level metrics and pass_flag
        summary: dict with global metrics and safety_pct
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Global metrics
    global_mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    global_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    
    # Build combined dataframe
    eval_df = groups_df.copy().reset_index(drop=True)
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred
    eval_df["abs_error"] = np.abs(y_true - y_pred)
    eval_df["pct_error"] = np.where(y_true != 0, np.abs(y_true - y_pred) / np.abs(y_true) * 100, np.nan)
    
    # Group by all columns in groups_df
    group_cols = list(groups_df.columns)
    
    def group_metrics(g):
        mae = g["abs_error"].mean()
        mape = g["pct_error"].mean()
        rmse = np.sqrt((g["abs_error"] ** 2).mean())
        return pd.Series({"mae": mae, "mape": mape, "rmse": rmse})
    
    safety_df = eval_df.groupby(group_cols, observed=True).apply(group_metrics, include_groups=False).reset_index()
    
    # Determine pass_flag
    safety_df["pass_flag"] = safety_df["mae"] <= mae_factor_threshold * global_mae
    if mape_threshold is not None:
        safety_df["pass_flag"] = safety_df["pass_flag"] & (safety_df["mape"] <= mape_threshold)
    
    # Safety percentage
    safety_pct = 100 * safety_df["pass_flag"].sum() / len(safety_df) if len(safety_df) > 0 else 0.0
    
    summary = {
        "global_mae": float(global_mae),
        "global_mape": float(global_mape),
        "safety_pct": float(safety_pct),
        "n_groups": len(safety_df),
        "n_passing": int(safety_df["pass_flag"].sum()),
    }
    
    return safety_df, summary


def save_json(obj, path: Path) -> None:
    """Save object as JSON to the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_metrics_and_safety(
    metrics_dict: dict, safety_df: pd.DataFrame, out_dir: Path
) -> None:
    """Save metrics to JSON and safety report to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(metrics_dict, out_dir / "metrics.json")
    safety_df.to_csv(out_dir / "safety_report.csv", index=False)
