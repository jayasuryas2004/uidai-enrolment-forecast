"""
phase4_model_registry.py
========================

Simple model registry for Phase-4 experiments.

╔════════════════════════════════════════════════════════════════════════════╗
║  IMPORTANT: Phase-4 v2 and v3 models are FROZEN as official baselines.     ║
║  DO NOT overwrite their artifacts. All experiments must save to exp_dir.   ║
╚════════════════════════════════════════════════════════════════════════════╝

This module provides:
    1. Phase4ModelPaths - Frozen dataclass defining artifact paths (v2)
    2. Phase4V3FinalPaths - Frozen dataclass for v3 baseline paths
    3. PHASE4_V2_FINAL - The frozen final v2 model paths (DO NOT OVERWRITE)
    4. PHASE4_V3_FINAL - The frozen final v3 baseline paths (DO NOT OVERWRITE)
    5. get_experiment_paths() - Get paths for new experiments
    6. check_not_overwriting_final() - Safety check for v2
    7. check_not_overwriting_v3_final() - Safety check for v3

Author: UIDAI Forecast Team
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set


# =============================================================================
# Frozen Dataclass for Model Paths
# =============================================================================

@dataclass(frozen=True)
class Phase4ModelPaths:
    """
    Paths for final and experimental Phase-4 models and artifacts.

    This is a frozen (immutable) dataclass to prevent accidental modifications.

    Attributes
    ----------
    final_model : Path
        Path to the final trained XGBoost model (.pkl).
    final_encoders : Path
        Path to the LabelEncoders used for categorical columns (.pkl).
    final_metrics : Path
        Path to the evaluation metrics JSON.
    final_params : Path
        Path to the best hyperparameters JSON.
    final_search_results : Path
        Path to the random search results CSV.
    exp_dir : Path
        Base directory for all experiments (new work goes here).
    """

    final_model: Path
    final_encoders: Path
    final_metrics: Path
    final_params: Path
    final_search_results: Path
    exp_dir: Path

    def all_final_paths(self) -> Set[Path]:
        """Return set of all final artifact paths (for protection checks)."""
        return {
            self.final_model,
            self.final_encoders,
            self.final_metrics,
            self.final_params,
            self.final_search_results,
        }


# =============================================================================
# Phase-4 v3 Frozen Dataclass (Leakage-Safe Baseline)
# =============================================================================

@dataclass(frozen=True)
class Phase4V3FinalPaths:
    """
    Paths for the FINAL Phase-4 v3 baseline model (leakage-safe, production-style).

    This is the official model for UIDAI forecasting, validated with:
        - Expanding-window time-series CV (4 folds, 1-month gap)
        - Leakage-safe feature engineering (lags, rolling, holidays, policy)
        - CV metrics: R² ≈ 0.95, MAE ≈ 124

    Attributes
    ----------
    final_model : Path
        Path to the final trained XGBoost model (.pkl).
    final_encoders : Path
        Path to the LabelEncoders used for categorical columns (.pkl).
    final_metrics : Path
        Path to the evaluation metrics JSON.
    final_params : Path
        Path to the hyperparameters JSON.
    exp_dir : Path
        Base directory for v3 experiments (new work goes here).
    """

    final_model: Path
    final_encoders: Path
    final_metrics: Path
    final_params: Path
    exp_dir: Path

    def all_final_paths(self) -> Set[Path]:
        """Return set of all final artifact paths (for protection checks)."""
        return {
            self.final_model,
            self.final_encoders,
            self.final_metrics,
            self.final_params,
        }


# =============================================================================
# FROZEN FINAL MODEL - DO NOT OVERWRITE
# =============================================================================

PHASE4_V2_FINAL = Phase4ModelPaths(
    # =========================================================================
    # ⚠️  THESE ARE THE FINAL PHASE-4 v2 ARTIFACTS - DO NOT OVERWRITE ⚠️
    # =========================================================================
    final_model=Path("artifacts/xgb_phase4_v2_tuned_best.pkl"),
    final_encoders=Path("artifacts/xgb_phase4_v2_tuned_best.encoders.pkl"),
    final_metrics=Path("artifacts/xgb_phase4_v2_tuned_best_metrics.json"),
    final_params=Path("artifacts/xgb_phase4_v2_tuned_best_params.json"),
    final_search_results=Path("artifacts/xgb_phase4_v2_random_search_results.csv"),
    # =========================================================================
    # All new experiments MUST save under this directory
    # =========================================================================
    exp_dir=Path("artifacts/experiments/phase4_v2"),
)


# =============================================================================
# FROZEN FINAL v3 BASELINE - DO NOT OVERWRITE
# =============================================================================
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  🔒 PRODUCTION MODEL LOCK (Jan 2026)                                       ║
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║  PHASE4_V3_FINAL is the ONLY production model for UIDAI forecasting.       ║
# ║                                                                            ║
# ║  LightGBM experiment (src/models/phase4_lgbm_experiment.py) was tested     ║
# ║  but underperformed XGBoost v3:                                            ║
# ║    - XGBoost v3: R² = 0.955, MAE = 116.6                                   ║
# ║    - LightGBM:   R² = 0.865, MAE = 242.8                                   ║
# ║                                                                            ║
# ║  LightGBM is NOT registered here and is kept as a documented experiment.   ║
# ║  Next focus: SHAP explainability + uncertainty quantification.             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

PHASE4_V3_FINAL = Phase4V3FinalPaths(
    # =========================================================================
    # ⚠️  THESE ARE THE FINAL PHASE-4 v3 BASELINE ARTIFACTS - DO NOT OVERWRITE ⚠️
    # =========================================================================
    # Model: XGBoost with leakage-safe time-series features
    # Validation: Expanding-window CV (4 folds, 1-month gap)
    # CV Metrics: R² ≈ 0.95, MAE ≈ 124 (stable across folds)
    # =========================================================================
    final_model=Path("artifacts/xgb_phase4_v3_baseline.pkl"),
    final_encoders=Path("artifacts/xgb_phase4_v3_baseline_encoders.pkl"),
    final_metrics=Path("artifacts/xgb_phase4_v3_baseline_metrics.json"),
    final_params=Path("artifacts/xgb_phase4_v3_baseline_params.json"),
    # =========================================================================
    # All new v3 experiments MUST save under this directory
    # =========================================================================
    exp_dir=Path("artifacts/experiments/phase4_v3"),
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_experiment_paths(exp_name: str, model_family: str = "phase4_v2") -> Dict[str, Path]:
    """
    Return a dict of Paths for saving experiment-specific artifacts
    without touching the final model files.

    Parameters
    ----------
    exp_name : str
        Name of the experiment (e.g., "exp1_small_lr_tweak", "exp2_drop_festival").
        Will be used as a subdirectory name under exp_dir.
    model_family : str
        Model family: "phase4_v2" or "phase4_v3". Default is "phase4_v2".

    Returns
    -------
    Dict[str, Path]
        Dictionary with keys:
            - "dir": Experiment directory (will be created if needed)
            - "model": Path to save model.pkl
            - "encoders": Path to save encoders.pkl
            - "metrics": Path to save metrics.json
            - "params": Path to save params.json
            - "search_results": Path to save random_search_results.csv

    Example
    -------
    >>> paths = get_experiment_paths("exp1_smaller_lr", model_family="phase4_v3")
    >>> paths["model"]
    PosixPath('artifacts/experiments/phase4_v3/exp1_smaller_lr/model.pkl')
    """
    if model_family == "phase4_v3":
        exp_dir = PHASE4_V3_FINAL.exp_dir / exp_name
    else:
        exp_dir = PHASE4_V2_FINAL.exp_dir / exp_name
    
    return {
        "dir": exp_dir,
        "model": exp_dir / "model.pkl",
        "encoders": exp_dir / "encoders.pkl",
        "metrics": exp_dir / "metrics.json",
        "params": exp_dir / "params.json",
        "search_results": exp_dir / "random_search_results.csv",
    }


def check_not_overwriting_final(output_path: Path) -> None:
    """
    Raise RuntimeError if output_path matches any FINAL Phase-4 v2 artifact.

    Call this BEFORE saving any artifact to ensure we don't accidentally
    overwrite the frozen final model.

    Parameters
    ----------
    output_path : Path
        The path where an artifact is about to be saved.

    Raises
    ------
    RuntimeError
        If output_path matches any of the final artifact paths.

    Example
    -------
    >>> check_not_overwriting_final(Path("artifacts/xgb_phase4_v2_tuned_best.pkl"))
    RuntimeError: Refusing to overwrite FINAL Phase-4 v2 artifact: ...
    """
    # Normalize paths for comparison
    output_resolved = output_path.resolve()
    
    for final_path in PHASE4_V2_FINAL.all_final_paths():
        if output_resolved == final_path.resolve() and final_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite FINAL Phase-4 v2 artifact: {final_path}\n"
                f"Use get_experiment_paths() to save to an experiment directory instead."
            )


def check_not_overwriting_v3_final(output_path: Path) -> None:
    """
    Raise RuntimeError if output_path matches any FINAL Phase-4 v3 baseline artifact.

    Call this BEFORE saving any artifact to ensure we don't accidentally
    overwrite the frozen v3 baseline model.

    Parameters
    ----------
    output_path : Path
        The path where an artifact is about to be saved.

    Raises
    ------
    RuntimeError
        If output_path matches any of the v3 final artifact paths.

    Example
    -------
    >>> check_not_overwriting_v3_final(Path("artifacts/xgb_phase4_v3_baseline.pkl"))
    RuntimeError: Refusing to overwrite FINAL Phase-4 v3 baseline artifact: ...
    """
    # Normalize paths for comparison
    output_resolved = output_path.resolve()
    
    for final_path in PHASE4_V3_FINAL.all_final_paths():
        if output_resolved == final_path.resolve() and final_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite FINAL Phase-4 v3 baseline artifact: {final_path}\n"
                f"Use get_experiment_paths(model_family='phase4_v3') to save to an experiment directory instead."
            )


def check_not_overwriting_any_final(output_path: Path) -> None:
    """
    Raise RuntimeError if output_path matches any FINAL artifact (v2 or v3).

    Use this as a universal protection check in training scripts.

    Parameters
    ----------
    output_path : Path
        The path where an artifact is about to be saved.

    Raises
    ------
    RuntimeError
        If output_path matches any frozen final artifact.
    """
    check_not_overwriting_final(output_path)
    check_not_overwriting_v3_final(output_path)


def ensure_experiment_dir(exp_name: str, model_family: str = "phase4_v2") -> Path:
    """
    Create and return the experiment directory for a given experiment name.

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    model_family : str
        Model family: "phase4_v2" or "phase4_v3". Default is "phase4_v2".

    Returns
    -------
    Path
        The created experiment directory.
    """
    paths = get_experiment_paths(exp_name, model_family=model_family)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    return paths["dir"]


def list_experiments(model_family: str = "phase4_v2") -> list[str]:
    """
    List all existing experiment names under exp_dir.

    Parameters
    ----------
    model_family : str
        Model family: "phase4_v2" or "phase4_v3". Default is "phase4_v2".

    Returns
    -------
    list[str]
        List of experiment directory names.
    """
    if model_family == "phase4_v3":
        exp_dir = PHASE4_V3_FINAL.exp_dir
    else:
        exp_dir = PHASE4_V2_FINAL.exp_dir
    
    if not exp_dir.exists():
        return []
    
    return [
        d.name for d in exp_dir.iterdir()
        if d.is_dir()
    ]


# =============================================================================
# Module Info
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE-4 MODEL REGISTRY")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("FINAL v2 Artifact Paths (FROZEN - DO NOT OVERWRITE):")
    print("-" * 70)
    print(f"  Model:          {PHASE4_V2_FINAL.final_model}")
    print(f"  Encoders:       {PHASE4_V2_FINAL.final_encoders}")
    print(f"  Metrics:        {PHASE4_V2_FINAL.final_metrics}")
    print(f"  Params:         {PHASE4_V2_FINAL.final_params}")
    print(f"  Search Results: {PHASE4_V2_FINAL.final_search_results}")
    print(f"  Experiments:    {PHASE4_V2_FINAL.exp_dir}")
    
    print("\n" + "-" * 70)
    print("FINAL v3 BASELINE Artifact Paths (FROZEN - DO NOT OVERWRITE):")
    print("-" * 70)
    print(f"  Model:          {PHASE4_V3_FINAL.final_model}")
    print(f"  Encoders:       {PHASE4_V3_FINAL.final_encoders}")
    print(f"  Metrics:        {PHASE4_V3_FINAL.final_metrics}")
    print(f"  Params:         {PHASE4_V3_FINAL.final_params}")
    print(f"  Experiments:    {PHASE4_V3_FINAL.exp_dir}")
    
    print("\n" + "-" * 70)
    print("Existing Experiments:")
    print("-" * 70)
    
    v2_exps = list_experiments("phase4_v2")
    v3_exps = list_experiments("phase4_v3")
    
    print(f"  v2 experiments: {v2_exps if v2_exps else 'None'}")
    print(f"  v3 experiments: {v3_exps if v3_exps else 'None'}")
    
    print("\n" + "-" * 70)
    print("Example usage:")
    print("-" * 70)
    print("  from src.phase4_model_registry import get_experiment_paths, PHASE4_V3_FINAL")
    print("  paths = get_experiment_paths('exp1_lr_sweep', model_family='phase4_v3')")
    print("  # Save artifacts to paths['model'], paths['metrics'], etc.")
    print("=" * 70)
