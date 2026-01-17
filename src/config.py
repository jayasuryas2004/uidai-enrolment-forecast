"""
UIDAI ASRIS - Centralized Configuration Module

This module provides a single source of truth for all configuration parameters,
thresholds, and constants used across the forecasting pipeline. Having centralized
configuration enables:
- Easy tuning without code changes
- Clear documentation of all parameters
- Consistent behavior across modules
- Simple deployment configuration

Author: UIDAI ASRIS Team
Version: 1.0.0
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


# =============================================================================
# Environment Detection
# =============================================================================
def _get_project_root() -> Path:
    """Detect project root by looking for known markers."""
    current = Path(__file__).resolve().parent.parent
    markers = ["run_training.py", "notebooks", "data"]
    
    for _ in range(5):  # Max 5 levels up
        if all((current / marker).exists() for marker in markers[:1]):
            return current
        current = current.parent
    
    # Fallback to parent of src/
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _get_project_root()


# =============================================================================
# Data Configuration
# =============================================================================
@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Paths (relative to project root)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    modeling_csv: str = "district_month_modeling.csv"
    
    # Column names
    date_col: str = "month_date"
    id_cols: List[str] = field(default_factory=lambda: ["state", "district"])
    target_col: str = "target_enrolment_next_month"
    
    # Train/validation split
    val_fraction: float = 0.2  # 20% for validation (chronological)
    
    # Feature exclusion patterns (to prevent leakage)
    leakage_patterns: List[str] = field(default_factory=lambda: ["_next_month"])
    
    @property
    def data_path(self) -> Path:
        """Full path to the modeling CSV."""
        return self.data_dir / self.modeling_csv
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 < self.val_fraction < 1:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if not self.target_col:
            raise ValueError("target_col cannot be empty")
        if not self.date_col:
            raise ValueError("date_col cannot be empty")


# =============================================================================
# Model Configuration
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration for the champion XGBoost model."""
    
    # XGBoost hyperparameters (tuned for UIDAI enrolment data)
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.5  # L2 regularization
    reg_alpha: float = 0.2   # L1 regularization
    random_state: int = 42
    
    # Early stopping
    early_stopping_rounds: int = 30
    
    # Model output
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    model_name: str = "uidai_xgb_champion"
    
    @property
    def hyperparams(self) -> Dict[str, Any]:
        """Return hyperparameters as dictionary for XGBoost."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "random_state": self.random_state,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")


# =============================================================================
# Drift Detection Configuration
# =============================================================================
@dataclass
class DriftConfig:
    """Configuration for drift detection and monitoring."""
    
    # Features to monitor for drift
    drift_features: List[str] = field(default_factory=lambda: [
        "total_enrolment",
        "total_enrolment_prev_1",
        "enrolment_diff_1",
        "total_demo_updates",
        "total_bio_updates",
    ])
    
    # Drift detection parameters
    reference_periods: int = 3  # Number of periods to use as baseline
    threshold_pct: float = 30.0  # % change to flag as drift
    
    # Volume drop threshold for incomplete data detection
    volume_drop_threshold_pct: float = 50.0
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be > 0, got {self.threshold_pct}")
        if self.reference_periods < 1:
            raise ValueError(f"reference_periods must be >= 1, got {self.reference_periods}")


# =============================================================================
# Alert Configuration
# =============================================================================
@dataclass
class AlertConfig:
    """Configuration for alert generation thresholds."""
    
    # High load alert thresholds
    high_load_volume_percentile: int = 90  # Top 10% by predicted volume
    high_load_growth_threshold: float = 0.20  # 20% growth rate
    
    # Underperforming alert thresholds
    underperform_volume_percentile: int = 70  # Top 30% by predicted volume
    underperform_rel_error_threshold: float = 0.30  # 30% relative error
    
    # Small value epsilon for division safety
    epsilon: float = 1e-6
    
    # Lag column for growth calculation
    lag_col: str = "total_enrolment_prev_1"
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.high_load_volume_percentile <= 100:
            raise ValueError(f"high_load_volume_percentile must be in [0, 100]")
        if not 0 <= self.underperform_volume_percentile <= 100:
            raise ValueError(f"underperform_volume_percentile must be in [0, 100]")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0")


# =============================================================================
# Retraining Configuration
# =============================================================================
@dataclass
class RetrainingConfig:
    """Configuration for automated retraining decisions."""
    
    # Performance degradation thresholds
    mae_degradation_pct: float = 10.0  # Trigger if MAE increases by 10%
    r2_min_threshold: float = 0.50  # Trigger if R² drops below 0.5
    
    # Drift-based retraining
    drift_count_threshold: int = 3  # Trigger if N features show drift
    
    # Minimum data requirements
    min_training_samples: int = 1000
    min_validation_samples: int = 100
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.mae_degradation_pct <= 0:
            raise ValueError(f"mae_degradation_pct must be > 0")
        if not 0 <= self.r2_min_threshold <= 1:
            raise ValueError(f"r2_min_threshold must be in [0, 1]")


# =============================================================================
# Cross-Validation Configuration
# =============================================================================
@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    
    n_splits: int = 4  # Number of time-series CV folds
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")


# =============================================================================
# Master Configuration
# =============================================================================
@dataclass
class PipelineConfig:
    """
    Master configuration for the entire UIDAI forecasting pipeline.
    
    Usage:
        config = PipelineConfig()  # Use defaults
        config.validate()  # Check all values
        
        # Or customize:
        config = PipelineConfig(
            data=DataConfig(val_fraction=0.25),
            model=ModelConfig(max_depth=6),
        )
    """
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.data.validate()
        self.model.validate()
        self.drift.validate()
        self.alert.validate()
        self.retraining.validate()
        self.cv.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/serialization."""
        return {
            "project_root": str(PROJECT_ROOT),
            "data": {
                "data_path": str(self.data.data_path),
                "date_col": self.data.date_col,
                "target_col": self.data.target_col,
                "val_fraction": self.data.val_fraction,
            },
            "model": self.model.hyperparams,
            "drift": {
                "features": self.drift.drift_features,
                "threshold_pct": self.drift.threshold_pct,
            },
            "alert": {
                "high_load_percentile": self.alert.high_load_volume_percentile,
                "high_load_growth": self.alert.high_load_growth_threshold,
                "underperform_percentile": self.alert.underperform_volume_percentile,
                "underperform_error": self.alert.underperform_rel_error_threshold,
            },
        }


# =============================================================================
# Default Instance
# =============================================================================
# Create a default configuration instance for easy import
DEFAULT_CONFIG = PipelineConfig()


# =============================================================================
# Security Note
# =============================================================================
# This module intentionally does NOT include any:
# - Database connection strings
# - API keys or secrets
# - Passwords or credentials
#
# For production deployment, use environment variables or a secrets manager
# (e.g., AWS Secrets Manager, Azure Key Vault, HashiCorp Vault) for sensitive data.


if __name__ == "__main__":
    # Validate and display configuration
    config = PipelineConfig()
    config.validate()
    print("✓ Configuration validated successfully")
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data path: {config.data.data_path}")
    print(f"Model directory: {config.model.model_dir}")
    print(f"\nModel hyperparameters:")
    for k, v in config.model.hyperparams.items():
        print(f"  {k}: {v}")
