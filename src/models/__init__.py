# src/models/__init__.py
"""
Model experiment modules for Phase-4 forecasting.

This package contains experimental model implementations that can be
compared against the frozen v3 XGBoost baseline.

Modules:
    - phase4_lgbm_experiment: LightGBM experiment with same CV/features as v3
    - phase4_ensemble_eval: Weighted blend ensemble evaluator

Example Usage:
    from src.models.phase4_lgbm_experiment import Phase4LGBMExperiment
    from src.models.phase4_ensemble_eval import Phase4EnsembleEval
"""

from .phase4_lgbm_experiment import (
    Phase4LGBMExperiment,
    CVFoldResult,
    DEFAULT_LGBM_PARAMS,
)
from .phase4_ensemble_eval import (
    Phase4EnsembleEval,
    EnsembleResult,
)

__all__ = [
    "Phase4LGBMExperiment",
    "CVFoldResult",
    "DEFAULT_LGBM_PARAMS",
    "Phase4EnsembleEval",
    "EnsembleResult",
]
