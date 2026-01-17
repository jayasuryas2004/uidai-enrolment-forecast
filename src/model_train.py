"""
UIDAI ASRIS - Model Training Module

This module provides production-ready functions for:
- Building the champion XGBoost pipeline
- Training with early stopping and validation
- Saving models with versioned filenames and metadata
- Loading saved models for inference

All functions include input validation, type hints, and defensive checks
to ensure robustness in production environments.

Author: UIDAI ASRIS Team
Version: 1.1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any, Union
import json
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Local imports
from .config import PipelineConfig, ModelConfig, DEFAULT_CONFIG, PROJECT_ROOT


# =============================================================================
# Module Logger
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation
# =============================================================================
def _validate_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    context: str = "Training",
) -> None:
    """
    Validate training data before model fitting.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    context : str
        Context for error messages.
        
    Raises
    ------
    ValueError
        If validation fails.
    """
    if X is None or y is None:
        raise ValueError(f"{context}: X and y cannot be None")
    
    if len(X) != len(y):
        raise ValueError(
            f"{context}: X and y must have same length. "
            f"Got X={len(X)}, y={len(y)}"
        )
    
    if len(X) == 0:
        raise ValueError(f"{context}: Empty training data")
    
    if y.isna().any():
        n_na = y.isna().sum()
        raise ValueError(f"{context}: Target has {n_na} missing values")


# =============================================================================
# Pipeline Building
# =============================================================================
def build_champion_pipeline(
    numeric_features: List[str],
    categorical_features: List[str] = None,
    hyperparams: Dict[str, Any] = None,
    config: Union[PipelineConfig, ModelConfig, None] = None,
) -> Pipeline:
    """
    Build a fresh (unfitted) Pipeline with the champion model configuration.
    
    The pipeline consists of:
    1. ColumnTransformer for preprocessing (OneHotEncoder for categoricals)
    2. XGBRegressor with production-tuned hyperparameters
    
    Parameters
    ----------
    numeric_features : list of str
        List of numeric feature column names.
    categorical_features : list of str, optional
        List of categorical feature column names.
    hyperparams : dict, optional
        XGBoost hyperparameters. Uses config defaults if not provided.
    config : PipelineConfig or ModelConfig, optional
        Configuration object for model hyperparameters.
        
    Returns
    -------
    Pipeline
        Fresh sklearn Pipeline with preprocessing + XGBRegressor.
        
    Examples
    --------
    >>> pipeline = build_champion_pipeline(
    ...     numeric_features=['total_enrolment', 'growth_rate'],
    ...     categorical_features=['region']
    ... )
    """
    # Resolve configuration
    if config is None:
        model_config = DEFAULT_CONFIG.model
    elif isinstance(config, PipelineConfig):
        model_config = config.model
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        model_config = DEFAULT_CONFIG.model
    
    if hyperparams is None:
        hyperparams = model_config.hyperparams.copy()
    
    if categorical_features is None:
        categorical_features = []
    
    logger.info(f"Building pipeline with {len(numeric_features)} numeric, {len(categorical_features)} categorical features")
    
    # Build transformers
    transformers = []
    
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
             categorical_features)
        )
    
    if numeric_features:
        transformers.append(
            ("num", "passthrough", numeric_features)
        )
    
    if not transformers:
        raise ValueError("At least one feature type (numeric or categorical) is required")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    # Build XGBoost model (exclude early_stopping_rounds from constructor)
    model_params = {k: v for k, v in hyperparams.items() 
                    if k != "early_stopping_rounds"}
    model_params["n_jobs"] = -1  # Use all available cores
    model_params["verbosity"] = 0  # Suppress XGBoost warnings
    
    xgb_model = XGBRegressor(**model_params)
    
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb_model),
    ])


# =============================================================================
# Model Training
# =============================================================================
def train_champion_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
    hyperparams: Dict[str, Any] = None,
    use_early_stopping: bool = True,
    verbose: bool = True,
    config: Union[PipelineConfig, ModelConfig, None] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train the champion XGBoost model with optional early stopping.
    
    This function:
    1. Validates input data
    2. Auto-detects feature types if not provided
    3. Builds the preprocessing + model pipeline
    4. Trains with early stopping (if validation set provided)
    5. Computes and returns training metrics
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame, optional
        Validation features (required if use_early_stopping=True).
    y_val : pd.Series, optional
        Validation target (required if use_early_stopping=True).
    numeric_features : list of str, optional
        Numeric feature names. Auto-detected if not provided.
    categorical_features : list of str, optional
        Categorical feature names. Auto-detected if not provided.
    hyperparams : dict, optional
        Model hyperparameters.
    use_early_stopping : bool, default=True
        Whether to use early stopping.
    verbose : bool, default=True
        Print training progress.
    config : PipelineConfig or ModelConfig, optional
        Configuration object.
        
    Returns
    -------
    pipeline : Pipeline
        Trained pipeline.
    train_info : dict
        Training information (metrics, best_iteration, hyperparams, etc.).
        
    Raises
    ------
    ValueError
        If validation data is missing when early stopping is requested.
    """
    # Resolve configuration
    if config is None:
        model_config = DEFAULT_CONFIG.model
    elif isinstance(config, PipelineConfig):
        model_config = config.model
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        model_config = DEFAULT_CONFIG.model
    
    if hyperparams is None:
        hyperparams = model_config.hyperparams.copy()
    
    # Validate training data
    _validate_training_data(X_train, y_train, "Training")
    
    if use_early_stopping:
        if X_val is None or y_val is None:
            logger.warning("Early stopping requested but no validation set provided. Disabling.")
            use_early_stopping = False
        else:
            _validate_training_data(X_val, y_val, "Validation")
    
    # Auto-detect feature types if not provided
    if numeric_features is None:
        numeric_features = [c for c in X_train.columns 
                           if X_train[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    if categorical_features is None:
        categorical_features = [c for c in X_train.columns 
                                if X_train[c].dtype == 'object']
    
    # Build pipeline
    pipeline = build_champion_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        hyperparams=hyperparams,
        config=config,
    )
    
    if verbose:
        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    logger.info(f"Training model with {len(X_train):,} samples")
    
    train_info = {
        "n_train": len(X_train),
        "n_features": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "hyperparams": hyperparams.copy(),
        "trained_at": datetime.now().isoformat(),
    }
    
    if use_early_stopping and X_val is not None and y_val is not None:
        if verbose:
            print(f"Validation samples: {len(X_val):,}")
            print(f"Early stopping rounds: {hyperparams.get('early_stopping_rounds', 30)}")
            print("-" * 60)
        
        # Fit preprocessor first
        pipeline.named_steps["preprocess"].fit(X_train)
        X_train_transformed = pipeline.named_steps["preprocess"].transform(X_train)
        X_val_transformed = pipeline.named_steps["preprocess"].transform(X_val)
        
        # Set early stopping and fit model
        es_rounds = hyperparams.get("early_stopping_rounds", 30)
        pipeline.named_steps["model"].set_params(early_stopping_rounds=es_rounds)
        
        pipeline.named_steps["model"].fit(
            X_train_transformed,
            y_train,
            eval_set=[
                (X_train_transformed, y_train),
                (X_val_transformed, y_val),
            ],
            verbose=50 if verbose else 0,
        )
        
        train_info["best_iteration"] = pipeline.named_steps["model"].best_iteration
        train_info["best_score"] = float(pipeline.named_steps["model"].best_score)
        train_info["n_val"] = len(X_val)
        train_info["used_early_stopping"] = True
        
        if verbose:
            print("-" * 60)
            print(f"✓ Best iteration: {train_info['best_iteration']}")
            print(f"✓ Best validation RMSE: {train_info['best_score']:.4f}")
        
        logger.info(f"Best iteration: {train_info['best_iteration']}, best score: {train_info['best_score']:.4f}")
    else:
        # Simple fit without early stopping
        pipeline.fit(X_train, y_train)
        train_info["best_iteration"] = hyperparams.get("n_estimators", 500)
        train_info["used_early_stopping"] = False
        
        if verbose:
            print("✓ Training complete (no early stopping)")
        
        logger.info("Training complete (no early stopping)")
    
    # Compute training metrics
    y_train_pred = pipeline.predict(X_train)
    train_info["train_mae"] = float(mean_absolute_error(y_train, y_train_pred))
    train_info["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_info["train_r2"] = float(r2_score(y_train, y_train_pred))
    
    if X_val is not None and y_val is not None:
        y_val_pred = pipeline.predict(X_val)
        train_info["val_mae"] = float(mean_absolute_error(y_val, y_val_pred))
        train_info["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        train_info["val_r2"] = float(r2_score(y_val, y_val_pred))
    
    if verbose:
        print(f"\nTraining Metrics:")
        print(f"  MAE:  {train_info['train_mae']:.2f}")
        print(f"  RMSE: {train_info['train_rmse']:.2f}")
        print(f"  R²:   {train_info['train_r2']:.4f}")
        if "val_mae" in train_info:
            print(f"\nValidation Metrics:")
            print(f"  MAE:  {train_info['val_mae']:.2f}")
            print(f"  RMSE: {train_info['val_rmse']:.2f}")
            print(f"  R²:   {train_info['val_r2']:.4f}")
        print("=" * 60)
    
    return pipeline, train_info


# =============================================================================
# Model Saving
# =============================================================================
def save_model(
    model: Pipeline,
    model_dir: Union[str, Path] = None,
    model_name: str = None,
    version: str = None,
    train_info: Dict[str, Any] = None,
    config: Union[PipelineConfig, ModelConfig, None] = None,
) -> Tuple[Path, Path]:
    """
    Save trained model with versioned filename and metadata.
    
    This function:
    1. Creates the model directory if needed
    2. Saves the model using joblib (efficient for sklearn pipelines)
    3. Saves metadata JSON with training info and version
    
    Parameters
    ----------
    model : Pipeline
        Trained sklearn Pipeline to save.
    model_dir : str or Path, optional
        Directory to save the model. Uses config default if not provided.
    model_name : str, optional
        Base name for the model file.
    version : str, optional
        Version string (e.g., '202511'). Auto-generated if not provided.
    train_info : dict, optional
        Training information to save as metadata.
    config : PipelineConfig or ModelConfig, optional
        Configuration object for defaults.
        
    Returns
    -------
    model_path : Path
        Path to saved model file.
    metadata_path : Path
        Path to saved metadata file.
        
    Examples
    --------
    >>> model_path, meta_path = save_model(pipeline, version="202511")
    >>> print(f"Model saved to: {model_path}")
    """
    # Resolve configuration
    if config is None:
        model_config = DEFAULT_CONFIG.model
    elif isinstance(config, PipelineConfig):
        model_config = config.model
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        model_config = DEFAULT_CONFIG.model
    
    if model_dir is None:
        model_dir = model_config.model_dir
    if model_name is None:
        model_name = model_config.model_name
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate version if not provided
    if version is None:
        version = datetime.now().strftime("%Y%m")
    
    # Save model
    model_filename = f"{model_name}_{version}.joblib"
    model_path = model_dir / model_filename
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Build metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "saved_at": datetime.now().isoformat(),
        "file_path": str(model_path),
        "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
    }
    
    if train_info:
        # Convert numpy types to Python types for JSON serialization
        clean_info = {}
        for k, v in train_info.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_info[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_info[k] = v.tolist()
            elif isinstance(v, dict):
                clean_info[k] = {
                    kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                clean_info[k] = v
            else:
                clean_info[k] = v
        metadata["train_info"] = clean_info
    
    # Save metadata
    metadata_filename = f"{model_name}_{version}_metadata.json"
    metadata_path = model_dir / metadata_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model_path, metadata_path


def load_model(model_path: Union[str, Path]) -> Pipeline:
    """
    Load a saved model.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the saved model file.
        
    Returns
    -------
    Pipeline
        Loaded sklearn Pipeline ready for predictions.
        
    Raises
    ------
    FileNotFoundError
        If model file does not exist.
        
    Examples
    --------
    >>> model = load_model("models/uidai_xgb_champion_202511.joblib")
    >>> predictions = model.predict(X_new)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"✓ Model loaded: {model_path}")
    
    return model


def load_model_with_metadata(
    model_path: Union[str, Path],
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load a saved model along with its metadata.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the saved model file.
        
    Returns
    -------
    model : Pipeline
        Loaded sklearn Pipeline.
    metadata : dict
        Model metadata including training info.
    """
    model_path = Path(model_path)
    model = load_model(model_path)
    
    # Try to load metadata
    metadata_path = model_path.with_suffix("").with_name(
        model_path.stem + "_metadata.json"
    )
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")
    
    return model, metadata


# =============================================================================
# Module Entry Point
# =============================================================================
if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("Model training module loaded.")
    print(f"Default model directory: {DEFAULT_CONFIG.model.model_dir}")
    print(f"\nChampion hyperparameters:")
    for k, v in DEFAULT_CONFIG.model.hyperparams.items():
        print(f"  {k}: {v}")
