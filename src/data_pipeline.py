"""
UIDAI ASRIS - Data Pipeline Module

This module provides production-ready functions for:
- Loading raw data with validation
- Cleaning and preprocessing data
- Building features for the champion model
- Creating time-based train/validation splits

All functions include input validation, type hints, and defensive checks
to ensure robustness in production environments.

Author: UIDAI ASRIS Team
Version: 1.1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import logging

# Local imports
from .config import PipelineConfig, DataConfig, DEFAULT_CONFIG, PROJECT_ROOT


# =============================================================================
# Module Logger
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation Helpers
# =============================================================================
def _validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    context: str = "DataFrame",
) -> None:
    """
    Validate that a DataFrame has required columns and is not empty.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_cols : list of str
        Required column names.
    context : str
        Context string for error messages.
        
    Raises
    ------
    ValueError
        If validation fails.
    """
    if df is None:
        raise ValueError(f"{context}: DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{context}: Expected DataFrame, got {type(df).__name__}")
    
    if df.empty:
        raise ValueError(f"{context}: DataFrame is empty (0 rows)")
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: Missing required columns: {missing}")


def _validate_date_range(
    df: pd.DataFrame,
    date_col: str,
    min_periods: int = 3,
) -> None:
    """
    Validate that date column has sufficient time range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column.
    date_col : str
        Name of date column.
    min_periods : int
        Minimum number of unique periods required.
        
    Raises
    ------
    ValueError
        If date range is insufficient.
    """
    dates = pd.to_datetime(df[date_col])
    n_periods = dates.dt.to_period("M").nunique()
    
    if n_periods < min_periods:
        raise ValueError(
            f"Insufficient date range: {n_periods} months found, "
            f"minimum {min_periods} required for reliable modeling"
        )


# =============================================================================
# Data Loading
# =============================================================================
def load_raw_data(
    config: Union[PipelineConfig, DataConfig, Dict[str, Any], None] = None,
) -> pd.DataFrame:
    """
    Load the UIDAI modeling data from configured paths with validation.
    
    This function:
    1. Resolves the data path from configuration
    2. Validates the file exists and is readable
    3. Loads the CSV with appropriate dtypes
    4. Performs basic sanity checks on loaded data
    
    Parameters
    ----------
    config : PipelineConfig, DataConfig, dict, or None
        Configuration object or dictionary with keys:
        - data_dir: Path to data directory
        - modeling_csv: Name of the modeling CSV file
        If None, uses DEFAULT_CONFIG.
        
    Returns
    -------
    pd.DataFrame
        Raw loaded DataFrame.
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the loaded data fails validation.
        
    Examples
    --------
    >>> df = load_raw_data()  # Use default config
    >>> df = load_raw_data({"data_dir": "custom/path"})
    """
    # Resolve configuration
    if config is None:
        csv_path = DEFAULT_CONFIG.data.data_path
    elif isinstance(config, PipelineConfig):
        csv_path = config.data.data_path
    elif isinstance(config, DataConfig):
        csv_path = config.data_path
    elif isinstance(config, dict):
        # Legacy dict-based config
        data_dir = Path(config.get("data_dir", DEFAULT_CONFIG.data.data_dir))
        csv_name = config.get("modeling_csv", DEFAULT_CONFIG.data.modeling_csv)
        csv_path = data_dir / csv_name
    else:
        raise TypeError(f"config must be PipelineConfig, DataConfig, dict, or None")
    
    # Validate file exists
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Please ensure the data pipeline has been run to generate this file."
        )
    
    # Load data
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Basic validation
    if df.empty:
        raise ValueError(f"Loaded file is empty: {csv_path}")
    
    n_rows, n_cols = df.shape
    logger.info(f"✓ Loaded {n_rows:,} rows × {n_cols} columns from {csv_path.name}")
    print(f"✓ Loaded {n_rows:,} rows × {n_cols} columns from {csv_path.name}")
    
    return df


# =============================================================================
# Data Cleaning
# =============================================================================
def clean_data(
    df: pd.DataFrame,
    date_col: str = None,
    id_cols: List[str] = None,
    target_col: str = None,
    config: Union[PipelineConfig, DataConfig, None] = None,
) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw data with validation.
    
    Cleaning steps:
    1. Parse and validate date column
    2. Sort by date and ID columns (chronological order)
    3. Validate target column (no missing values, numeric)
    4. Remove duplicates if present
    5. Reset index for clean indexing
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    date_col : str, optional
        Name of the date column. Uses config default if not provided.
    id_cols : list of str, optional
        List of ID columns (e.g., ['state', 'district']).
    target_col : str, optional
        Name of the target column.
    config : PipelineConfig or DataConfig, optional
        Configuration object for defaults.
        
    Returns
    -------
    pd.DataFrame
        Cleaned and sorted DataFrame.
        
    Raises
    ------
    ValueError
        If required columns are missing or target has issues.
    TypeError
        If target column is not numeric.
    """
    # Resolve config defaults
    if config is None:
        config = DEFAULT_CONFIG
    if isinstance(config, PipelineConfig):
        data_config = config.data
    else:
        data_config = DEFAULT_CONFIG.data
    
    if date_col is None:
        date_col = data_config.date_col
    if id_cols is None:
        id_cols = data_config.id_cols.copy()
    if target_col is None:
        target_col = data_config.target_col
    
    logger.info("Starting data cleaning")
    df = df.copy()
    
    # Validate required columns
    required_cols = [date_col, target_col]
    _validate_dataframe(df, required_cols, context="clean_data")
    
    # Parse date column
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    except Exception as e:
        raise ValueError(f"Failed to parse date column '{date_col}': {e}")
    
    # Validate date range
    _validate_date_range(df, date_col, min_periods=3)
    
    # Sort by date and IDs (chronological order is critical for time series)
    available_id_cols = [c for c in id_cols if c in df.columns]
    sort_cols = [date_col] + available_id_cols
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Check for duplicates
    if available_id_cols:
        dup_cols = [date_col] + available_id_cols
        n_dups = df.duplicated(subset=dup_cols, keep="first").sum()
        if n_dups > 0:
            logger.warning(f"Removing {n_dups} duplicate rows")
            df = df.drop_duplicates(subset=dup_cols, keep="first")
    
    # Validate target column
    n_na = df[target_col].isna().sum()
    if n_na > 0:
        raise ValueError(
            f"Target column '{target_col}' has {n_na} missing values. "
            f"Missing targets must be handled before modeling."
        )
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(
            f"Target column '{target_col}' must be numeric, "
            f"got dtype: {df[target_col].dtype}"
        )
    
    # Validate target range (sanity check for enrolment data)
    if (df[target_col] < 0).any():
        n_neg = (df[target_col] < 0).sum()
        logger.warning(f"Found {n_neg} negative values in target column")
    
    logger.info(f"✓ Cleaned data: {len(df):,} rows, sorted by {sort_cols}")
    print(f"✓ Cleaned data: {len(df):,} rows, sorted by {sort_cols}")
    
    return df


# =============================================================================
# Feature Engineering
# =============================================================================
def build_features(
    df: pd.DataFrame,
    target_col: str = None,
    date_col: str = None,
    id_cols: List[str] = None,
    exclude_patterns: List[str] = None,
    config: Union[PipelineConfig, DataConfig, None] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Build feature matrix X and target vector y from cleaned data.
    
    This function:
    1. Identifies numeric and categorical features
    2. Excludes leakage columns (anything ending with '_next_month')
    3. Excludes ID columns and date columns from features
    4. Returns feature lists for pipeline construction
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output from clean_data).
    target_col : str, optional
        Name of the target column.
    date_col : str, optional
        Name of the date column.
    id_cols : list of str, optional
        List of ID columns to exclude from features.
    exclude_patterns : list of str, optional
        Additional column name patterns to exclude.
    config : PipelineConfig or DataConfig, optional
        Configuration object for defaults.
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    numeric_features : list of str
        List of numeric feature column names.
    categorical_features : list of str
        List of categorical feature column names.
        
    Raises
    ------
    ValueError
        If no features remain after exclusions.
    """
    # Resolve config defaults
    if config is None:
        config = DEFAULT_CONFIG
    if isinstance(config, PipelineConfig):
        data_config = config.data
    else:
        data_config = DEFAULT_CONFIG.data
    
    if target_col is None:
        target_col = data_config.target_col
    if date_col is None:
        date_col = data_config.date_col
    if id_cols is None:
        id_cols = data_config.id_cols.copy()
    if exclude_patterns is None:
        exclude_patterns = data_config.leakage_patterns.copy()
    
    logger.info("Building feature matrix")
    
    # Validate input
    _validate_dataframe(df, [target_col], context="build_features")
    
    # Build exclusion set
    exclude_cols = {target_col, date_col}
    
    # Exclude year_month string column if present
    if "year_month" in df.columns:
        exclude_cols.add("year_month")
    
    # Exclude ID columns
    for c in id_cols:
        if c in df.columns:
            exclude_cols.add(c)
    
    # Exclude leakage patterns (columns that contain future information)
    for col in df.columns:
        for pattern in exclude_patterns:
            if pattern in col and col != target_col:
                exclude_cols.add(col)
    
    # Select feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if not feature_cols:
        raise ValueError(
            f"No features remaining after exclusions. "
            f"Excluded: {sorted(exclude_cols)}"
        )
    
    # Identify categorical vs numeric
    categorical_features = []
    numeric_features = []
    
    for col in feature_cols:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            logger.warning(f"Skipping column with unknown dtype: {col} ({df[col].dtype})")
    
    # Build X and y
    all_features = numeric_features + categorical_features
    X = df[all_features].copy()
    y = df[target_col].copy()
    
    # Validate no NaN in features (could cause issues in some models)
    n_nan_cols = X.isna().any().sum()
    if n_nan_cols > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        logger.warning(f"Found NaN values in {n_nan_cols} feature columns: {nan_cols[:5]}...")
    
    logger.info(f"✓ Built features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"✓ Built features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"  Excluded columns: {sorted(exclude_cols)}")
    
    return X, y, numeric_features, categorical_features


# =============================================================================
# Train/Validation Split
# =============================================================================
def make_train_val_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str = None,
    val_fraction: float = None,
    config: Union[PipelineConfig, DataConfig, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """
    Create time-based train/validation split (no shuffling).
    
    This function performs a chronological split:
    - Training set: older observations
    - Validation set: newer observations
    - No shuffling (preserves temporal order for time series)
    
    Parameters
    ----------
    df : pd.DataFrame
        Full sorted DataFrame (for date reference).
    X : pd.DataFrame
        Feature matrix aligned with df.
    y : pd.Series
        Target vector aligned with df.
    date_col : str, optional
        Name of the date column.
    val_fraction : float, optional
        Fraction of data for validation (0 < val_fraction < 1).
    config : PipelineConfig or DataConfig, optional
        Configuration object for defaults.
        
    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    split_idx : int
        Index where validation begins.
        
    Raises
    ------
    ValueError
        If val_fraction is invalid or data is too small.
    """
    # Resolve config defaults
    if config is None:
        config = DEFAULT_CONFIG
    if isinstance(config, PipelineConfig):
        data_config = config.data
    else:
        data_config = DEFAULT_CONFIG.data
    
    if date_col is None:
        date_col = data_config.date_col
    if val_fraction is None:
        val_fraction = data_config.val_fraction
    
    logger.info(f"Creating train/val split with {val_fraction:.0%} validation")
    
    # Validate
    n = len(df)
    
    if not (0 < val_fraction < 1):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    
    split_idx = int(n * (1 - val_fraction))
    
    # Ensure minimum samples
    n_train = split_idx
    n_val = n - split_idx
    
    MIN_SAMPLES = 100
    if n_train < MIN_SAMPLES:
        raise ValueError(
            f"Training set too small: {n_train} samples (minimum: {MIN_SAMPLES})"
        )
    if n_val < MIN_SAMPLES:
        raise ValueError(
            f"Validation set too small: {n_val} samples (minimum: {MIN_SAMPLES})"
        )
    
    # Split
    X_train = X.iloc[:split_idx].copy()
    X_val = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_val = y.iloc[split_idx:].copy()
    
    # Get date ranges for logging
    train_end = df[date_col].iloc[split_idx - 1]
    val_start = df[date_col].iloc[split_idx]
    val_end = df[date_col].iloc[-1]
    
    logger.info(f"✓ Train: {len(X_train):,} rows (up to {train_end})")
    logger.info(f"✓ Val:   {len(X_val):,} rows ({val_start} to {val_end})")
    
    print(f"✓ Train/Val split:")
    print(f"  Train: {len(X_train):,} rows (up to {train_end})")
    print(f"  Val:   {len(X_val):,} rows ({val_start} to {val_end})")
    
    return X_train, X_val, y_train, y_val, split_idx


# =============================================================================
# Full Pipeline Function
# =============================================================================
def run_data_pipeline(
    config: Union[PipelineConfig, Dict[str, Any], None] = None,
) -> Dict[str, Any]:
    """
    Run the complete data pipeline: load → clean → features → split.
    
    This is the main entry point for data preparation. It orchestrates
    all data processing steps and returns everything needed for model training.
    
    Parameters
    ----------
    config : PipelineConfig, dict, or None
        Configuration object or dictionary.
        
    Returns
    -------
    dict
        Dictionary containing:
        - df: Full cleaned DataFrame
        - X: Full feature matrix
        - y: Full target vector
        - X_train, X_val: Feature matrices
        - y_train, y_val: Target vectors
        - numeric_features, categorical_features: Feature lists
        - split_idx: Split index
        
    Raises
    ------
    FileNotFoundError
        If data file does not exist.
    ValueError
        If data validation fails.
    """
    # Resolve configuration
    if config is None:
        pipe_config = DEFAULT_CONFIG
    elif isinstance(config, PipelineConfig):
        pipe_config = config
    elif isinstance(config, dict):
        # Legacy dict-based config - use defaults with dict overrides
        pipe_config = DEFAULT_CONFIG
    else:
        raise TypeError(f"config must be PipelineConfig, dict, or None")
    
    logger.info("=" * 60)
    logger.info("DATA PIPELINE START")
    logger.info("=" * 60)
    
    print("=" * 60)
    print("DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    df = load_raw_data(config)
    
    # Step 2: Clean
    df = clean_data(df, config=pipe_config)
    
    # Step 3: Build features
    X, y, numeric_features, categorical_features = build_features(df, config=pipe_config)
    
    # Step 4: Split
    X_train, X_val, y_train, y_val, split_idx = make_train_val_split(
        df, X, y, config=pipe_config
    )
    
    logger.info("=" * 60)
    logger.info("DATA PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    print("=" * 60)
    
    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "split_idx": split_idx,
    }


# =============================================================================
# Module Entry Point
# =============================================================================
if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test the pipeline
    print("Testing data pipeline...")
    try:
        result = run_data_pipeline()
        print(f"\n✓ Pipeline complete. Output keys: {list(result.keys())}")
        print(f"  Training samples: {len(result['X_train']):,}")
        print(f"  Validation samples: {len(result['X_val']):,}")
        print(f"  Numeric features: {len(result['numeric_features'])}")
        print(f"  Categorical features: {len(result['categorical_features'])}")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise
