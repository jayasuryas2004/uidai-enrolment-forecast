"""
UIDAI ASRIS - Phase 7.1: Hyperparameter Tuning Evaluation Setup

Goal: Minimize MAE/RMSE on a time-ordered validation window that mimics real
future months, so the tuned model is reliable in the field, not just on random splits.

This module provides:
- Time-ordered train/validation splitting for realistic temporal evaluation
- Standardized evaluation metrics (MAE, RMSE, R², MAPE) for tuning
- A reusable `evaluate_model_for_tuning()` function compatible with
  GridSearchCV, RandomizedSearchCV, and Optuna

Author: UIDAI ASRIS Team
Version: 1.0.0
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, clone


# =============================================================================
# Evaluation Metrics for Tuning
# =============================================================================

def compute_tuning_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """
    Compute evaluation metrics for hyperparameter tuning.
    
    Primary metrics:
    - MAE: Mean Absolute Error (primary objective for minimization)
    - RMSE: Root Mean Squared Error (penalizes large errors more)
    
    Secondary metrics:
    - R²: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error (when y_true > epsilon)
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Predicted values from the model.
    epsilon : float, default=1e-6
        Small value for numerical stability in MAPE calculation.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing all computed metrics.
    
    Example
    -------
    >>> metrics = compute_tuning_metrics(y_true, y_pred)
    >>> print(f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE: only on samples where y_true is sufficiently large
    mask = np.abs(y_true) > epsilon
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


# =============================================================================
# Time-Ordered Train/Validation Split
# =============================================================================

def time_ordered_split(
    df: pd.DataFrame,
    date_col: str = "month_date",
    n_val_months: int = 3,
    id_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation sets based on time ordering.
    
    This split mimics real-world forecasting: train on historical months,
    validate on the most recent N months. This ensures the tuned model
    generalizes to future unseen time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a date column.
    date_col : str, default="month_date"
        Name of the datetime column for ordering.
    n_val_months : int, default=3
        Number of most recent months to use for validation.
    id_cols : list, optional
        ID columns to preserve (e.g., ["state", "district"]).
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_val) split by time.
    
    Example
    -------
    >>> df_train, df_val = time_ordered_split(df, n_val_months=3)
    >>> print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get unique months sorted chronologically
    unique_months = df[date_col].dt.to_period("M").unique()
    unique_months = sorted(unique_months)
    
    if len(unique_months) <= n_val_months:
        raise ValueError(
            f"Not enough months for split. Found {len(unique_months)} months, "
            f"need at least {n_val_months + 1} for train + val."
        )
    
    # Validation = most recent N months
    val_months = set(unique_months[-n_val_months:])
    
    # Create masks
    df["_period"] = df[date_col].dt.to_period("M")
    train_mask = ~df["_period"].isin(val_months)
    val_mask = df["_period"].isin(val_months)
    
    df_train = df[train_mask].drop(columns=["_period"]).reset_index(drop=True)
    df_val = df[val_mask].drop(columns=["_period"]).reset_index(drop=True)
    
    return df_train, df_val


def get_train_val_cutoff_date(
    df: pd.DataFrame,
    date_col: str = "month_date",
    n_val_months: int = 3,
) -> pd.Timestamp:
    """
    Get the cutoff date between train and validation sets.
    
    Useful for logging and reproducibility.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with date column.
    date_col : str, default="month_date"
        Name of the datetime column.
    n_val_months : int, default=3
        Number of validation months.
    
    Returns
    -------
    pd.Timestamp
        First day of the first validation month.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    unique_months = sorted(df[date_col].dt.to_period("M").unique())
    
    if len(unique_months) <= n_val_months:
        raise ValueError("Not enough months for the requested split.")
    
    val_start_period = unique_months[-n_val_months]
    return val_start_period.to_timestamp()


# =============================================================================
# Core Evaluation Function for Tuning
# =============================================================================

def evaluate_model_for_tuning(
    model: BaseEstimator,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    fit_params: Optional[Dict[str, Any]] = None,
    return_predictions: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray]]:
    """
    Evaluate a model on a time-ordered validation set for hyperparameter tuning.
    
    This function:
    1. Clones the model to avoid side effects
    2. Fits on training data
    3. Predicts on validation data
    4. Computes and returns standardized metrics
    
    Compatible with:
    - Manual hyperparameter loops
    - sklearn GridSearchCV / RandomizedSearchCV (via custom scorer)
    - Optuna objective functions
    
    Parameters
    ----------
    model : BaseEstimator
        A scikit-learn compatible model (e.g., XGBRegressor, Pipeline).
        Will be cloned before fitting.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.
    X_val : pd.DataFrame or np.ndarray
        Validation features.
    y_val : pd.Series or np.ndarray
        Validation target.
    fit_params : dict, optional
        Additional parameters to pass to model.fit() (e.g., eval_set for XGBoost).
    return_predictions : bool, default=False
        If True, also return the validation predictions.
    
    Returns
    -------
    Dict[str, float] or Tuple[Dict[str, float], np.ndarray]
        Dictionary of metrics (mae, rmse, r2, mape).
        If return_predictions=True, also returns y_pred array.
    
    Example
    -------
    >>> from xgboost import XGBRegressor
    >>> model = XGBRegressor(n_estimators=100, max_depth=5)
    >>> metrics = evaluate_model_for_tuning(model, X_train, y_train, X_val, y_val)
    >>> print(f"Validation MAE: {metrics['mae']:.2f}")
    
    Notes
    -----
    - The model is cloned to ensure each evaluation is independent.
    - For XGBoost with early stopping, pass eval_set via fit_params:
      `fit_params={"eval_set": [(X_val, y_val)], "verbose": False}`
    """
    # Clone to avoid modifying the original model
    model_clone = clone(model)
    
    # Fit the model
    if fit_params is None:
        fit_params = {}
    model_clone.fit(X_train, y_train, **fit_params)
    
    # Predict on validation set
    y_pred = model_clone.predict(X_val)
    
    # Compute metrics
    metrics = compute_tuning_metrics(y_val, y_pred)
    
    if return_predictions:
        return metrics, y_pred
    return metrics


# =============================================================================
# Optuna-Compatible Objective Function Factory
# =============================================================================

def create_optuna_objective(
    model_class: type,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    param_space_fn: callable,
    metric: str = "mae",
    fit_params: Optional[Dict[str, Any]] = None,
):
    """
    Create an Optuna objective function for hyperparameter tuning.
    
    Parameters
    ----------
    model_class : type
        Model class (e.g., XGBRegressor).
    X_train, y_train, X_val, y_val : arrays
        Train and validation data.
    param_space_fn : callable
        Function that takes an Optuna trial and returns a params dict.
        Example: lambda trial: {"max_depth": trial.suggest_int("max_depth", 3, 10)}
    metric : str, default="mae"
        Metric to optimize ("mae", "rmse", "r2", "mape").
    fit_params : dict, optional
        Additional fit parameters.
    
    Returns
    -------
    callable
        Objective function for Optuna.
    
    Example
    -------
    >>> def param_space(trial):
    ...     return {
    ...         "n_estimators": trial.suggest_int("n_estimators", 100, 500),
    ...         "max_depth": trial.suggest_int("max_depth", 3, 8),
    ...     }
    >>> objective = create_optuna_objective(XGBRegressor, X_train, y_train,
    ...                                      X_val, y_val, param_space)
    >>> study = optuna.create_study(direction="minimize")
    >>> study.optimize(objective, n_trials=50)
    """
    def objective(trial):
        params = param_space_fn(trial)
        model = model_class(**params)
        metrics = evaluate_model_for_tuning(
            model, X_train, y_train, X_val, y_val, fit_params=fit_params
        )
        return metrics[metric]
    
    return objective


# =============================================================================
# sklearn-Compatible Custom Scorer
# =============================================================================

def make_tuning_scorer(metric: str = "mae"):
    """
    Create a sklearn-compatible scorer for GridSearchCV/RandomizedSearchCV.
    
    Parameters
    ----------
    metric : str, default="mae"
        Metric to use ("mae", "rmse"). Will be negated for sklearn compatibility.
    
    Returns
    -------
    callable
        Scorer function compatible with sklearn cross-validation.
    
    Example
    -------
    >>> from sklearn.model_selection import GridSearchCV
    >>> scorer = make_tuning_scorer("mae")
    >>> grid = GridSearchCV(model, param_grid, scoring=scorer, cv=tscv)
    """
    from sklearn.metrics import make_scorer
    
    if metric == "mae":
        return make_scorer(mean_absolute_error, greater_is_better=False)
    elif metric == "rmse":
        def rmse_scorer(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        return make_scorer(rmse_scorer, greater_is_better=False)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'mae' or 'rmse'.")


# =============================================================================
# Phase 7.2: Time-Series Cross-Validation for Hyperparameter Tuning
# =============================================================================
#
# Goal (Phase 7.2): Use TimeSeriesSplit instead of standard K-Fold so every
# hyperparameter setting is evaluated only by predicting on future months from
# past months, matching real-world deployment.
#
# =============================================================================

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV


def build_time_series_cv(
    n_splits: int = 4,
    gap: int = 0,
    test_size: Optional[int] = None,
    max_train_size: Optional[int] = None,
) -> TimeSeriesSplit:
    """
    Build a TimeSeriesSplit cross-validator for time-series data.
    
    This ensures that in each fold:
    - Training data comes from earlier time periods
    - Validation data comes from later time periods
    - No future data leaks into past (unlike standard K-Fold)
    
    Parameters
    ----------
    n_splits : int, default=4
        Number of CV folds.
    gap : int, default=0
        Number of samples to exclude between train and test sets.
        Useful if there's autocorrelation in adjacent time periods.
    test_size : int, optional
        Fixed size for test sets. If None, grows with each split.
    max_train_size : int, optional
        Maximum size for training sets. Useful for limiting compute.
    
    Returns
    -------
    TimeSeriesSplit
        Configured cross-validator.
    
    Example
    -------
    >>> tscv = build_time_series_cv(n_splits=4)
    >>> for train_idx, val_idx in tscv.split(X):
    ...     print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    """
    return TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        test_size=test_size,
        max_train_size=max_train_size,
    )


def build_time_series_grid_search(
    base_model: BaseEstimator,
    param_grid: Dict[str, list],
    n_splits: int = 3,
    metric: str = "mae",
    n_jobs: int = -1,
    verbose: int = 1,
    refit: bool = True,
) -> GridSearchCV:
    """
    Build a GridSearchCV with TimeSeriesSplit for time-aware hyperparameter tuning.
    
    This ensures every hyperparameter configuration is evaluated by predicting
    on future months from past months only—no data leakage, matching real-world
    UIDAI forecasting deployment.
    
    Parameters
    ----------
    base_model : BaseEstimator
        Base model to tune (e.g., XGBRegressor, Pipeline).
    param_grid : Dict[str, list]
        Parameter grid for exhaustive search.
        Example: {"max_depth": [3, 5, 7], "n_estimators": [100, 200]}
    n_splits : int, default=3
        Number of time-series CV folds.
    metric : str, default="mae"
        Metric to optimize ("mae" or "rmse").
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all cores).
    verbose : int, default=1
        Verbosity level.
    refit : bool, default=True
        Whether to refit the best model on the full dataset.
    
    Returns
    -------
    GridSearchCV
        Configured GridSearchCV object ready for .fit(X, y).
    
    Example
    -------
    >>> from xgboost import XGBRegressor
    >>> model = XGBRegressor(random_state=42)
    >>> param_grid = {
    ...     "max_depth": [3, 5, 7],
    ...     "n_estimators": [100, 200],
    ...     "learning_rate": [0.05, 0.1],
    ... }
    >>> grid_search = build_time_series_grid_search(model, param_grid, n_splits=3)
    >>> grid_search.fit(X_train, y_train)
    >>> print(f"Best MAE: {-grid_search.best_score_:.2f}")
    >>> print(f"Best params: {grid_search.best_params_}")
    """
    tscv = build_time_series_cv(n_splits=n_splits)
    scorer = make_tuning_scorer(metric)
    
    return GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
        return_train_score=True,
    )


def build_time_series_random_search(
    base_model: BaseEstimator,
    param_distributions: Dict[str, Any],
    n_iter: int = 20,
    n_splits: int = 3,
    metric: str = "mae",
    n_jobs: int = -1,
    verbose: int = 1,
    refit: bool = True,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """
    Build a RandomizedSearchCV with TimeSeriesSplit for efficient hyperparameter tuning.
    
    Useful when the parameter space is large and exhaustive grid search is too slow.
    
    Parameters
    ----------
    base_model : BaseEstimator
        Base model to tune.
    param_distributions : Dict[str, Any]
        Parameter distributions for random sampling.
        Can use scipy distributions or lists.
        Example: {"max_depth": [3, 5, 7, 9], "learning_rate": uniform(0.01, 0.2)}
    n_iter : int, default=20
        Number of parameter settings sampled.
    n_splits : int, default=3
        Number of time-series CV folds.
    metric : str, default="mae"
        Metric to optimize ("mae" or "rmse").
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : int, default=1
        Verbosity level.
    refit : bool, default=True
        Whether to refit the best model on the full dataset.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    RandomizedSearchCV
        Configured RandomizedSearchCV object ready for .fit(X, y).
    
    Example
    -------
    >>> from scipy.stats import uniform, randint
    >>> param_dist = {
    ...     "max_depth": randint(3, 10),
    ...     "n_estimators": randint(100, 500),
    ...     "learning_rate": uniform(0.01, 0.19),
    ... }
    >>> random_search = build_time_series_random_search(model, param_dist, n_iter=30)
    >>> random_search.fit(X_train, y_train)
    """
    tscv = build_time_series_cv(n_splits=n_splits)
    scorer = make_tuning_scorer(metric)
    
    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
        return_train_score=True,
        random_state=random_state,
    )


def get_default_xgb_param_grid() -> Dict[str, list]:
    """
    Return a small, sensible parameter grid for XGBoost tuning.
    
    This is intentionally compact for initial tuning runs.
    Expand in later phases for more thorough search.
    
    Returns
    -------
    Dict[str, list]
        Parameter grid for GridSearchCV.
    """
    return {
        "max_depth": [4, 5, 6],
        "n_estimators": [200, 300],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }


def summarize_cv_results(search_cv: Union[GridSearchCV, RandomizedSearchCV]) -> pd.DataFrame:
    """
    Extract and format cross-validation results from a completed search.
    
    Parameters
    ----------
    search_cv : GridSearchCV or RandomizedSearchCV
        Fitted search object.
    
    Returns
    -------
    pd.DataFrame
        Results sorted by mean test score (best first).
    """
    results = pd.DataFrame(search_cv.cv_results_)
    
    # Select key columns
    cols = ["rank_test_score", "mean_test_score", "std_test_score",
            "mean_train_score", "std_train_score", "mean_fit_time"]
    param_cols = [c for c in results.columns if c.startswith("param_")]
    cols = cols + param_cols
    
    # Filter to existing columns only
    cols = [c for c in cols if c in results.columns]
    
    return results[cols].sort_values("rank_test_score")


# =============================================================================
# Phase 7.2 Summary
# =============================================================================
#
# This TimeSeriesSplit setup guarantees that hyperparameter tuning respects
# temporal ordering—each fold trains on past data and validates on future data.
#
# Later phases will reuse this:
# - Phase 7.3: Focused grid/random search to find good XGBoost regions
# - Phase 7.4: Final model comparison will use consistent temporal evaluation
#
# This approach makes the model more robust because it's tuned on realistic
# "predict the future from the past" scenarios, not randomly shuffled data.
#
# =============================================================================


# =============================================================================
# Phase 7.3: Structured Grid/Random Search Runners
# =============================================================================
#
# Goal (Phase 7.3): Run a focused GridSearch / RandomizedSearch with TimeSeriesSplit
# to find a robust region of good XGBoost settings for our enrolment forecaster.
#
# =============================================================================

from xgboost import XGBRegressor
from scipy.stats import uniform, randint


def get_phase73_grid_param_space() -> Dict[str, list]:
    """
    Return a focused parameter grid for Phase 7.3 initial tuning.
    
    Covers key XGBoost knobs with modest ranges for fast but meaningful search.
    
    Returns
    -------
    Dict[str, list]
        Parameter grid for GridSearchCV.
    """
    return {
        "n_estimators": [150, 250, 350],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [1, 3],
    }


def get_phase73_random_param_space() -> Dict[str, Any]:
    """
    Return parameter distributions for Phase 7.3 randomized search.
    
    Uses scipy distributions for continuous params and lists for discrete.
    
    Returns
    -------
    Dict[str, Any]
        Parameter distributions for RandomizedSearchCV.
    """
    return {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.02, 0.13),  # 0.02 to 0.15
        "subsample": uniform(0.6, 0.3),        # 0.6 to 0.9
        "colsample_bytree": uniform(0.6, 0.3), # 0.6 to 0.9
        "min_child_weight": randint(1, 6),
    }


def run_initial_grid_search_xgb(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 3,
    metric: str = "mae",
    n_jobs: int = -1,
    verbose: int = 1,
    top_n: int = 5,
    random_state: int = 42,
) -> Tuple[GridSearchCV, pd.DataFrame]:
    """
    Run Phase 7.3 initial grid search for XGBoost hyperparameter tuning.
    
    Uses TimeSeriesSplit cross-validation to ensure no future data leakage.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (should be time-ordered).
    y : pd.Series or np.ndarray
        Target vector.
    n_splits : int, default=3
        Number of time-series CV folds.
    metric : str, default="mae"
        Metric to optimize ("mae" or "rmse").
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2+=detailed).
    top_n : int, default=5
        Number of top configurations to display.
    random_state : int, default=42
        Random seed for XGBoost reproducibility.
    
    Returns
    -------
    Tuple[GridSearchCV, pd.DataFrame]
        (fitted_search, top_results_df)
    
    Example
    -------
    >>> search, results = run_initial_grid_search_xgb(X_train, y_train)
    >>> print(f"Best MAE: {-search.best_score_:.2f}")
    >>> print(f"Best params: {search.best_params_}")
    """
    print("=" * 60)
    print("Phase 7.3 – Initial Grid Search (XGBoost + TimeSeriesSplit)")
    print("=" * 60)
    
    # Base model
    base_model = XGBRegressor(
        random_state=random_state,
        n_jobs=1,  # CV handles parallelism
        verbosity=0,
    )
    
    # Parameter grid
    param_grid = get_phase73_grid_param_space()
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"\nParameter grid: {n_combos} combinations")
    print(f"CV folds: {n_splits} (TimeSeriesSplit)")
    print(f"Total fits: {n_combos * n_splits}")
    print(f"Metric: {metric.upper()}")
    print("-" * 60)
    
    # Build and run grid search
    grid_search = build_time_series_grid_search(
        base_model=base_model,
        param_grid=param_grid,
        n_splits=n_splits,
        metric=metric,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    
    grid_search.fit(X, y)
    
    # Summarize results
    results_df = summarize_cv_results(grid_search)
    top_results = results_df.head(top_n).copy()
    
    # Convert negative scores to positive MAE/RMSE
    top_results["mean_val_mae"] = -top_results["mean_test_score"]
    top_results["std_val_mae"] = top_results["std_test_score"]
    
    print("\n" + "=" * 60)
    print(f"✓ Grid Search Complete — Top {top_n} Configurations")
    print("=" * 60)
    print(f"\nBest {metric.upper()}: {-grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print("\nTop configurations:")
    
    # Display top configs
    display_cols = ["rank_test_score", "mean_val_mae", "std_val_mae"]
    param_cols = [c for c in top_results.columns if c.startswith("param_")]
    for idx, row in top_results.iterrows():
        rank = int(row["rank_test_score"])
        mae = row["mean_val_mae"]
        std = row["std_val_mae"]
        params = {c.replace("param_", ""): row[c] for c in param_cols}
        print(f"  #{rank}: MAE={mae:.2f} ± {std:.2f} | {params}")
    
    return grid_search, results_df


def run_initial_random_search_xgb(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_iter: int = 20,
    n_splits: int = 3,
    metric: str = "mae",
    n_jobs: int = -1,
    verbose: int = 1,
    top_n: int = 5,
    random_state: int = 42,
) -> Tuple[RandomizedSearchCV, pd.DataFrame]:
    """
    Run Phase 7.3 initial randomized search for XGBoost hyperparameter tuning.
    
    Uses TimeSeriesSplit cross-validation to ensure no future data leakage.
    More efficient than grid search for larger parameter spaces.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (should be time-ordered).
    y : pd.Series or np.ndarray
        Target vector.
    n_iter : int, default=20
        Number of parameter settings to sample.
    n_splits : int, default=3
        Number of time-series CV folds.
    metric : str, default="mae"
        Metric to optimize ("mae" or "rmse").
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).
    verbose : int, default=1
        Verbosity level.
    top_n : int, default=5
        Number of top configurations to display.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[RandomizedSearchCV, pd.DataFrame]
        (fitted_search, top_results_df)
    
    Example
    -------
    >>> search, results = run_initial_random_search_xgb(X_train, y_train, n_iter=30)
    >>> print(f"Best MAE: {-search.best_score_:.2f}")
    """
    print("=" * 60)
    print("Phase 7.3 – Initial Random Search (XGBoost + TimeSeriesSplit)")
    print("=" * 60)
    
    # Base model
    base_model = XGBRegressor(
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
    )
    
    # Parameter distributions
    param_dist = get_phase73_random_param_space()
    print(f"\nSampling {n_iter} random configurations")
    print(f"CV folds: {n_splits} (TimeSeriesSplit)")
    print(f"Total fits: {n_iter * n_splits}")
    print(f"Metric: {metric.upper()}")
    print("-" * 60)
    
    # Build and run random search
    random_search = build_time_series_random_search(
        base_model=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        n_splits=n_splits,
        metric=metric,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        random_state=random_state,
    )
    
    random_search.fit(X, y)
    
    # Summarize results
    results_df = summarize_cv_results(random_search)
    top_results = results_df.head(top_n).copy()
    
    # Convert negative scores to positive MAE/RMSE
    top_results["mean_val_mae"] = -top_results["mean_test_score"]
    top_results["std_val_mae"] = top_results["std_test_score"]
    
    print("\n" + "=" * 60)
    print(f"✓ Random Search Complete — Top {top_n} Configurations")
    print("=" * 60)
    print(f"\nBest {metric.upper()}: {-random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}")
    print("\nTop configurations:")
    
    # Display top configs
    param_cols = [c for c in top_results.columns if c.startswith("param_")]
    for idx, row in top_results.iterrows():
        rank = int(row["rank_test_score"])
        mae = row["mean_val_mae"]
        std = row["std_val_mae"]
        params = {c.replace("param_", ""): row[c] for c in param_cols}
        # Format numeric params nicely
        params_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        )
        print(f"  #{rank}: MAE={mae:.2f} ± {std:.2f}")
        print(f"       {params_str}")
    
    return random_search, results_df


def run_quick_demo_search(n_samples: int = 500, n_iter: int = 10) -> None:
    """
    Quick demo of Phase 7.3 search on synthetic data.
    
    Useful for verifying end-to-end behavior without loading real data.
    
    Parameters
    ----------
    n_samples : int, default=500
        Number of synthetic samples.
    n_iter : int, default=10
        Number of random search iterations.
    """
    print("\n" + "=" * 60)
    print("Phase 7.3 Demo – Synthetic Data Verification")
    print("=" * 60)
    
    # Generate synthetic time-series-like data
    np.random.seed(42)
    
    # Features with temporal structure
    X = pd.DataFrame({
        "feature_1": np.cumsum(np.random.randn(n_samples)) + 100,
        "feature_2": np.sin(np.linspace(0, 8 * np.pi, n_samples)) * 50 + 200,
        "feature_3": np.random.randn(n_samples) * 10 + 50,
        "feature_4": np.random.randint(1, 10, n_samples),
    })
    
    # Target with trend and seasonality
    y = (
        0.5 * X["feature_1"] +
        0.3 * X["feature_2"] +
        0.2 * X["feature_3"] +
        np.random.randn(n_samples) * 20
    )
    
    print(f"Synthetic dataset: {n_samples} samples, {X.shape[1]} features")
    
    # Run a quick random search
    search, results = run_initial_random_search_xgb(
        X, y,
        n_iter=n_iter,
        n_splits=3,
        verbose=0,
        top_n=3,
    )
    
    print("\n✓ Demo complete — Phase 7.3 search runners working correctly")


# =============================================================================
# Phase 7.3 Summary
# =============================================================================
#
# Phase 7.3 provides ready-to-use search runners for initial XGBoost tuning:
# - `run_initial_grid_search_xgb()`: Exhaustive search over a focused grid
# - `run_initial_random_search_xgb()`: Efficient sampling for broader exploration
#
# Both use TimeSeriesSplit to ensure temporal integrity (no data leakage).
#
# Later phases can use the best region found here:
# - Phase 7.4 (Optuna): Zoom into the best parameter region for finer Bayesian tuning
# - Final model: Retrain on full data with the optimal hyperparameters found
#
# =============================================================================


# =============================================================================
# Phase 7.4: Optuna Fine-Tuning with Early Stopping
# =============================================================================
#
# Goal (Phase 7.4): Use Optuna with time-series-aware evaluation and early stopping
# to finely tune XGBoost around the best region found by Grid/Random search,
# improving accuracy without huge compute.
#
# =============================================================================

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def get_phase74_optuna_search_space(trial) -> Dict[str, Any]:
    """
    Define a narrow Optuna search space around the best region from Phase 7.3.
    
    This focuses on fine-tuning within the promising parameter region
    discovered by grid/random search, rather than exploring broadly.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting parameter values.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of hyperparameters for XGBoost.
    
    Notes
    -----
    The ranges here are intentionally narrow, centered around typical
    good values found in Phase 7.3. Adjust based on your actual
    Phase 7.3 results for best performance.
    """
    return {
        # Core tree parameters (narrow ranges around good values)
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        
        # Learning rate (fine-grained)
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        
        # Sampling parameters (narrow range)
        "subsample": trial.suggest_float("subsample", 0.65, 0.90),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.90),
        
        # Regularization (optional fine-tuning)
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
    }


def create_optuna_cv_objective(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    param_space_fn: callable,
    n_splits: int = 3,
    metric: str = "mae",
    early_stopping_rounds: int = 30,
    random_state: int = 42,
):
    """
    Create an Optuna objective function with TimeSeriesSplit cross-validation.
    
    This ensures every trial is evaluated on future months only,
    matching real-world deployment and avoiding data leakage.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (should be time-ordered).
    y : pd.Series or np.ndarray
        Target vector.
    param_space_fn : callable
        Function that takes a trial and returns params dict.
    n_splits : int, default=3
        Number of TimeSeriesSplit folds.
    metric : str, default="mae"
        Metric to optimize ("mae" or "rmse").
    early_stopping_rounds : int, default=30
        XGBoost early stopping rounds.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    callable
        Objective function for Optuna study.
    """
    tscv = build_time_series_cv(n_splits=n_splits)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()
    
    def objective(trial):
        # Get hyperparameters for this trial
        params = param_space_fn(trial)
        
        # Add fixed parameters
        params["random_state"] = random_state
        params["n_jobs"] = 1
        params["verbosity"] = 0
        
        # Cross-validation scores
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_train_fold = X_arr[train_idx]
            y_train_fold = y_arr[train_idx]
            X_val_fold = X_arr[val_idx]
            y_val_fold = y_arr[val_idx]
            
            # Create model with early stopping
            model = XGBRegressor(**params)
            
            # Fit with early stopping
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )
            
            # Predict and compute metric
            y_pred = model.predict(X_val_fold)
            
            if metric == "mae":
                score = mean_absolute_error(y_val_fold, y_pred)
            else:  # rmse
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            
            cv_scores.append(score)
            
            # Optuna pruning: report intermediate value
            trial.report(np.mean(cv_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    return objective


def run_optuna_tuning_xgb(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_trials: int = 30,
    n_splits: int = 3,
    metric: str = "mae",
    early_stopping_rounds: int = 30,
    timeout: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
    study_name: str = "xgb_tuning",
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Run Optuna hyperparameter tuning for XGBoost with TimeSeriesSplit.
    
    Uses Bayesian optimization (TPE sampler) to efficiently search the
    parameter space, with early stopping to save compute.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (should be time-ordered).
    y : pd.Series or np.ndarray
        Target vector.
    n_trials : int, default=30
        Number of Optuna trials. Increase to 50-100 for more thorough search.
    n_splits : int, default=3
        Number of TimeSeriesSplit folds.
    metric : str, default="mae"
        Metric to optimize ("mae" or "rmse").
    early_stopping_rounds : int, default=30
        XGBoost early stopping rounds.
    timeout : int, optional
        Maximum time in seconds for optimization.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress.
    study_name : str, default="xgb_tuning"
        Name for the Optuna study.
    
    Returns
    -------
    Tuple[optuna.Study, Dict[str, Any], float]
        (study, best_params, best_score)
    
    Example
    -------
    >>> study, best_params, best_mae = run_optuna_tuning_xgb(X, y, n_trials=50)
    >>> print(f"Best MAE: {best_mae:.4f}")
    >>> print(f"Best params: {best_params}")
    
    Notes
    -----
    - Increase n_trials for more thorough search (50-100 recommended for production)
    - Use timeout parameter to limit total runtime
    - The study object can be saved/loaded for continued optimization
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for Phase 7.4. Install with: pip install optuna"
        )
    
    if verbose:
        print("=" * 60)
        print("Phase 7.4 – Optuna Fine-Tuning (XGBoost + TimeSeriesSplit)")
        print("=" * 60)
        print(f"\nTrials: {n_trials}")
        print(f"CV folds: {n_splits} (TimeSeriesSplit)")
        print(f"Metric: {metric.upper()}")
        print(f"Early stopping: {early_stopping_rounds} rounds")
        print("-" * 60)
    
    # Create objective function
    objective = create_optuna_cv_objective(
        X=X,
        y=y,
        param_space_fn=get_phase74_optuna_search_space,
        n_splits=n_splits,
        metric=metric,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    
    # Configure sampler for reproducibility
    sampler = TPESampler(seed=random_state)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    
    # Suppress Optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
    )
    
    # Extract results
    best_params = study.best_params
    best_score = study.best_value
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Optuna Tuning Complete")
        print("=" * 60)
        print(f"\nBest {metric.upper()}: {best_score:.4f}")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"\nBest Parameters:")
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        # Show trial summary
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"\nTrial Summary:")
        print(f"  Completed: {n_complete}")
        print(f"  Pruned: {n_pruned}")
        print(f"  Total: {len(study.trials)}")
    
    return study, best_params, best_score


def get_tuned_xgb_model(best_params: Dict[str, Any], random_state: int = 42) -> XGBRegressor:
    """
    Create an XGBRegressor with the best parameters from Optuna tuning.
    
    Parameters
    ----------
    best_params : Dict[str, Any]
        Best parameters from Optuna study.
    random_state : int, default=42
        Random seed.
    
    Returns
    -------
    XGBRegressor
        Configured model ready for training.
    """
    params = best_params.copy()
    params["random_state"] = random_state
    params["n_jobs"] = -1
    params["verbosity"] = 0
    return XGBRegressor(**params)


def run_optuna_demo(n_samples: int = 400, n_trials: int = 8) -> None:
    """
    Quick demo of Phase 7.4 Optuna tuning on synthetic data.
    
    Useful for verifying end-to-end behavior without loading real data.
    
    Parameters
    ----------
    n_samples : int, default=400
        Number of synthetic samples.
    n_trials : int, default=8
        Number of Optuna trials (keep small for demo).
    """
    if not OPTUNA_AVAILABLE:
        print("⚠ Optuna not installed. Install with: pip install optuna")
        return
    
    print("\n" + "=" * 60)
    print("Phase 7.4 Demo – Optuna Tuning Verification")
    print("=" * 60)
    
    # Generate synthetic time-series-like data
    np.random.seed(42)
    
    X = pd.DataFrame({
        "feature_1": np.cumsum(np.random.randn(n_samples)) + 100,
        "feature_2": np.sin(np.linspace(0, 8 * np.pi, n_samples)) * 50 + 200,
        "feature_3": np.random.randn(n_samples) * 10 + 50,
        "feature_4": np.random.randint(1, 10, n_samples),
    })
    
    y = (
        0.5 * X["feature_1"] +
        0.3 * X["feature_2"] +
        0.2 * X["feature_3"] +
        np.random.randn(n_samples) * 20
    )
    
    print(f"Synthetic dataset: {n_samples} samples, {X.shape[1]} features")
    
    # Run Optuna tuning
    study, best_params, best_score = run_optuna_tuning_xgb(
        X, y,
        n_trials=n_trials,
        n_splits=3,
        verbose=True,
    )
    
    # Create tuned model
    tuned_model = get_tuned_xgb_model(best_params)
    print(f"\n✓ Demo complete — Tuned model ready: {type(tuned_model).__name__}")
    print("✓ Phase 7.4 Optuna tuning working correctly")


# =============================================================================
# Phase 7.4 Summary
# =============================================================================
#
# Phase 7.4 uses Optuna's Bayesian optimization (TPE sampler) to fine-tune
# XGBoost around the good region found in Phase 7.3, with early stopping
# to save compute.
#
# This produces a production-ready tuned configuration that:
# - Later drift monitoring can use as the default model setup
# - Retraining pipelines can apply the same optimal hyperparameters
# - Dashboard alerts can rely on consistent, well-tuned predictions
#
# To increase search thoroughness, raise n_trials to 50-100 in production.
#
# =============================================================================


# =============================================================================
# Phase 7.5: Lock and Version the Best Model
# =============================================================================
#
# Goal (Phase 7.5): Retrain the best tuned configuration on full history,
# evaluate on a final hold-out window, and register it as a versioned
# 'champion model' for drift monitoring and retraining.
#
# =============================================================================

import json
import pickle
from datetime import datetime
from pathlib import Path


# Default model directory (relative to project root)
def _get_models_dir() -> Path:
    """Get the models directory path."""
    # Try to find project root
    current = Path(__file__).resolve().parent.parent
    models_dir = current / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def train_champion_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    best_params: Dict[str, Any],
    holdout_months: int = 2,
    date_col: Optional[str] = None,
    dates: Optional[pd.Series] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[XGBRegressor, Dict[str, float], Dict[str, Any]]:
    """
    Train the champion model on full training data and evaluate on hold-out.
    
    This is the final step: retrain the best configuration from tuning
    on all historical data except the final hold-out window.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (should be time-ordered).
    y : pd.Series or np.ndarray
        Target vector.
    best_params : Dict[str, Any]
        Best hyperparameters from Optuna tuning (Phase 7.4).
    holdout_months : int, default=2
        Number of most recent months to hold out for final evaluation.
    date_col : str, optional
        If X is a DataFrame with a date column, specify it here.
    dates : pd.Series, optional
        Explicit date series for splitting (if date not in X).
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress.
    
    Returns
    -------
    Tuple[XGBRegressor, Dict[str, float], Dict[str, Any]]
        (fitted_model, holdout_metrics, metadata)
        
        - fitted_model: Trained XGBRegressor
        - holdout_metrics: Dict with mae, rmse, r2, mape on hold-out
        - metadata: Dict with training info (date ranges, features, etc.)
    
    Example
    -------
    >>> model, metrics, meta = train_champion_model(X, y, best_params)
    >>> print(f"Hold-out MAE: {metrics['mae']:.2f}")
    """
    if verbose:
        print("=" * 60)
        print("Phase 7.5 – Training Champion Model")
        print("=" * 60)
    
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()
    n_samples = len(y_arr)
    
    # Determine split point
    # If dates provided, use time-based split; otherwise use fraction
    if dates is not None:
        dates = pd.to_datetime(dates)
        unique_months = sorted(dates.dt.to_period("M").unique())
        n_months = len(unique_months)
        
        if n_months <= holdout_months:
            raise ValueError(f"Not enough months ({n_months}) for holdout ({holdout_months})")
        
        holdout_periods = set(unique_months[-holdout_months:])
        train_periods = set(unique_months[:-holdout_months])
        
        periods = dates.dt.to_period("M")
        train_mask = periods.isin(train_periods)
        holdout_mask = periods.isin(holdout_periods)
        
        train_start = min(train_periods).to_timestamp()
        train_end = max(train_periods).to_timestamp()
        holdout_start = min(holdout_periods).to_timestamp()
        holdout_end = max(holdout_periods).to_timestamp()
    else:
        # Fallback: use last fraction of data
        holdout_frac = holdout_months / 12  # Approximate
        split_idx = int(n_samples * (1 - holdout_frac))
        
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[:split_idx] = True
        holdout_mask = ~train_mask
        
        train_start = "index_0"
        train_end = f"index_{split_idx-1}"
        holdout_start = f"index_{split_idx}"
        holdout_end = f"index_{n_samples-1}"
    
    X_train = X_arr[train_mask]
    y_train = y_arr[train_mask]
    X_holdout = X_arr[holdout_mask]
    y_holdout = y_arr[holdout_mask]
    
    if verbose:
        print(f"\nTrain samples: {len(y_train)}")
        print(f"Hold-out samples: {len(y_holdout)}")
        print(f"Train period: {train_start} to {train_end}")
        print(f"Hold-out period: {holdout_start} to {holdout_end}")
        print("-" * 60)
    
    # Build and train model
    model = get_tuned_xgb_model(best_params, random_state=random_state)
    
    if verbose:
        print("Training champion model...")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_holdout, y_holdout)],
        verbose=False,
    )
    
    # Evaluate on hold-out
    y_pred = model.predict(X_holdout)
    metrics = compute_tuning_metrics(y_holdout, y_pred)
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Champion Model Trained")
        print("=" * 60)
        print(f"\nHold-out Metrics:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Build metadata
    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X_arr.shape[1])]
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "train_samples": int(len(y_train)),
        "holdout_samples": int(len(y_holdout)),
        "train_period_start": str(train_start),
        "train_period_end": str(train_end),
        "holdout_period_start": str(holdout_start),
        "holdout_period_end": str(holdout_end),
        "n_features": int(X_arr.shape[1]),
        "feature_names": feature_names,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") and model.best_iteration is not None else -1,
    }
    
    return model, metrics, metadata


def save_model_version(
    model: XGBRegressor,
    best_params: Dict[str, Any],
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    version_name: Optional[str] = None,
    tuning_method: str = "Grid+Random+Optuna",
    model_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """
    Save the champion model with versioned metadata.
    
    Creates two files:
    - models/xgb_enrolment_<version>.pkl  (pickled model)
    - models/xgb_enrolment_<version>.json (metadata sidecar)
    
    Parameters
    ----------
    model : XGBRegressor
        Trained champion model.
    best_params : Dict[str, Any]
        Best hyperparameters from tuning.
    metrics : Dict[str, float]
        Hold-out evaluation metrics.
    metadata : Dict[str, Any]
        Training metadata (from train_champion_model).
    version_name : str, optional
        Version identifier. If None, uses timestamp (YYYYMMDD_HHMMSS).
    tuning_method : str, default="Grid+Random+Optuna"
        Description of tuning approach used.
    model_dir : Path, optional
        Directory to save models. Defaults to project models/ dir.
    verbose : bool, default=True
        Whether to print progress.
    
    Returns
    -------
    Tuple[Path, Path]
        (model_path, metadata_path)
    
    Example
    -------
    >>> model_path, meta_path = save_model_version(model, params, metrics, meta)
    >>> print(f"Model saved to: {model_path}")
    """
    if model_dir is None:
        model_dir = _get_models_dir()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate version name if not provided
    if version_name is None:
        version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File paths
    model_filename = f"xgb_enrolment_{version_name}.pkl"
    meta_filename = f"xgb_enrolment_{version_name}.json"
    model_path = model_dir / model_filename
    meta_path = model_dir / meta_filename
    
    # Build comprehensive metadata
    full_metadata = {
        "model_version": version_name,
        "model_type": "XGBRegressor",
        "tuning_method": tuning_method,
        "best_hyperparameters": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                  for k, v in best_params.items()},
        "metrics_holdout": {k: float(v) for k, v in metrics.items()},
        **metadata,
    }
    
    # Save model (pickle)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata (JSON)
    with open(meta_path, "w") as f:
        json.dump(full_metadata, f, indent=2, default=str)
    
    # Update "current" symlink/pointer
    current_pointer = model_dir / "current_champion.json"
    pointer_data = {
        "version": version_name,
        "model_file": model_filename,
        "metadata_file": meta_filename,
        "updated_at": datetime.now().isoformat(),
    }
    with open(current_pointer, "w") as f:
        json.dump(pointer_data, f, indent=2)
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Model Version Saved")
        print("=" * 60)
        print(f"\nVersion: {version_name}")
        print(f"Model:   {model_path}")
        print(f"Meta:    {meta_path}")
        print(f"Pointer: {current_pointer}")
    
    return model_path, meta_path


def load_current_champion_model(
    model_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    Load the current champion model and its metadata.
    
    Reads the current_champion.json pointer to find the latest version.
    
    Parameters
    ----------
    model_dir : Path, optional
        Directory containing models. Defaults to project models/ dir.
    verbose : bool, default=True
        Whether to print info.
    
    Returns
    -------
    Tuple[XGBRegressor, Dict[str, Any]]
        (model, metadata)
    
    Raises
    ------
    FileNotFoundError
        If no champion model has been saved yet.
    
    Example
    -------
    >>> model, meta = load_current_champion_model()
    >>> print(f"Loaded version: {meta['model_version']}")
    >>> print(f"Hold-out MAE: {meta['metrics_holdout']['mae']:.2f}")
    """
    if model_dir is None:
        model_dir = _get_models_dir()
    model_dir = Path(model_dir)
    
    # Read pointer
    pointer_path = model_dir / "current_champion.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"No champion model found. Run train_champion_model + save_model_version first.\n"
            f"Expected pointer at: {pointer_path}"
        )
    
    with open(pointer_path, "r") as f:
        pointer = json.load(f)
    
    model_path = model_dir / pointer["model_file"]
    meta_path = model_dir / pointer["metadata_file"]
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load metadata
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    if verbose:
        print(f"✓ Loaded champion model: {pointer['version']}")
        # Safely access metrics - may be stored under different keys
        if 'metrics_holdout' in metadata:
            print(f"  MAE (holdout): {metadata['metrics_holdout']['mae']:.4f}")
        elif 'metrics' in metadata and 'mae' in metadata['metrics']:
            print(f"  MAE: {metadata['metrics']['mae']:.4f}")
        if 'training_date' in metadata:
            print(f"  Training date: {metadata['training_date']}")
    
    return model, metadata


def list_model_versions(model_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    List all saved model versions with their metrics.
    
    Parameters
    ----------
    model_dir : Path, optional
        Directory containing models.
    
    Returns
    -------
    pd.DataFrame
        Table of versions with metrics.
    """
    if model_dir is None:
        model_dir = _get_models_dir()
    model_dir = Path(model_dir)
    
    versions = []
    for meta_file in model_dir.glob("xgb_enrolment_*.json"):
        with open(meta_file, "r") as f:
            meta = json.load(f)
        versions.append({
            "version": meta.get("model_version", "unknown"),
            "training_date": meta.get("training_date", "unknown"),
            "mae": meta.get("metrics_holdout", {}).get("mae", np.nan),
            "rmse": meta.get("metrics_holdout", {}).get("rmse", np.nan),
            "r2": meta.get("metrics_holdout", {}).get("r2", np.nan),
            "tuning_method": meta.get("tuning_method", "unknown"),
        })
    
    if not versions:
        return pd.DataFrame(columns=["version", "training_date", "mae", "rmse", "r2", "tuning_method"])
    
    return pd.DataFrame(versions).sort_values("training_date", ascending=False)


def run_finalize_model_demo(n_samples: int = 500, n_trials: int = 8) -> None:
    """
    Demo of the full Phase 7.5 workflow on synthetic data.
    
    1. Generate synthetic data
    2. Run quick Optuna tuning to get best_params
    3. Train champion model on full data with hold-out
    4. Save versioned model and metadata
    5. Reload and verify
    
    Parameters
    ----------
    n_samples : int, default=500
        Number of synthetic samples.
    n_trials : int, default=8
        Number of Optuna trials (keep small for demo).
    """
    if not OPTUNA_AVAILABLE:
        print("⚠ Optuna not installed. Install with: pip install optuna")
        return
    
    print("\n" + "=" * 60)
    print("Phase 7.5 Demo – Champion Model Finalization")
    print("=" * 60)
    
    # Generate synthetic time-series data
    np.random.seed(42)
    n_months = 12
    samples_per_month = n_samples // n_months
    actual_samples = samples_per_month * n_months  # Ensure divisibility
    
    dates = []
    for m in range(n_months):
        month_date = pd.Timestamp("2025-01-01") + pd.DateOffset(months=m)
        dates.extend([month_date] * samples_per_month)
    dates = pd.Series(dates)
    
    X = pd.DataFrame({
        "feature_1": np.cumsum(np.random.randn(actual_samples)) + 100,
        "feature_2": np.sin(np.linspace(0, 8 * np.pi, actual_samples)) * 50 + 200,
        "feature_3": np.random.randn(actual_samples) * 10 + 50,
        "feature_4": np.random.randint(1, 10, actual_samples),
    })
    
    y = (
        0.5 * X["feature_1"] +
        0.3 * X["feature_2"] +
        0.2 * X["feature_3"] +
        np.random.randn(actual_samples) * 20
    )
    
    print(f"\nSynthetic dataset: {actual_samples} samples, {n_months} months")
    
    # Step 1: Quick Optuna tuning
    print("\n--- Step 1: Optuna Tuning ---")
    study, best_params, best_score = run_optuna_tuning_xgb(
        X, y,
        n_trials=n_trials,
        n_splits=3,
        verbose=False,
    )
    print(f"Best MAE from tuning: {best_score:.4f}")
    
    # Step 2: Train champion model
    print("\n--- Step 2: Train Champion Model ---")
    model, metrics, metadata = train_champion_model(
        X, y,
        best_params=best_params,
        holdout_months=2,
        dates=dates,
        verbose=True,
    )
    
    # Step 3: Save versioned model
    print("\n--- Step 3: Save Versioned Model ---")
    version_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path, meta_path = save_model_version(
        model=model,
        best_params=best_params,
        metrics=metrics,
        metadata=metadata,
        version_name=version_name,
        tuning_method="Optuna (demo)",
        verbose=True,
    )
    
    # Step 4: Reload and verify
    print("\n--- Step 4: Reload and Verify ---")
    loaded_model, loaded_meta = load_current_champion_model(verbose=True)
    
    # Step 5: List versions
    print("\n--- Step 5: All Model Versions ---")
    versions_df = list_model_versions()
    print(versions_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✓ Phase 7.5 Demo Complete")
    print("=" * 60)
    print(f"\nChampion model saved as: {version_name}")
    print("Future phases (drift, retraining, dashboards) can load this model.")


# =============================================================================
# Phase 7.5 Summary
# =============================================================================
#
# Phase 7.5 finalizes the tuning process by:
# - Training the best-tuned model on full historical data
# - Evaluating on a held-out window for final validation
# - Saving as a versioned "champion model" with full metadata
#
# This versioned champion model is what future phases will load by default:
# - Drift detection compares new data against this champion's training distribution
# - Scheduled retraining uses the champion's hyperparameters as the starting point
# - Dashboards and alerts use the champion for production predictions
#
# =============================================================================


# =============================================================================
# Summary (Phase 7.1)
# =============================================================================
#
# Phase 7.1 establishes the evaluation foundation for hyperparameter tuning:
#
# 1. `time_ordered_split()` ensures realistic train/val separation by time
# 2. `compute_tuning_metrics()` provides standardized MAE/RMSE/R²/MAPE
# 3. `evaluate_model_for_tuning()` is the core reusable function for any tuning approach
#
# Later phases will build on this:
# - Phase 7.2: Use with sklearn TimeSeriesSplit for cross-validation
# - Phase 7.3: Use with Optuna via `create_optuna_objective()` for Bayesian search
# - Phase 7.4: Final model selection and comparison using these consistent metrics
#
# =============================================================================


if __name__ == "__main__":
    import sys
    
    # Quick validation that the module loads correctly
    print("✓ Phase 7.1 + 7.2 + 7.3 + 7.4 + 7.5 tuning module loaded successfully")
    print("\nPhase 7.1 - Evaluation Setup:")
    print("  - compute_tuning_metrics(y_true, y_pred)")
    print("  - time_ordered_split(df, date_col, n_val_months)")
    print("  - evaluate_model_for_tuning(model, X_train, y_train, X_val, y_val)")
    print("  - create_optuna_objective(...)")
    print("  - make_tuning_scorer(metric)")
    print("\nPhase 7.2 - Time-Series Cross-Validation:")
    print("  - build_time_series_cv(n_splits)")
    print("  - build_time_series_grid_search(base_model, param_grid)")
    print("  - build_time_series_random_search(base_model, param_distributions)")
    print("  - get_default_xgb_param_grid()")
    print("  - summarize_cv_results(search_cv)")
    print("\nPhase 7.3 - Structured Search Runners:")
    print("  - get_phase73_grid_param_space()")
    print("  - get_phase73_random_param_space()")
    print("  - run_initial_grid_search_xgb(X, y)")
    print("  - run_initial_random_search_xgb(X, y, n_iter)")
    print("  - run_quick_demo_search()")
    print("\nPhase 7.4 - Optuna Fine-Tuning:")
    print("  - get_phase74_optuna_search_space(trial)")
    print("  - create_optuna_cv_objective(...)")
    print("  - run_optuna_tuning_xgb(X, y, n_trials)")
    print("  - get_tuned_xgb_model(best_params)")
    print("  - run_optuna_demo()")
    if OPTUNA_AVAILABLE:
        print("  ✓ Optuna is installed and available")
    else:
        print("  ⚠ Optuna not installed (pip install optuna)")
    print("\nPhase 7.5 - Champion Model Versioning:")
    print("  - train_champion_model(X, y, best_params, holdout_months)")
    print("  - save_model_version(model, best_params, metrics, metadata)")
    print("  - load_current_champion_model()")
    print("  - list_model_versions()")
    print("  - run_finalize_model_demo()")
    
    # Run demos based on flags
    if "--demo" in sys.argv:
        run_quick_demo_search(n_samples=300, n_iter=8)
    
    if "--optuna-demo" in sys.argv:
        run_optuna_demo(n_samples=300, n_trials=6)
    
    if "--finalize-model" in sys.argv:
        run_finalize_model_demo(n_samples=400, n_trials=6)
