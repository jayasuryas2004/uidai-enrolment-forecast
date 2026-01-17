# Phase-4 Model Documentation

## Overview

Phase-4 is the final modeling phase for the UIDAI district-level demand forecasting hackathon. It uses XGBoost with enhanced calendar and group-aggregate features.

---

## ⚠️ FROZEN FINAL MODELS ⚠️

**Both Phase-4 v2 and v3 models are frozen as FINAL models.**

All further experiments **MUST** save under:
```
artifacts/experiments/phase4_v2/<exp_name>/   # for v2 experiments
artifacts/experiments/phase4_v3/<exp_name>/   # for v3 experiments
```
to avoid overwriting the final artifacts.

---

## Phase-4 v3: Official Baseline Model (Frozen) 🏆

The v3 baseline is the **production-style, leakage-safe model** recommended for UIDAI forecasting.

### Why v3 Baseline?

| Aspect | v2 Single Split | v3 Baseline (Time-Series CV) |
|--------|-----------------|------------------------------|
| Validation | Single train/val/test split | Expanding-window CV (4 folds, 1-month gap) |
| Leakage Safety | ⚠️ Potential leakage | ✅ Leakage-safe |
| R² (Val/CV) | 0.9998 | 0.95 ± 0.05 |
| MAE (Val/CV) | 10.59 | ~124 ± 128 |
| Robustness | Single point estimate | Stable across time periods |

The v3 baseline has **realistic, generalizable metrics** validated with proper time-series cross-validation.

### v3 Baseline Features

- **Lag features:** [1, 2, 3, 6] months (leakage-safe with shift)
- **Rolling features:** [3, 6] month windows (mean, std, min, max)
- **Holiday dummies:** Festival peak, exam months, monsoon, harvest
- **Policy dummies:** Phase 1/2/3, time trend, FY start/end
- **Calendar features:** Month sin/cos, year, quarter

### v3 Baseline Artifacts (DO NOT OVERWRITE)

| File | Description |
|------|-------------|
| `artifacts/xgb_phase4_v3_baseline.pkl` | Final XGBoost model |
| `artifacts/xgb_phase4_v3_baseline_encoders.pkl` | LabelEncoders for categorical columns |
| `artifacts/xgb_phase4_v3_baseline_metrics.json` | CV metrics and configuration |
| `artifacts/xgb_phase4_v3_baseline_params.json` | Hyperparameters |

### v3 Baseline CV Metrics

| Fold | Train Rows | Val Month | R² | MAE |
|------|------------|-----------|------|-----|
| 1 | 119 | 2025-09 | ~0.86 | ~343 |
| 2 | 218 | 2025-10 | ~0.96 | ~84 |
| 3 | 498 | 2025-11 | ~0.99 | ~49 |
| 4 | 1,460 | 2025-12 | ~0.99 | ~21 |
| **Mean** | - | - | **~0.95** | **~124** |

### v3 Baseline Hyperparameters

```python
{
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "n_estimators": 500
}
```

### Using v3 Baseline for Inference

```python
from src.phase4_v3_inference import Phase4V3Model

# Load the frozen v3 baseline
model = Phase4V3Model()
model.load()

# Make predictions
predictions = model.predict(df_new)

# Check CV metrics
print(model.get_cv_metrics())
```

Or use the convenience function:

```python
from src.phase4_v3_inference import predict_with_v3

predictions = predict_with_v3(df_new)
```

---

## Phase-4 v2: Original Tuned Model (Frozen)

## Phase-4 v2: Original Tuned Model (Frozen)

The v2 model was tuned with RandomizedSearchCV on a single train/val/test split.

### v2 Final Artifacts (DO NOT OVERWRITE)

| File | Description |
|------|-------------|
| `artifacts/xgb_phase4_v2_tuned_best.pkl` | Final XGBoost model |
| `artifacts/xgb_phase4_v2_tuned_best.encoders.pkl` | LabelEncoders for categorical columns |
| `artifacts/xgb_phase4_v2_tuned_best_metrics.json` | Train/Val/Test metrics |
| `artifacts/xgb_phase4_v2_tuned_best_params.json` | Best hyperparameters |
| `artifacts/xgb_phase4_v2_random_search_results.csv` | All random search trials |

### v2 Final Model Metrics

| Split | R² | MAE | RMSE |
|-------|------|-------|-------|
| Train | 0.9999 | 15.06 | 20.00 |
| Val | 0.9998 | 10.59 | 15.30 |
| **Test** | **0.9980** | **29.71** | 54.87 |

### v2 Best Hyperparameters

```python
{
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1
}
```

---

## Running New Experiments

### Using `--exp-name` Flag

To run a new experiment without overwriting the final model:

```bash
# Run experiment with a unique name
python -m src.run_phase4_v2_with_enhanced_features \
    --n-trials 20 \
    --exp-name exp1_higher_depth

# Artifacts will be saved to:
# artifacts/experiments/phase4_v2/exp1_higher_depth/
#   ├── model.pkl
#   ├── encoders.pkl
#   ├── metrics.json
#   ├── params.json
#   └── random_search_results.csv
```

### Using the Model Registry

```python
from src.phase4_model_registry import (
    PHASE4_V2_FINAL,
    get_experiment_paths,
    check_not_overwriting_final,
)

# Get paths for a new experiment
paths = get_experiment_paths("exp2_drop_festival")
paths["dir"].mkdir(parents=True, exist_ok=True)

# Save artifacts safely
joblib.dump(model, paths["model"])
with open(paths["metrics"], "w") as f:
    json.dump(metrics, f)
```

### Loading the Final Model

```python
import joblib
from src.phase4_model_registry import PHASE4_V2_FINAL

# Load final model and encoders
model = joblib.load(PHASE4_V2_FINAL.final_model)
encoders = joblib.load(PHASE4_V2_FINAL.final_encoders)

# Make predictions
predictions = model.predict(X_test_encoded)
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `src/phase4_model_registry.py` | Model registry with frozen paths for v2 and v3 |
| `src/phase4_v3_inference.py` | Production inference API for v3 baseline |
| `src/freeze_phase4_v3_baseline.py` | One-time script to freeze v3 baseline |
| `src/run_phase4_v3_timeseries_tuning.py` | Time-series hyperparameter tuning (experiments) |
| `src/run_phase4_v3_enhanced_cv.py` | Time-series CV with enhanced features |
| `src/run_phase4_v2_with_enhanced_features.py` | Original v2 training/tuning script |
| `src/validation/time_series_cv.py` | Expanding-window CV fold generator |
| `src/features/timeseries_features.py` | Lag and rolling features (leakage-safe) |
| `src/features/holiday_features.py` | Indian holiday/festival dummies |
| `src/features/policy_features.py` | Policy phase dummies |
| `src/preprocessing/time_series_cleaning.py` | Outlier detection and missing data handling |

---

## Data Preprocessing: Outlier & Missing Data Handling

We implemented a **leakage-safe time-series cleaning module** that:

### Outlier Detection
- Detects abnormal spikes/drops in enrolment using a **moving z-score per district**
- Uses only past data (shift before rolling) to avoid leakage
- Marks outliers with an `is_outlier_event` flag so the model can learn from them
- Optionally caps extreme outliers with a rolling median to avoid distorting the model

### Missing Data Imputation
- **LOCF** (Last Observation Carried Forward) for short gaps (≤3 months)
- **State-level mean** fallback for longer gaps
- **Linear interpolation** option for smooth transitions
- All imputation uses only past/current data to avoid leakage

### Leakage Safety
- This cleaning runs inside each time-series CV fold
- The model only sees information that would have been available at prediction time
- Safe to use for both training and inference

### Usage Example

```python
from src.preprocessing.time_series_cleaning import (
    CleaningConfig, clean_uidai_time_series
)

config = CleaningConfig(
    outlier_method="zscore_moving",
    outlier_window=3,
    outlier_z_thresh=3.0,
    missing_method="locf",
    max_locf_gap=3,
)

df_clean = clean_uidai_time_series(
    df=df_train,
    state_col="state",
    district_col="district",
    date_col="month_date",
    target_col="total_enrolment",
    config=config,
)

# Check results
print(f"Outliers flagged: {df_clean['is_outlier_event'].sum()}")
print(f"Values imputed: {df_clean['total_enrolment_was_imputed'].sum()}")
```

### Output Columns

After cleaning, the DataFrame includes:
- `is_outlier_event`: 1 if the row was flagged as an outlier
- `total_enrolment_original`: Original value before any changes
- `total_enrolment_was_capped`: True if outlier was capped
- `total_enrolment_was_imputed`: True if value was imputed

---

## Experiment Naming Conventions

Use descriptive experiment names:

- `exp1_higher_depth` - Testing deeper trees
- `exp2_drop_festival` - Removing festival month features
- `exp3_lr_sweep` - Learning rate grid search
- `exp4_feature_selection` - Feature importance-based selection

---

## Experiments We Tried

### LightGBM Experiment (January 2026)

We ran a **LightGBM experiment** using the same CV folds and feature engineering as XGBoost v3.

| Model | R² (mean) | R² (std) | MAE (mean) | MAE (std) | Status |
|-------|-----------|----------|------------|-----------|--------|
| **XGBoost v3** | **0.955** | 0.049 | **116.6** | 119.8 | ✅ PRODUCTION |
| LightGBM | 0.865 | 0.179 | 242.8 | 288.8 | ⚪ Documented experiment |

**Findings:**
- XGBoost v3 significantly outperforms LightGBM on this dataset
- LightGBM showed unstable performance on fold 1 (small training set)
- R² dropped by ~9 percentage points vs XGBoost v3
- MAE more than doubled

**Conclusion:**
- ✅ **XGBoost v3 remains the only production model**
- ⚪ LightGBM is kept as a documented experiment only
- ⛔ No further hyperparameter tuning is planned before final submission

**Artifacts:**
- `artifacts/phase4_lgbm_experiment_metrics.json` - CV metrics
- `artifacts/phase4_lgbm_experiment_oof.parquet` - OOF predictions
- `src/models/phase4_lgbm_experiment.py` - Experiment module (NOT in registry)

### Future Work (Post-Submission)

The focus now shifts to **explainability and decision support**:

1. **SHAP Global Importance** - Summary plot for XGBoost v3 feature importance
2. **SHAP Local Explanations** - Waterfall charts for individual district/month predictions
3. **Uncertainty Quantification** - Prediction intervals using quantile regression or conformal prediction
4. **Planning Widgets** - What-if scenarios for UIDAI capacity planning

---

## Visualizations

The `src/phase4_visualizations.py` module generates publication-quality figures for the v3 model.

### Generate All Visualizations

```bash
python -m src.phase4_visualizations --output-dir reports/figures
```

### Available Visualizations

| Figure | Description |
|--------|-------------|
| `forecast_vs_actual_grid.png` | Grid of 6 representative districts showing actual vs predicted |
| `forecast_vs_actual_detailed.png` | Detailed view for one district |
| `residuals_over_time.png` | Time-series of residuals by district |
| `residuals_scatter.png` | Predictions vs residuals + Actual vs Predicted scatter |
| `residuals_distribution.png` | Histogram and boxplot by month |
| `cv_fold_structure.png` | Expanding-window CV diagram (train/gap/val blocks) |
| `cv_fold_metrics.png` | R² and MAE bar charts by CV fold |

### Individual Plot Functions

```python
from src.phase4_visualizations import (
    plot_forecast_vs_actual_single_district,
    plot_forecast_vs_actual_grid,
    plot_residuals_over_time,
    plot_residuals_scatter,
    plot_residual_distribution,
    plot_cv_fold_diagram,
    plot_cv_fold_detailed,
)

# Example: Single district forecast plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_forecast_vs_actual_single_district(df, "Bihar", "Patna", ax=ax)
plt.savefig("patna_forecast.png")

# Example: CV fold diagram
fig = plot_cv_fold_diagram(n_months=9, n_folds=4, gap_months=1)
plt.savefig("cv_structure.png")
```

### Visualization Features

- **Colorblind-friendly palette**: Blue (train/forecast), Red (validation/residual), Green (actual)
- **Publication-ready**: 150 DPI, clean white background, proper axis labels
- **Annotated metrics**: MAE and R² shown on plots
- **Leakage-safe emphasis**: CV diagram highlights "past → future with gap" approach

---

## Comparing Experiments

List all experiments:
```python
from src.phase4_model_registry import list_experiments
print(list_experiments())
```

Compare to final model:
```python
import json
from src.phase4_model_registry import PHASE4_V2_FINAL, get_experiment_paths

# Load final metrics
with open(PHASE4_V2_FINAL.final_metrics) as f:
    final_metrics = json.load(f)

# Load experiment metrics
exp_paths = get_experiment_paths("exp1_higher_depth")
with open(exp_paths["metrics"]) as f:
    exp_metrics = json.load(f)

# Compare
print(f"Final Test MAE: {final_metrics['split']['test']['mae']}")
print(f"Exp Test MAE:   {exp_metrics['split']['test']['mae']}")
```
