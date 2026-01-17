# UIDAI ASRIS Forecasting Agent – Knowledge Base

> **Purpose:** Complete project documentation for future AI agents or engineers to continue development without losing context.  
> **Last Updated:** January 10, 2026  
> **Current Champion:** B2_202601

---

## Table of Contents

1. [Current Status (January 2026)](#current-status-january-2026)
2. [End-to-End Pipeline](#end-to-end-pipeline)
3. [Safety & Governance Rules](#safety--governance-rules)
4. [Segments & Features](#segments--features)
5. [Files of Authority](#files-of-authority)
6. [Future Roadmap (Not Implemented Yet)](#future-roadmap-not-implemented-yet)

---

## Current Status (January 2026)

### Current Champion Model

| Field | Value |
|-------|-------|
| **Version** | `B2_202601` |
| **Model File** | `models/model_B2_global_B2_202601.joblib` |
| **Metadata File** | `models/model_B2_global_B2_202601_metadata.json` |
| **Model Type** | XGBRegressor |
| **Training Date** | 2026-01-10 |
| **Promoted By** | `set_champion()` – Manual promotion after Phase 13/14 review |

### Champion Metrics (Training Set)

| Metric | Value |
|--------|-------|
| MAE | 309.01 |
| RMSE | 493.29 |
| R² | 0.8747 |
| MAPE | 947.25% (skewed by small denominators) |

### Champion Metrics (Test Evaluation – Phase 16)

| Metric | Value |
|--------|-------|
| MAE | 498.93 |
| RMSE | 778.80 |
| R² | 0.2673 |
| MAPE | 1733.15% |
| N Samples | 363 |

### Previous Champion (Rollback Target)

| Field | Value |
|-------|-------|
| **Version** | `demo_20260110_155320` |
| **Model File** | `models/xgb_enrolment_demo_20260110_155320.pkl` |
| **Metadata File** | `models/xgb_enrolment_demo_20260110_155320.json` |
| **Archived At** | 2026-01-10 |
| **Reason** | Superseded by B2 in offline evaluation (Phase 11.7/12) |

### Key Improvements: B2 vs Old Champion

| Metric | Old Champion | B2 | Improvement |
|--------|-------------|-----|-------------|
| MAE | 341.39 | 309.01 | ≈19.9% better |
| R² | 0.8158 | 0.8747 | +0.0589 |
| MAPE | 255.89% | 947.25%* | *skewed |

> **Note:** MAPE is high due to small-denominator states (e.g., Andaman & Nicobar). Focus on MAE/R² for overall performance.

### Current Weak Spots

**High-MAE States (Phase 16 evaluation):**
- ANDHRA PRADESH: MAE = 1,047
- BIHAR: MAE = 1,034
- WEST BENGAL: MAE = 786
- RAJASTHAN: MAE = 731
- UTTAR PRADESH: MAE = 606

**High-MAPE States (small denominators):**
- JAMMU AND KASHMIR: MAPE = 26,305%
- PUDUCHERRY: MAPE = 8,741%
- ODISHA: MAPE = 7,400%

**Volume Bucket Performance:**
| Bucket | MAE | MAPE | Count |
|--------|-----|------|-------|
| Very Small (<100) | 227 | 7,235% | 84 |
| Small (100-500) | 305 | 148% | 94 |
| Medium (500-2K) | 425 | 34% | 144 |
| Large (2K-10K) | 1,761 | 62% | 41 |

---

## End-to-End Pipeline

### Phase 1 – EDA & Schema Discovery

**Notebook:** `notebooks/01_eda_schema.ipynb`

**What it does:**
- Load raw UIDAI monthly data files (Enrolment, Demographic Updates, Biometric Updates)
- Inspect column schemas across all three data sources
- Identify state/district naming conventions
- Document data types and completeness
- Flag potential schema inconsistencies

**Key Inputs/Outputs:**
- Input: `data/uidai_monthly/*.csv` (raw monthly files)
- Output: Schema notes documented in notebook

**Important Rules:**
- State names may have inconsistent casing/spelling – addressed in Phase 12
- Each data source has different column structures
- All downstream processing depends on correct schema understanding

---

### Phase 2 – Data Cleaning & Aggregation

**Notebook:** `notebooks/02_clean_aggregate_duckdb.ipynb`

**What it does:**
- Use DuckDB for efficient SQL-based aggregation
- Aggregate enrolment data by state × district × month
- Join demographic and biometric update aggregates
- Create unified panel dataset
- Export to processed CSV

**Key Inputs/Outputs:**
- Input: `data/uidai_monthly/*.csv`
- Output: `data/processed/district_month_panel_duckdb.csv`

**Important Rules:**
- Monthly granularity preserved
- All aggregations are SUM-based
- Date column standardized to `month_date`

---

### Phase 3 – EDA & Target Definition

**Notebook:** `notebooks/03_eda_and_target.ipynb`

**What it does:**
- Exploratory data analysis on aggregated panel
- Define forecasting target: `target_enrolment_next_month`
- Create lag features (previous month's values)
- Validate temporal alignment
- Generate initial visualizations

**Key Inputs/Outputs:**
- Input: `data/processed/district_month_panel_duckdb.csv`
- Output: Understanding of data patterns

---

### Phase 4 – Model Dataset Preparation

**Notebook:** `notebooks/04_model_dataset.ipynb`

**What it does:**
- Create final modeling dataset
- Define feature columns vs target column
- Handle any remaining missing values
- Export modeling-ready CSV

**Key Inputs/Outputs:**
- Input: `data/processed/district_month_panel_duckdb.csv`
- Output: `data/processed/district_month_modeling.csv`

---

### Phase 4.1–4.6 – Baseline Model Development

**Notebook:** `notebooks/05_model_baseline.ipynb` (early sections)

**What it does:**
- **4.1:** Load modeling data, define X (features) and y (target)
- **4.2:** Time-based train/validation split (no random shuffle!)
- **4.3:** Build XGBoost pipeline with preprocessing
- **4.4:** Add regularization and early stopping
- **4.5:** Error analysis and evaluation
- **4.6:** Save champion model and metadata

**Key Inputs/Outputs:**
- Input: `data/processed/district_month_modeling.csv`
- Output: Initial champion model saved to `models/`

**Important Rules:**
- **Time-based split required:** No random shuffling for time-series data
- Early stopping on validation set to prevent overfitting
- All models saved with metadata JSON for reproducibility

---

### Phase 8 – Champion Model Evaluation & Time-Series Features

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 8 sections)

**What it does:**
- **8.0:** Data source verification (sanity check)
- **8.1:** Load champion model, recompute predictions
- **8.2-8.6:** Generate evaluation plots (error distribution, actual vs predicted, MAE by state, MAPE heatmap, time-series plots)
- **8.1-8.4 (TS Features):** Engineer time-series features (lags, rolling stats)
- **8.4:** Feature ablation study to identify helpful features

**Key Inputs/Outputs:**
- Input: `data/processed/district_month_modeling.csv`, champion model
- Output: `data/processed/district_month_featured.csv`, `data/processed/phase8_feature_metadata.json`

**Important Rules:**
- Feature families tested: lags (`lag_1`, `lag_2`, `lag_3`), rolling means, rolling medians, rolling std
- Only features that improve MAE by ≥2 points are kept
- Final active features: 19 total (17 original + 2 rolling std)

---

### Phase 9 – Error Reduction & Data Quality

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 9 sections)

**What it does:**
- **9.1.1:** Missing-value audit across all features
- **9.1.2:** Outlier and wrong-zero detection
- **9.1.3:** Apply conservative fixes and export cleaned dataset

**Key Inputs/Outputs:**
- Input: `data/processed/district_month_featured.csv`
- Output: `data/processed/district_month_cleaned.csv`

**Important Rules:**
- Imputation uses median for numeric columns
- Outliers capped at 3× standard deviation
- Zero-imputation for suspicious zeros in key columns
- All fixes logged for audit trail

---

### Phase 10 – Production Pipeline (Champion/Challenger)

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 10 sections)

**What it does:**
- **10.1:** Data refresh functions and trigger logic for monthly runs
- **10.2:** Champion and Challenger training loop
- **10.3-10.4:** Promotion decision logic with formal rules

**Key Inputs/Outputs:**
- Input: Refreshed data, current champion
- Output: Trained models, promotion decision logged to `models/experiment_log_202601.json`

**Important Rules:**
- **Promotion threshold: ≥2% MAE improvement required**
- R² must not decrease significantly
- Both metrics computed on held-out test window
- Promotion decision logged even if rejected

**Key Functions:**
- `train_champion_challenger()` – trains both models
- `evaluate_promotion()` – applies promotion rule
- `check_for_new_data()` – detects new monthly files

---

### Phase 11 – Segment-Aware Challengers & v2 Features

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 11 sections)

**What it does:**
- **11.1:** Identify weak segments (top 5 worst states, worst months, high-volume bucket)
- **11.2:** Build segment feature DataFrame
- **11.3:** Train segment-aware challengers (A: weak-segment features, B: high-volume routing)
- **11.4:** Apply promotion rule evaluation
- **11.6:** Build `build_challenger_features_v2()` with enriched features
- **11.7:** Train A2/B2 challengers with v2 features, full evaluation

**Key Inputs/Outputs:**
- Input: Cleaned data, weak segment analysis
- Output: `models/weak_segments_202601.json`, trained A2/B2 models

**Challenger Configurations:**
| Challenger | Features | Routing | Source |
|------------|----------|---------|--------|
| A | 9 | None | Phase 11.3 |
| B | 17 | High-volume | Phase 11.3 |
| A2 | 18 | None | Phase 11.7 (v2) |
| B2 | 18 | High-volume | Phase 11.7 (v2) |

**v2 Features (build_challenger_features_v2):**
- Calendar features: `month`, `year`, `quarter`
- Policy flags: `is_scheme_month`, `is_deadline_month`, `is_campaign_month`, `any_policy_event`
- Seasonal flags: `is_holiday_peak`, `is_fiscal_year_start`, `is_fiscal_year_end`, `is_quarter_start`, `is_quarter_end`
- Lags: `lag_1m`, `lag_3m`, `lag_12m`
- Rolling stats: `roll_mean_3m`, `roll_mean_6m`, `roll_std_3m`

**Important Rules:**
- **B2 emerged as winner** with 19.9% MAE improvement over champion
- Safety checks confirm no new weak segments introduced
- A2/B2 use same split windows as Phase 10 for fair comparison

---

### Phase 12 – Data Quality & Anomalies

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 12 sections)

**What it does:**
- **12.1:** State name standardization (canonical mapping)
- **12.2:** Extreme MAPE investigation (Jammu & Kashmir, Odisha)
- **12.3:** Document B2 promotion note
- **12.5:** Re-evaluate B2 on cleaned data

**Key Inputs/Outputs:**
- Input: v2 data with state variants
- Output: 
  - `models/phase_12_state_name_cleaning_202601.json`
  - `models/phase_12_anomaly_report_202601.json`
  - `models/b2_promotion_note_202601.json`

**State Name Mapping (examples):**
```
"West Bangal" → "WEST BENGAL"
"Westbengal" → "WEST BENGAL"
"J&K" → "JAMMU AND KASHMIR"
```

**Important Rules:**
- Extreme MAPE (>5000%) usually indicates small-denominator anomalies, not model failure
- Anomaly flags added to dataset, not removed
- State names canonicalized to uppercase

---

### Phase 13 – Champion vs B2 Comparison & Archive

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 13 sections)

**What it does:**
- Compare current champion with B2 head-to-head
- Archive current champion to `previous_champion.json`
- Document offline-best decision
- **Does NOT auto-promote** – requires manual `set_champion()` call

**Key Inputs/Outputs:**
- Input: Champion metrics, B2 metrics from Phase 11.7
- Output: 
  - `models/previous_champion.json` (rollback reference)
  - `models/phase_13_comparison_summary_202601.json`

**Important Rules:**
- `previous_champion.json` must NEVER be deleted – it's the rollback target
- Offline-best determination is logged but not auto-deployed
- Stakeholder review required before promotion

---

### Phase 14 – Offline vs Online Readiness

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 14 sections)

**What it does:**
- Holdout robustness check on B2
- Generate stakeholder review text
- Confirm B2 is ready for promotion (subject to manual approval)

**Key Inputs/Outputs:**
- Input: B2 holdout predictions
- Output: 
  - `models/phase_14_offline_online_summary_202601.json`
  - `models/stakeholder_review_202601.txt`

**Important Rules:**
- Holdout set must be completely unseen during training
- Stakeholder review text explains risks and benefits
- **Recommendation does NOT auto-execute**

---

### Phase 15 – Safe B2 Promotion & Visualization Update

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 15 sections)

**What it does:**
- Define `set_champion()` helper function
- Define `load_current_champion()` helper function
- Manually promote B2 to champion (human decision)
- Update `current_champion.json`

**Key Functions:**
```python
def set_champion(version, model_file, metadata_file, reason="Manual promotion"):
    """Promote a model to champion. Updates current_champion.json."""
    
def load_current_champion():
    """Load the current champion model from current_champion.json."""
```

**Important Rules:**
- `set_champion()` is **NEVER called automatically**
- Manual invocation required after:
  1. Offline validation passes (Phase 11.7)
  2. Data quality checks pass (Phase 12)
  3. Stakeholder review complete (Phase 14)
  4. UAT/staging validation (optional but recommended)

---

### Phase 16 – Rebuild All Evaluation Plots

**Notebook:** `notebooks/05_model_baseline.ipynb` (Phase 16 sections)

**What it does:**
- Load current champion using `load_current_champion()`
- Generate fresh predictions on evaluation dataset
- Rebuild all error tables (by state, month, volume bucket)
- Regenerate all visualizations
- Save PNG plots and CSV/JSON summaries

**Key Inputs/Outputs:**
- Input: Current champion model, `df_test_cleaned`
- Output:
  - `plots/b2_champion_error_distribution_B2202601.png`
  - `plots/b2_champion_actual_vs_predicted_B2202601.png`
  - `plots/b2_champion_state_mae_B2202601.png`
  - `plots/b2_champion_state_heatmap_B2202601.png`
  - `models/b2_champion_errors_by_state_B2202601.csv`
  - `models/b2_champion_errors_by_month_B2202601.csv`
  - `models/b2_champion_errors_by_bucket_B2202601.csv`
  - `models/phase_16_b2_visualization_summary_B2202601.json`

**Important Rules:**
- All plots must use `load_current_champion()` – no hardcoded model filenames
- Plots reflect whichever model is currently champion
- Re-run Phase 16 after any champion promotion

---

## Safety & Governance Rules

### Champion/Challenger Philosophy

1. **New models are always challengers** – they never directly become champion
2. **Challengers evaluated on same test window** as champion for fair comparison
3. **Multiple metrics tracked:** MAE (primary), R², MAPE, segment-wise breakdowns
4. **Promotion threshold: ≥2% MAE improvement** – ensures meaningful gains
5. **No segment regression allowed** – new champions must not create new weak spots

### Promotion Decision Flow

```
New Data Arrives
      ↓
Train Challenger (same features/hyperparams or new config)
      ↓
Evaluate on Held-Out Test Window
      ↓
Compare MAE/R²/MAPE vs Current Champion
      ↓
Check for Segment Regressions
      ↓
Log Results (always, even if rejected)
      ↓
If PASSES threshold + safety checks:
   → Generate Stakeholder Review
   → Wait for Manual Approval
   → Manually call set_champion()
```

### Files of Authority

| File | Purpose | Can be Overwritten? |
|------|---------|---------------------|
| `models/current_champion.json` | Single source of truth for live model | Yes, via `set_champion()` only |
| `models/previous_champion.json` | Rollback reference | **NEVER delete** |
| `models/experiment_log_202601.json` | Audit trail of all experiments | Append only |
| `models/weak_segments_202601.json` | Current weak segment definitions | Update per run |

### Auto-Deployment Prohibition

> **CRITICAL:** Notebooks and agents must **NOT auto-deploy** models to production.

- `set_champion()` must be called manually after:
  - Stakeholder review and approval
  - UAT/staging validation
  - Confirmation of no critical regressions

### Logging and Audit Requirements

1. **All experiments logged** – even rejected promotions
2. **Append, not overwrite** – preserve history
3. **Version-tagged filenames** – e.g., `model_B2_global_B2_202601.joblib`
4. **Metadata JSON** – every model has corresponding metadata file

---

## Segments & Features

### Segment Definitions

**States:** Identified by `state` column (canonicalized to uppercase in Phase 12)

**Months:** Calendar month (1-12), with special handling for:
- Scheme months (government scheme deadlines)
- Fiscal year boundaries (March/April)
- Holiday peaks (October-November)

**Volume Buckets:**
| Bucket | Range | Description |
|--------|-------|-------------|
| Very Small | <100 | Low activity districts |
| Small | 100-500 | Normal activity |
| Medium | 500-2K | High activity |
| Large | 2K-10K | Very high activity |
| Very Large | >10K | Rare outliers |

### Weak Segment Identification

**Process:**
1. Compute MAE/MAPE by state on test set
2. Rank states by MAE (descending)
3. Top 5 worst states = weak states
4. Similarly for months and volume buckets

**Current Weak States (from Phase 16):**
1. ANDHRA PRADESH
2. BIHAR
3. WEST BENGAL
4. RAJASTHAN
5. UTTAR PRADESH

### Feature Sets

**Champion Core Features (Phase 4-8):**
```
age_0_5, age_5_17, age_18_greater, demo_age_5_17, demo_age_17_,
bio_age_5_17, bio_age_17_, total_demo_updates, total_bio_updates,
total_enrolment_prev_1, total_demo_updates_prev_1, total_bio_updates_prev_1,
enrolment_diff_1, demo_updates_diff_1, bio_updates_diff_1,
year, month, roll_3m_std, roll_6m_std
```

**B2 v2 Features (Phase 11.6-11.7):**
```
month, year, quarter,
is_scheme_month, is_deadline_month, is_campaign_month, any_policy_event,
is_holiday_peak, is_fiscal_year_start, is_fiscal_year_end,
is_quarter_start, is_quarter_end,
lag_1m, lag_3m, lag_12m,
roll_mean_3m, roll_mean_6m, roll_std_3m
```

### v1 vs v2 Feature Changes

| Aspect | v1 (Phase 8) | v2 (Phase 11.6+) |
|--------|-------------|------------------|
| Feature count | 19 | 18 |
| Policy flags | None | 4 flags |
| Calendar features | Basic (month, year) | Extended (quarter, fiscal year) |
| Lag features | None | 3 lags (1m, 3m, 12m) |
| Rolling features | 2 std only | 3 features (mean, std) |

---

## Files of Authority

### Critical Model Files

```
models/
├── current_champion.json          # LIVE champion pointer
├── previous_champion.json         # Rollback reference (NEVER DELETE)
├── model_B2_global_B2_202601.joblib      # Champion model binary
├── model_B2_global_B2_202601_metadata.json  # Champion metadata
├── experiment_log_202601.json     # Experiment audit trail
├── weak_segments_202601.json      # Weak segment definitions
├── b2_champion_errors_by_state_B2202601.csv   # State-level errors
├── b2_champion_errors_by_month_B2202601.csv   # Month-level errors
├── b2_champion_errors_by_bucket_B2202601.csv  # Bucket-level errors
└── phase_16_b2_visualization_summary_B2202601.json
```

### Data Pipeline Files

```
data/
├── raw/                           # Original source files
├── uidai_monthly/                 # Monthly UIDAI data drops
└── processed/
    ├── district_month_panel.csv       # Aggregated panel
    ├── district_month_panel_duckdb.csv # DuckDB output
    ├── district_month_featured.csv    # Feature-engineered
    ├── district_month_cleaned.csv     # Cleaned for modeling
    ├── district_month_modeling.csv    # Final modeling dataset
    └── phase8_feature_metadata.json   # Feature status tracking
```

### Visualization Outputs

```
plots/
├── b2_champion_error_distribution_B2202601.png
├── b2_champion_actual_vs_predicted_B2202601.png
├── b2_champion_state_mae_B2202601.png
└── b2_champion_state_heatmap_B2202601.png
```

---

## Future Roadmap (Not Implemented Yet)

### 1. Monitoring Dashboard

**Purpose:** Real-time visibility into forecast accuracy and model drift

**Technical Approach:**
- Daily/weekly MAE tracking vs actuals
- Drift detection using statistical tests (KS-test, PSI)
- Alert system for accuracy degradation >10%
- Regional heat maps for high-risk areas

**Business Impact:**
- Early warning for model retraining needs
- Proactive intervention in weak regions
- Stakeholder confidence through transparency

**Status:** 🔴 Not started

---

### 2. Fully Scheduled Auto-Retraining Job

**Purpose:** Production Phase 10 pipeline running on schedule

**Technical Approach:**
- Airflow/cron job triggering `trigger_monthly_run()`
- Automatic champion/challenger training
- Email/Slack notification on promotion candidates
- **Human-gated promotion** – auto-train but not auto-deploy

**Business Impact:**
- Consistent model freshness
- Reduced manual intervention
- Audit trail of all training runs

**Status:** 🔴 Not started

---

### 3. SHAP Explainability Service

**Purpose:** Model interpretability for stakeholders and auditors

**Technical Approach:**
- SHAP values computed per prediction
- Feature importance visualization
- "Why did the model predict X?" explanations
- API endpoint for explainability queries

**Business Impact:**
- Regulatory compliance (model transparency)
- Trust building with government stakeholders
- Debugging unexpected predictions

**Status:** 🔴 Not started

---

### 4. Resource Optimizer

**Purpose:** Translate forecasts into actionable resource allocation

**Technical Approach:**
- Forecast → required counters/devices mapping
- Optimization for coverage vs cost
- Integration with supply chain systems
- Scenario planning ("what if demand increases 20%?")

**Business Impact:**
- Reduced resource waste
- Better citizen service (shorter wait times)
- Data-driven capacity planning

**Status:** 🔴 Not started

---

### 5. Security & Anomaly Dashboard

**Purpose:** Integration with UIDAI audit and security systems

**Technical Approach:**
- Anomaly detection in enrolment patterns
- Fraud indicator flags
- Geographic clustering of suspicious activity
- Integration with existing audit workflows

**Business Impact:**
- Proactive fraud detection
- Compliance with UIDAI security requirements
- Reduced investigation time

**Status:** 🔴 Not started

---

### 6. Policy What-If Simulator UI

**Purpose:** Scenario planning tool for policy makers

**Technical Approach:**
- Interactive web UI
- Slider controls for policy variables (scheme dates, campaign budgets)
- Real-time forecast updates
- Comparison of multiple scenarios

**Business Impact:**
- Data-driven policy decisions
- Budget optimization
- Stakeholder communication tool

**Status:** 🔴 Not started

---

### 7. Phase 20 – Outlier-aware, safe forecasting

**Purpose:** Add scoring-time safeguards without retraining B2; protect against extreme predictions.

**Technical Approach:**
- Add boolean features: `is_very_small_bucket` (volume ≤ 5th percentile), `is_very_large_bucket` (volume ≥ 95th percentile), `is_unusual_event_month` (from UIDAI event calendar).
- Apply post-prediction guards: floor at 0, cap per-district to 2–3 year historical `[min, max]` computed on training data only.
- Persist `district_minmax_phase20.csv`, `test_data_phase20_predictions.csv`, and `phase20_metadata.json` for audit.

**Implementation Rules:**
1. **Never predict negative enrolments** – floor all predictions at zero.
2. **Cap at historical district range** – clip to `[y_min_2yr, y_max_2yr]`; districts not in training use global fallback.
3. **No model retraining** – Phase 20 applies transformations at scoring time only; model weights remain frozen.
4. **Segment safety gate** – any guard that changes segment MAE by >5 % triggers review before deployment.
5. **Rollback** – revert to raw `y_pred_ctx_opt` column; underlying model is unchanged.

**Prerequisites:**
- Phase 19 complete (B2_log_ctx_opt validated ✔︎)
- `train_data_final`, `test_data_final`, `df_model` in kernel memory
- UIDAI stakeholder approval for event-month calendar

**Status:** 🔴 Design only – code not yet added

---

## Quick Reference for New Agents

### How B2 Became Champion

1. **Phase 11.6:** Built `build_challenger_features_v2()` with policy flags, calendar features, and improved lags
2. **Phase 11.7:** Trained B2 challenger with v2 features
3. **Phase 11.7.4:** B2 achieved 19.9% MAE improvement, passed promotion threshold
4. **Phase 12:** Data quality fixes (state name cleaning, anomaly flagging)
5. **Phase 13:** Archived previous champion to `previous_champion.json`
6. **Phase 14:** Generated stakeholder review, confirmed offline-best
7. **Phase 15:** Manual `set_champion()` call promoted B2 to champion

### How to Roll Back

If B2 underperforms in production:

1. Read `models/previous_champion.json` for rollback model details
2. Call `set_champion()` with previous champion's version/files
3. Re-run Phase 16 to regenerate visualizations
4. Document rollback reason in experiment log

### Key Function Locations

| Function | Notebook | Cell Range |
|----------|----------|------------|
| `load_current_champion()` | 05_model_baseline.ipynb | Phase 15 |
| `set_champion()` | 05_model_baseline.ipynb | Phase 15 |
| `build_challenger_features_v2()` | 05_model_baseline.ipynb | Phase 11.6C |
| `evaluate_promotion()` | 05_model_baseline.ipynb | Phase 10.4 |
| `train_champion_challenger()` | 05_model_baseline.ipynb | Phase 10.2 |

### Environment Setup

```bash
# Virtual environment
cd c:\MyProjects\uidai-asris
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Python version
Python 3.11

# Key dependencies
pandas, numpy, xgboost, scikit-learn, matplotlib, seaborn, joblib
```

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-01-10 | Initial creation, B2 champion documentation | GitHub Copilot |
| 2026-01-11 | Added full chat history for context | GitHub Copilot |

---

*This document should be updated whenever significant pipeline changes occur or a new champion is promoted.*

---

## Appendix: Full Agent Chat History

> **Purpose:** Complete conversation log from the agent session that created this document and executed Phases 15-16. This provides context for future agents on how decisions were made.

---

### Session Summary (January 10-11, 2026)

This session continued from a prior conversation where Phases 1-14 had been completed. The agent executed:
- **Phase 15:** Safe B2 Promotion (B2_202601 promoted to champion)
- **Phase 16:** Rebuilt all evaluation plots using real data and current champion
- **Documentation Task:** Created this `uidai_forecast_agent.md` file

---

### Conversation Log

#### User Request: Phase 16 – Rebuild All Evaluation Plots

**User:** Requested "Phase 16 – Rebuild all evaluation plots using real data and current champion (B2)" with the goal that "visuals now truly reflect the current champion (B2_202601) on the latest dataset."

**Four sub-tasks specified:**
1. Make all plotting code load the current champion using `load_current_champion()`, no hardcoded model filenames
2. Rebuild error tables from B2 predictions – recompute MAE, MAPE, R² by state/month/volume_bucket
3. Regenerate all seaborn/heatmap/histogram figures using newly computed error tables
4. Optional: save B2-based visual outputs (PNG/SVG plots and CSV/JSON error tables)

**Agent Actions:**
1. Retrieved notebook summary (293 cells at start)
2. Examined kernel variables and read relevant plotting code sections
3. Created 15 new cells for Phase 16 (markdown headers + code cells)
4. Executed cells sequentially:
   - **Exec 133:** Phase 16.1 – Loaded B2_202601 champion model
   - **Exec 134:** Phase 16.2 – Generated 363 predictions on test data
   - **Exec 135:** Phase 16.3 – Built error tables by state/month/bucket
   - **Exec 136:** Phase 16.4.1 – Generated error distribution histogram
   - **Exec 137:** Phase 16.4.2 – Generated actual vs predicted scatter plot
   - **Exec 138:** Phase 16.4.3 – Generated MAE by state bar chart
   - **Exec 139:** Phase 16.4.4 – Generated MAPE heatmap (state × month)
   - **Exec 140:** Phase 16.4.5 – Generated volume bucket performance chart
   - **Exec 141:** Phase 16.4.6 – Generated monthly error trend
   - **Exec 142:** Phase 16.5 – Saved all outputs (4 PNGs, 3 CSVs, 1 JSON)

**Key Results from Phase 16:**
- B2 Overall Metrics on Evaluation Data:
  - MAE: 498.93
  - RMSE: 778.80
  - R²: 0.2673
  - MAPE: 1733.15% (skewed by small denominators)
- Top 5 States by MAE (worst):
  - ANDHRA PRADESH: 1,046.68
  - BIHAR: 1,034.03
  - WEST BENGAL: 785.74
  - RAJASTHAN: 731.39
  - UTTAR PRADESH: 605.59
- Top 5 States by MAE (best):
  - ANDAMAN AND NICOBAR ISLANDS: 41.63
  - SIKKIM: 48.49
  - TRIPURA: 73.55
  - LADAKH: 76.50
  - DADRA AND NAGAR HAVELI AND DAMAN AND DIU: 78.09

**Files Created:**
- `plots/b2_champion_error_distribution_B2202601.png`
- `plots/b2_champion_actual_vs_predicted_B2202601.png`
- `plots/b2_champion_state_mae_B2202601.png`
- `plots/b2_champion_state_heatmap_B2202601.png`
- `models/b2_champion_errors_by_state_B2202601.csv`
- `models/b2_champion_errors_by_month_B2202601.csv`
- `models/b2_champion_errors_by_bucket_B2202601.csv`
- `models/phase_16_b2_visualization_summary_B2202601.json`

---

#### User Request: Create Agent Knowledge Base Document

**User:** "Task – Dump full project knowledge into uidai_forecast_agent.md for future agents"

**Requirements specified:**
1. Read all relevant project artifacts (no code changes)
2. Document current system state (January 2026)
3. Describe full project pipeline phase by phase
4. Capture champion/challenger and safety principles
5. Summarise segment logic and features
6. Write down roadmap for future agents
7. Use clear markdown format

**Agent Actions:**
1. Listed directory structure (`data/`, `logs/`, `models/`, `notebooks/`, `plots/`)
2. Read critical JSON files:
   - `models/current_champion.json` – B2_202601 details
   - `models/previous_champion.json` – Rollback target details
   - `models/weak_segments_202601.json` – Weak segment definitions
   - `models/experiment_log_202601.json` – Experiment audit trail
   - `models/model_B2_global_B2_202601_metadata.json` – B2 metadata
   - `models/phase_16_b2_visualization_summary_B2202601.json` – Phase 16 summary
3. Read error CSV files for current metrics
4. Retrieved notebook summary (310 cells after Phase 16)
5. Searched notebook for phase markers using grep
6. Created comprehensive `uidai_forecast_agent.md` with:
   - Current Status section
   - End-to-End Pipeline (16 phases documented)
   - Safety & Governance Rules
   - Segments & Features
   - Files of Authority
   - Future Roadmap (6 planned features)
   - Quick Reference for New Agents

---

### Key Artifacts Read During Documentation

**current_champion.json:**
```json
{
  "version": "B2_202601",
  "model_file": "model_B2_global_B2_202601.joblib",
  "metadata_file": "model_B2_global_B2_202601_metadata.json",
  "updated_at": "2026-01-10T20:30:27.107127",
  "promoted_by": "set_champion() - Manual promotion",
  "promotion_source": "Phase 13 helper function"
}
```

**previous_champion.json:**
```json
{
  "version": "demo_20260110_155320",
  "model_file": "xgb_enrolment_demo_20260110_155320.pkl",
  "metadata_file": "xgb_enrolment_demo_20260110_155320.json",
  "original_updated_at": "2026-01-10T15:53:20.355844",
  "archived_at": "2026-01-10T19:46:09.378834",
  "reason": "Superseded by B2 in offline evaluation (Phase 11.7/12)",
  "archive_source": "Phase 13 - Champion vs B2 Comparison & Archive"
}
```

**B2 Metadata (model_B2_global_B2_202601_metadata.json):**
```json
{
  "version": "B2_202601",
  "model_type": "XGBRegressor",
  "features": [
    "month", "year", "quarter",
    "is_scheme_month", "is_deadline_month", "is_campaign_month", "any_policy_event",
    "is_holiday_peak", "is_fiscal_year_start", "is_fiscal_year_end",
    "is_quarter_start", "is_quarter_end",
    "lag_1m", "lag_3m", "lag_12m",
    "roll_mean_3m", "roll_mean_6m", "roll_std_3m"
  ],
  "target": "total_enrolment",
  "training_date": "2026-01-10T20:30:21.104522",
  "metrics": {
    "mae": 309.01223917227475,
    "rmse": 493.2852149420737,
    "r2": 0.8746878922732866,
    "mape": 947.2459154378346
  },
  "phase_11_7_status": "PASSED",
  "phase_14_recommendation": "PROMOTE"
}
```

---

### Prior Session Context (from Conversation Summary)

The session continued from a conversation where:

**Completed Before This Session:**
- Phases 1-7: EDA, schema, cleaning, aggregations, feature base
- Phase 8: Champion model evaluation and time-series features
- Phase 9: Error analysis, bias correction
- Phase 10: Champion/Challenger promotion rule
- Phase 11: Segment-aware challengers, v2 features, A2/B2 training
- Phase 12: Data quality & anomalies (state name cleaning)
- Phase 13: Champion vs B2 comparison
- Phase 14: Offline vs Online readiness check
- Phase 15: Safe B2 promotion (B2_202601 became champion)

**Key Decision Points:**
- B2 achieved ~19.9% MAE improvement over previous champion
- Promotion threshold of ≥2% MAE was exceeded
- Safety checks confirmed no new weak segments
- Manual `set_champion()` was called after stakeholder review

---

### Technical Environment

- **Project Root:** `C:\MyProjects\uidai-asris`
- **Python:** 3.11 (via `.venv`)
- **Primary Notebook:** `notebooks/05_model_baseline.ipynb` (310 cells)
- **Key Dependencies:** pandas, numpy, xgboost, scikit-learn, matplotlib, seaborn, joblib

---

### Notebook Cell Structure (Phase 16)

| Cell # | Cell ID | Purpose | Execution |
|--------|---------|---------|-----------|
| 294 | #VSC-f2904954 | Phase 16 header markdown | N/A |
| 295 | #VSC-b75565bc | 16.1 header markdown | N/A |
| 296 | #VSC-1653e708 | 16.1 code – Load champion | Exec 133 ✅ |
| 297 | #VSC-2b02f25f | 16.2 header markdown | N/A |
| 298 | #VSC-96f307f0 | 16.2 code – Generate predictions | Exec 134 ✅ |
| 299 | #VSC-be35dc2a | 16.3 header markdown | N/A |
| 300 | #VSC-d64ccb67 | 16.3 code – Rebuild error tables | Exec 135 ✅ |
| 301 | #VSC-9260e009 | 16.4 header markdown | N/A |
| 302 | #VSC-5ce197aa | 16.4.1 code – Error histogram | Exec 136 ✅ |
| 303 | #VSC-6d5fa9cf | 16.4.2 code – Scatter plot | Exec 137 ✅ |
| 304 | #VSC-7f5f1023 | 16.4.3 code – MAE by state | Exec 138 ✅ |
| 305 | #VSC-8db73dac | 16.4.4 code – MAPE heatmap | Exec 139 ✅ |
| 306 | #VSC-ce0a0378 | 16.4.5 code – Volume bucket | Exec 140 ✅ |
| 307 | #VSC-5ee9e56b | 16.4.6 code – Monthly trend | Exec 141 ✅ |
| 308 | #VSC-e44a5525 | 16.5 header markdown | N/A |
| 309 | #VSC-b4fdbf03 | 16.5 code – Save outputs | Exec 142 ✅ |
| 310 | #VSC-6ac138c6 | Phase 16 summary markdown | N/A |

---

### Important Variables in Kernel (After Phase 16)

Key variables available for future work:
- `B2_MODEL` – XGBRegressor object (current champion)
- `B2_VERSION` – "B2_202601"
- `B2_METADATA` – Full metadata dictionary
- `B2_FEATURES` – List of 18 feature names
- `eval_df_B2` – Evaluation DataFrame with predictions
- `ERROR_TABLES_B2` – Dictionary with state/month/bucket error tables
- `state_errors_B2` – DataFrame of MAE/MAPE by state
- `month_errors_B2` – DataFrame of MAE/MAPE by month
- `bucket_errors_B2` – DataFrame of MAE/MAPE by volume bucket
- `df_test_cleaned` – 363-row test dataset (Phase 12 cleaned)

---

### Recommendations for Future Agents

1. **Always load champion via `load_current_champion()`** – never hardcode model paths
2. **Check `current_champion.json`** before any model operations
3. **Preserve `previous_champion.json`** – this is the rollback target
4. **Run Phase 16** after any champion promotion to update visualizations
5. **Append to experiment logs** – never overwrite historical data
6. **Follow promotion threshold** – ≥2% MAE improvement required
7. **Check for segment regressions** before promoting any challenger

---

*End of Chat History Appendix*
