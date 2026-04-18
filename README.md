# FBI: ICU Weaning & Readmission ML Pipeline

This project audits, cleans, and models ICU clinical data to predict:
- **Weaning success**
- **Readmission risk**

It is built around a practical end-to-end workflow:
1. Data audit
2. Data fixing/cleaning
3. ML training + evaluation + interpretation
4. Benchmark/audit checks

---

## Why this project is needed

In ICU settings, early and reliable risk prediction can support better clinical decisions.  
This project aims to make that process more structured by:
- identifying data quality issues before modeling,
- reducing leakage risk (patient-level splitting),
- and producing interpretable model outputs (feature importance + SHAP).

---

## What we are doing in this project

### 1) Data auditing (`data_audit.py`)
- Validates schema and dtypes
- Detects duplicates and possible leakage signals
- Checks missing values and clinical range violations
- Evaluates class imbalance and bias indicators
- Generates audit plots and a text report in `audit_outputs/`

### 2) Data fixing (`data_fix.py`)
- Removes/aggregates duplicates
- Caps impossible/extreme clinical values
- Imputes missing values
- Encodes features (e.g., gender)
- Applies train/test split by `subject_id` (leakage prevention)
- Saves cleaned data to `fix_outputs/`

### 3) ML pipeline (`ml_pipeline.py`)
- Builds engineered features (PF ratio, RSBI, MAP, etc.)
- Trains Logistic Regression, Random Forest, XGBoost
- Evaluates with ROC-AUC, PR-AUC, confusion matrix, classification metrics
- Adds SHAP-based interpretation
- Saves plots/results to `ml_outputs/`

### 4) Benchmark audit (`benchmark_audit.py`)
- Adds robustness checks (stability, calibration, leakage checks)
- Compares with published benchmark ranges
- Produces additional quality-review outputs

---

## Repository structure

```text
FBI/
├── data_audit.py
├── data_fix.py
├── ml_pipeline.py
├── benchmark_audit.py
├── fix_encoding.py
├── fbi_data_local.csv
├── updated.csv
├── audit_outputs/
├── fix_outputs/
└── ml_outputs/
```

---

## Requirements

### Runtime
- Python **3.9+** (3.10+ recommended)

### Python packages
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `xgboost`
- `shap`

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost shap
```

---

## How to run

Run scripts in this order:

```bash
python data_audit.py
python data_fix.py
python ml_pipeline.py
python benchmark_audit.py
```

---

## Important current flaws / limitations

1. **Hardcoded Windows paths in scripts**  
   Current scripts use paths like `e:\fbi ML\...`.  
   You must update `DATA_PATH` and `OUT_DIR` in scripts to local paths before running in other environments.

2. **No packaging / dependency lock file yet**  
   There is no `requirements.txt` or `pyproject.toml`, so environment reproducibility is limited.

3. **No automated test suite configured**  
   There are currently no unit/integration tests in the repository.

4. **No CI/CD checks configured in repo files**  
   Quality checks are manual script execution today.

5. **Dataset dependence**  
   Pipeline assumes specific ICU schema columns are present and correctly named.

6. **External validation not integrated as a standard step**  
   Results are primarily on the current data flow; broader generalization needs further validation.

---

## Current outputs

- **Audit artifacts**: `audit_outputs/`
- **Cleaned datasets**: `fix_outputs/`
- **Model charts/interpretability artifacts**: `ml_outputs/`

These include class imbalance charts, correlation heatmaps, ROC/PR plots, feature importance, and SHAP summaries.

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file.
