# -*- coding: utf-8 -*-
"""
=============================================================================
DATA CLEANING & FIXING SCRIPT - ICU (MIMIC-IV) Dataset  [V2 - WITH NEW FEATURES]
Applies all fixes identified in the data audit
Now handles 28 columns including: icu_los_days, hosp_los_days,
charlson_comorbidity_index, sofa_score, hypertension, diabetes, copd, heart_failure
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

# -- paths -----------------------------------------------------------------
DATA_PATH = r"e:\fbi ML\updated.csv"
OUT_DIR   = r"e:\fbi ML\fix_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -- load ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("ORIGINAL dataset: {} rows x {} columns".format(df.shape[0], df.shape[1]))

id_cols      = ['subject_id', 'hadm_id', 'stay_id']
label_cols   = ['weaning_success', 'readmission']
binary_flags = ['hypertension', 'diabetes', 'copd', 'heart_failure']
feature_cols = [c for c in df.columns if c not in id_cols + label_cols]
numeric_features = [c for c in feature_cols if c not in ['gender'] + binary_flags]


# =========================================================================
#  FIX 1: REMOVE DUPLICATE ROWS (Critical - Data Leakage)
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 1: REMOVING DUPLICATE ROWS")
print("=" * 70)

# Strategy: Keep one row per (subject_id, hadm_id, stay_id) combination
# For labels with conflicting values across duplicates, take the majority vote
print("\nBefore de-duplication: {} rows".format(len(df)))

# Group by ID columns + all features, aggregate labels
group_cols = id_cols + [c for c in feature_cols]

# Method: For each unique (subject_id, hadm_id, stay_id), keep ONE row
# with the MAJORITY label value
def majority_vote(series):
    """Return the most common value (majority vote)."""
    return series.mode().iloc[0]

# First, let's aggregate by stay_id (the finest grain)
df_dedup = df.groupby('stay_id', as_index=False).agg({
    'subject_id': 'first',
    'hadm_id': 'first',
    'age': 'first',
    'gender': 'first',
    'heart_rate': 'first',
    'resp_rate': 'first',
    'spo2': 'first',
    'sys_bp': 'first',
    'dia_bp': 'first',
    'fio2': 'first',
    'peep': 'first',
    'tidal_volume': 'first',
    'pao2': 'first',
    'paco2': 'first',
    'ph': 'first',
    'lactate': 'first',
    'creatinine': 'first',
    # NEW FEATURES
    'icu_los_days': 'first',
    'hosp_los_days': 'first',
    'charlson_comorbidity_index': 'first',
    'sofa_score': 'first',
    'hypertension': 'first',
    'diabetes': 'first',
    'copd': 'first',
    'heart_failure': 'first',
    # LABELS (majority vote)
    'weaning_success': majority_vote,
    'readmission': majority_vote,
})

print("After de-duplication (1 row per stay_id): {} rows".format(len(df_dedup)))
print("Rows removed: {} ({:.1f}%)".format(
    len(df) - len(df_dedup), (len(df) - len(df_dedup)) / len(df) * 100))

df = df_dedup.copy()


# =========================================================================
#  FIX 2: CAP/REMOVE IMPOSSIBLE VALUES (Clinical Range Violations)
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 2: FIXING IMPOSSIBLE VALUES (Clinical Range Capping)")
print("=" * 70)

# Define clinical bounds -- values outside these are IMPOSSIBLE
# (not just unusual, but physically/physiologically impossible)
impossible_bounds = {
    'heart_rate':    (0, 300),      # absolute max even in extreme tachycardia
    'resp_rate':     (0, 80),       # absolute max
    'spo2':          (0, 100),      # SpO2 CANNOT exceed 100%
    'sys_bp':        (0, 350),      # extreme hypertensive crisis
    'dia_bp':        (0, 250),      # extreme
    'fio2':          (0, 100),      # 0-100% scale
    'peep':          (0, 40),       # max ventilator setting
    'tidal_volume':  (0, 2000),     # max even for large patients
    'ph':            (6.0, 8.0),    # extreme but possible
    'pao2':          (0, 700),      # max on 100% FiO2
    'paco2':         (0, 250),      # extreme hypercapnia
    'lactate':       (0, 30),       # extreme lactic acidosis
    'creatinine':    (0, 30),       # extreme renal failure
    # NEW FEATURES
    'icu_los_days':               (0, 365),   # max 1 year
    'hosp_los_days':              (0, 730),   # max 2 years
    'charlson_comorbidity_index': (0, 37),    # max possible CCI
    'sofa_score':                 (0, 24),    # max SOFA is 24
}

# Replace impossible values with NaN (they are clearly data errors)
for col, (lo, hi) in impossible_bounds.items():
    if col not in df.columns:
        continue
    mask_impossible = (df[col] < lo) | (df[col] > hi)
    n_impossible = mask_impossible.sum()
    if n_impossible > 0:
        print("  {}: {} impossible values set to NaN".format(col, n_impossible))
        df.loc[mask_impossible, col] = np.nan

# Now apply clinical Winsorisation (cap at realistic clinical bounds)
clinical_caps = {
    'heart_rate':    (30, 220),
    'resp_rate':     (4, 60),
    'spo2':          (50, 100),
    'sys_bp':        (40, 250),
    'dia_bp':        (10, 150),
    'fio2':          (21, 100),
    'peep':          (0, 30),
    'tidal_volume':  (50, 1500),
    'ph':            (6.5, 7.8),
    'pao2':          (20, 600),
    'paco2':         (10, 150),
    'lactate':       (0.1, 25),
    'creatinine':    (0.1, 25),
    # NEW FEATURES
    'icu_los_days':               (0, 200),
    'hosp_los_days':              (0, 365),
    'charlson_comorbidity_index': (0, 20),
    'sofa_score':                 (0, 24),
}

print("\nApplying clinical Winsorisation (capping at realistic bounds):")
for col, (lo, hi) in clinical_caps.items():
    if col not in df.columns:
        continue
    n_below = (df[col] < lo).sum()
    n_above = (df[col] > hi).sum()
    if n_below + n_above > 0:
        df[col] = df[col].clip(lower=lo, upper=hi)
        print("  {}: capped {} low, {} high".format(col, n_below, n_above))


# =========================================================================
#  FIX 3: IMPUTE MISSING VALUES
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 3: IMPUTING MISSING VALUES")
print("=" * 70)

# Check missing after cleanup
missing_before = df[numeric_features].isnull().sum()
print("\nMissing values per column (after impossible value removal):")
for col in numeric_features:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        print("  {}: {} ({:.2f}%)".format(col, n_miss, n_miss / len(df) * 100))

# Strategy: Median imputation (all columns have <9% missing)
# Median is robust to outliers and appropriate for clinical data
for col in numeric_features:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print("  -> {} imputed with median = {:.4f}".format(col, median_val))

# Fill binary ICD flags with 0 (NaN means no diagnosis = 0)
for col in binary_flags:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        df[col] = df[col].fillna(0).astype(int)
        print("  -> {} filled {} NaNs with 0".format(col, n_miss))

# Verify no missing values remain
all_feat_cols = numeric_features + binary_flags
total_missing = df[all_feat_cols].isnull().sum().sum()
print("\nTotal missing after imputation: {}".format(total_missing))


# =========================================================================
#  FIX 4: ENCODE GENDER
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 4: ENCODING GENDER")
print("=" * 70)

print("Before: {}".format(df['gender'].value_counts().to_dict()))

# Binary encoding: M=1, F=0
df['gender_encoded'] = (df['gender'] == 'M').astype(int)
print("After encoding: gender_encoded -> M=1, F=0")
print("  Value counts: {}".format(df['gender_encoded'].value_counts().to_dict()))

# Keep original gender column for reference, use gender_encoded for modeling


# =========================================================================
#  FIX 5: HANDLE SKEWED DISTRIBUTIONS (Log/Power Transform)
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 5: NORMALISING SKEWED FEATURES")
print("=" * 70)

# Identify features that need transformation (skewness > 1)
skewed_cols = []
for col in numeric_features:
    sk = df[col].skew()
    if abs(sk) > 1:
        skewed_cols.append((col, sk))
        print("  {}: skew = {:.3f}".format(col, sk))

# Apply log1p transform for positively skewed features
log_transform_cols = ['lactate', 'creatinine', 'icu_los_days', 'hosp_los_days']
for col in log_transform_cols:
    if col in df.columns:
        df[col + '_log'] = np.log1p(df[col])
        print("  -> Created {}_log (log1p transform)".format(col))

# Note: For model training, you can also use PowerTransformer (Yeo-Johnson)
# which handles both positive and negative skew
print("\n  [TIP] For remaining skewed features, use sklearn.preprocessing.PowerTransformer")
print("  during the ML pipeline (fits within the train-test split).")


# =========================================================================
#  FIX 6: HANDLE CLASS IMBALANCE (readmission)
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 6: CLASS IMBALANCE STRATEGIES (readmission)")
print("=" * 70)

print("\nCurrent class distribution:")
for label in label_cols:
    vc = df[label].value_counts()
    ratio = vc.min() / vc.max()
    print("  {}: {} | ratio = {:.3f}".format(label, vc.to_dict(), ratio))

print("\n  readmission has ~7:1 imbalance. Recommended approaches:")
print("  (These are applied during MODEL TRAINING, not data preprocessing)")
print("")
print("  APPROACH 1: class_weight='balanced' in the classifier")
print("    from sklearn.ensemble import RandomForestClassifier")
print("    clf = RandomForestClassifier(class_weight='balanced')")
print("")
print("  APPROACH 2: SMOTE oversampling (apply ONLY on training set)")
print("    from imblearn.over_sampling import SMOTE")
print("    smote = SMOTE(random_state=42)")
print("    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)")
print("")
print("  APPROACH 3: Stratified K-Fold cross-validation")
print("    from sklearn.model_selection import StratifiedKFold")
print("    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
print("")
print("  APPROACH 4: Custom sample_weight based on inverse frequency")
print("    from sklearn.utils.class_weight import compute_sample_weight")
print("    weights = compute_sample_weight('balanced', y_train)")


# =========================================================================
#  FIX 7: PATIENT-LEVEL TRAIN-TEST SPLIT (Prevent Leakage)
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 7: PATIENT-LEVEL TRAIN-TEST SPLIT")
print("=" * 70)

from sklearn.model_selection import GroupShuffleSplit

# Split by subject_id so no patient appears in both train and test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['subject_id']))

df_train = df.iloc[train_idx].copy()
df_test  = df.iloc[test_idx].copy()

# Verify no patient leakage
train_patients = set(df_train['subject_id'])
test_patients  = set(df_test['subject_id'])
overlap = train_patients & test_patients

print("  Train: {} rows, {} unique patients".format(len(df_train), len(train_patients)))
print("  Test:  {} rows, {} unique patients".format(len(df_test), len(test_patients)))
print("  Patient overlap: {} (should be 0)".format(len(overlap)))

if len(overlap) == 0:
    print("  [OK] No patient leakage between train and test sets!")
else:
    print("  [ERROR] Patient leakage detected!")


# =========================================================================
#  FIX 8: FEATURE SCALING (StandardScaler)
# =========================================================================
print("\n" + "=" * 70)
print("  FIX 8: FEATURE SCALING")
print("=" * 70)

# Define final feature columns for modeling
model_features = numeric_features + binary_flags + ['gender_encoded']
# Add log-transformed columns
for col in ['lactate_log', 'creatinine_log', 'icu_los_days_log', 'hosp_los_days_log']:
    if col in df.columns and col not in model_features:
        model_features.append(col)

# Fit scaler on TRAINING data only, transform both
scaler = StandardScaler()
df_train_scaled = df_train[model_features].copy()
df_test_scaled  = df_test[model_features].copy()

scaler.fit(df_train_scaled)
df_train_scaled = pd.DataFrame(
    scaler.transform(df_train_scaled),
    columns=model_features,
    index=df_train.index
)
df_test_scaled = pd.DataFrame(
    scaler.transform(df_test_scaled),
    columns=model_features,
    index=df_test.index
)

print("  StandardScaler fitted on training data and applied to both sets.")
print("  Scaled feature means (train): min={:.4f}, max={:.4f}".format(
    df_train_scaled.mean().min(), df_train_scaled.mean().max()))
print("  Scaled feature stds (train): min={:.4f}, max={:.4f}".format(
    df_train_scaled.std().min(), df_train_scaled.std().max()))


# =========================================================================
#  SAVE CLEANED DATASET
# =========================================================================
print("\n" + "=" * 70)
print("  SAVING CLEANED DATASET")
print("=" * 70)

# Save the full cleaned (pre-split) dataset
clean_path = os.path.join(OUT_DIR, 'fbi_data_cleaned.csv')
df.to_csv(clean_path, index=False)
print("  Cleaned full dataset: {}".format(clean_path))

# Save train/test splits
train_path = os.path.join(OUT_DIR, 'train_set.csv')
test_path  = os.path.join(OUT_DIR, 'test_set.csv')
df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)
print("  Training set: {}".format(train_path))
print("  Test set: {}".format(test_path))


# =========================================================================
#  VALIDATION: RE-RUN KEY CHECKS ON CLEANED DATA
# =========================================================================
print("\n" + "=" * 70)
print("  VALIDATION: POST-CLEANING CHECKS")
print("=" * 70)

# 1. No duplicates
n_dup = df.drop(columns=label_cols).duplicated().sum()
print("\n  1. Feature-duplicate rows: {} ({:.1f}%)".format(
    n_dup, n_dup / len(df) * 100))

# 2. No missing values
n_miss = df[all_feat_cols].isnull().sum().sum()
print("  2. Total missing values: {}".format(n_miss))

# 3. Clinical range check
violations_found = False
for col, (lo, hi) in clinical_caps.items():
    if col not in df.columns:
        continue
    if df[col].min() < lo or df[col].max() > hi:
        print("  3. [FAIL] {} still has out-of-range values".format(col))
        violations_found = True
if not violations_found:
    print("  3. All values within clinical ranges [OK]")

# 4. Class distribution
print("  4. Class distributions:")
for label in label_cols:
    vc = df[label].value_counts()
    print("     {}: {}".format(label, vc.to_dict()))

# 5. Patient-level split integrity
print("  5. Train-test patient overlap: {} [OK]".format(len(overlap)))

# 6. Dataset size
print("  6. Final dataset size: {} rows x {} columns".format(df.shape[0], df.shape[1]))

# 7. New features summary
print("  7. New features summary:")
for col in ['icu_los_days', 'hosp_los_days', 'charlson_comorbidity_index', 'sofa_score'] + binary_flags:
    print("     {}: mean={:.3f}, min={}, max={}".format(
        col, df[col].mean(), df[col].min(), df[col].max()))

print("\n" + "=" * 70)
print("  CLEANING COMPLETE!")
print("=" * 70)


# =========================================================================
#  GENERATE BEFORE/AFTER COMPARISON PLOTS
# =========================================================================
print("\nGenerating before/after comparison plots...")

# Reload original for comparison
df_orig = pd.read_csv(DATA_PATH)

# Plot: Distribution comparison for key features
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
plot_cols = ['heart_rate', 'resp_rate', 'spo2', 'sys_bp', 'dia_bp', 'fio2',
             'peep', 'tidal_volume', 'pao2', 'paco2', 'lactate', 'creatinine',
             'icu_los_days', 'hosp_los_days', 'charlson_comorbidity_index', 'sofa_score']

for i, col in enumerate(plot_cols):
    ax = axes[i]
    # Original (red, transparent)
    orig_vals = df_orig[col].dropna()
    # Clip original for better visualization
    q01 = orig_vals.quantile(0.01)
    q99 = orig_vals.quantile(0.99)
    orig_clipped = orig_vals[(orig_vals >= q01) & (orig_vals <= q99)]
    orig_clipped.hist(bins=50, ax=ax, alpha=0.4, color='red', label='Original', density=True)

    # Cleaned (blue)
    df[col].dropna().hist(bins=50, ax=ax, alpha=0.5, color='steelblue', label='Cleaned', density=True)

    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

plt.suptitle('Before vs After Cleaning (Distribution Comparison)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'before_after_distributions.png'), dpi=150)
plt.close(fig)
print("  -> Saved: before_after_distributions.png")

# Summary stats comparison
print("\n" + "-" * 50)
print("SUMMARY COMPARISON:")
print("-" * 50)
print("  Original: {:,} rows | Cleaned: {:,} rows".format(len(df_orig), len(df)))
print("  Rows removed: {:,} ({:.1f}%)".format(
    len(df_orig) - len(df), (len(df_orig) - len(df)) / len(df_orig) * 100))
print("  Missing values: {} -> {}".format(
    df_orig[numeric_features].isnull().sum().sum(), n_miss))
print("  Duplicate features: {:,} -> {:,}".format(
    df_orig.drop(columns=label_cols).duplicated().sum(), n_dup))
print("  Total columns: {} (including {} new features + {} log transforms)".format(
    len(df.columns), 8, 4))
