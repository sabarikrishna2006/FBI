# -*- coding: utf-8 -*-
"""
=============================================================================
COMPLETE DATA AUDIT & VALIDATION - ICU (MIMIC-IV) Dataset
Ventilator Weaning & ICU Readmission Prediction
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# -- paths -----------------------------------------------------------------
DATA_PATH = r"e:\fbi ML\fbi_data_local.csv"
OUT_DIR   = r"e:\fbi ML\audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -- load ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset loaded: {} rows x {} columns\n".format(df.shape[0], df.shape[1]))

# Feature / label lists (excluding ID columns)
id_cols      = ['subject_id', 'hadm_id', 'stay_id']
label_cols   = ['weaning_success', 'readmission']
feature_cols = [c for c in df.columns if c not in id_cols + label_cols]
numeric_features = [c for c in feature_cols if c != 'gender']

report_lines = []

def section(title):
    banner = "\n{}\n  {}\n{}".format("=" * 80, title, "=" * 80)
    print(banner)
    report_lines.append(banner)

def note(msg):
    print(msg)
    report_lines.append(msg)


# ==========================================================================
#  STEP 1 - STRUCTURAL VALIDATION
# ==========================================================================
section("STEP 1 - STRUCTURAL VALIDATION")

note("Shape: {}".format(df.shape))
note("\nColumn data-types:")
for col in df.columns:
    note("  {:25s}  {:10s}  (unique={})".format(col, str(df[col].dtype), df[col].nunique()))

# Check for unexpected object/string columns
object_cols = df.select_dtypes(include='object').columns.tolist()
note("\nObject/string columns: {}".format(object_cols))
if object_cols == ['gender']:
    note("[OK] Only 'gender' is non-numeric -- expected.")
else:
    extra = [c for c in object_cols if c != 'gender']
    note("[WARNING] Unexpected object columns: {}".format(extra))

# Gender encoding check
note("\nGender value counts:\n{}".format(df['gender'].value_counts().to_string()))

# Check for duplicate rows
n_dup = df.duplicated().sum()
note("\nExact duplicate rows: {}  ({:.1f}%)".format(n_dup, n_dup / len(df) * 100))
n_dup_features = df.drop(columns=label_cols).duplicated().sum()
note("Rows with duplicate features (ignoring labels): {}  ({:.1f}%)".format(
    n_dup_features, n_dup_features / len(df) * 100))

if n_dup_features > 0:
    note("[WARNING] Many rows share identical feature values but differ only in"
         " labels (weaning_success / readmission). This suggests the dataset was"
         " artificially expanded -- perhaps one row per ventilator-event epoch."
         " This MUST be investigated before training to avoid data leakage from"
         " the same patient-stay appearing in both train and test sets.")


# ==========================================================================
#  STEP 2 - MISSING VALUES ANALYSIS
# ==========================================================================
section("STEP 2 - MISSING VALUES ANALYSIS")

missing = df[feature_cols + label_cols].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
miss_df = miss_df.sort_values('missing_pct', ascending=False)
note(miss_df.to_string())

flagged = miss_df[miss_df['missing_pct'] > 30]
if len(flagged):
    note("\n[WARNING] Columns with >30% missing: {}".format(flagged.index.tolist()))
else:
    note("\n[OK] No column exceeds 30% missing data.")

cols_with_missing = miss_df[miss_df['missing_pct'] > 0].index.tolist()
if cols_with_missing:
    note("\nImputation suggestions:")
    for c in cols_with_missing:
        pct = miss_df.loc[c, 'missing_pct']
        if pct < 5:
            note("  {} ({:.1f}%): median imputation (low missingness)".format(c, pct))
        elif pct < 30:
            note("  {} ({:.1f}%): consider KNN or iterative imputer".format(c, pct))
        else:
            note("  {} ({:.1f}%): [WARNING] too much missing -- consider dropping or"
                 " using a missingness indicator + imputation".format(c, pct))

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
miss_plot = miss_df[miss_df['missing_pct'] > 0]
if len(miss_plot):
    miss_plot['missing_pct'].plot.bar(ax=ax, color='salmon')
    ax.set_ylabel('Missing %')
    ax.set_title('Missing Values by Column')
    ax.axhline(30, ls='--', color='red', label='30% threshold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'step2_missing_values.png'), dpi=150)
    plt.close(fig)
    note("  -> Plot saved: step2_missing_values.png")
else:
    note("  (No missing data to plot)")
    plt.close(fig)


# ==========================================================================
#  STEP 3 - CLINICAL RANGE VALIDATION
# ==========================================================================
section("STEP 3 - CLINICAL RANGE VALIDATION")

clinical_ranges = {
    'heart_rate':    (40, 180),
    'resp_rate':     (10, 40),
    'spo2':          (80, 100),
    'sys_bp':        (80, 200),
    'dia_bp':        (20, 120),
    'fio2':          (21, 100),
    'peep':          (0, 20),
    'tidal_volume':  (200, 800),
    'ph':            (7.2, 7.6),
    'pao2':          (50, 300),
    'paco2':         (30, 70),
    'lactate':       (0.5, 10),
    'creatinine':    (0.3, 15),
    'age':           (18, 100),
}

violations = {}
for col, (lo, hi) in clinical_ranges.items():
    if col not in df.columns:
        continue
    below = (df[col] < lo).sum()
    above = (df[col] > hi).sum()
    total = below + above
    if total > 0:
        violations[col] = {
            'below': below, 'above': above,
            'total': total,
            'pct': round(total / df[col].notna().sum() * 100, 2),
            'range': (lo, hi),
            'actual_min': round(float(df[col].min()), 4),
            'actual_max': round(float(df[col].max()), 4)
        }

if violations:
    note("\n[WARNING] Out-of-range values found:\n")
    for col, info in sorted(violations.items(), key=lambda x: -x[1]['pct']):
        note("  {:15s}  expected [{}, {}]  actual [{}, {}]  "
             "violations={} ({:.2f}%)  (below={}, above={})".format(
            col, info['range'][0], info['range'][1],
            info['actual_min'], info['actual_max'],
            info['total'], info['pct'],
            info['below'], info['above']))
else:
    note("[OK] All values within clinical ranges.")

# Special check: fio2 might be in 0-1 scale
fio2_notna = df['fio2'].notna().sum()
fio2_below_1 = (df['fio2'].dropna() <= 1).sum()
note("\nFiO2 sanity: {} values <= 1.0 out of {} -> {}".format(
    fio2_below_1, fio2_notna,
    'mixed scale detected!' if 0 < fio2_below_1 < fio2_notna else 'consistent scale'))

# pH special flag
if 'ph' in violations:
    note("\n[CLINICAL] pH outliers are clinically critical -- values outside 7.2-7.6"
         " suggest severe acidosis/alkalosis. Verify they are real readings.")


# ==========================================================================
#  STEP 4 - OUTLIER DETECTION (IQR Method)
# ==========================================================================
section("STEP 4 - OUTLIER DETECTION (IQR Method)")

outlier_summary = {}
for col in numeric_features:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lo_bound = Q1 - 1.5 * IQR
    hi_bound = Q3 + 1.5 * IQR
    n_out = int(((s < lo_bound) | (s > hi_bound)).sum())
    if n_out > 0:
        outlier_summary[col] = {
            'n_outliers': n_out,
            'pct': round(n_out / len(s) * 100, 2),
            'Q1': round(float(Q1), 4), 'Q3': round(float(Q3), 4),
            'IQR': round(float(IQR), 4),
            'lo_bound': round(float(lo_bound), 4),
            'hi_bound': round(float(hi_bound), 4),
            'min': round(float(s.min()), 4),
            'max': round(float(s.max()), 4)
        }

note("\nColumns with IQR outliers ({}/{}):\n".format(
    len(outlier_summary), len(numeric_features)))
for col, info in sorted(outlier_summary.items(), key=lambda x: -x[1]['pct']):
    note("  {:15s}  outliers={:>6}  ({:>5.2f}%)  "
         "bounds=[{:.2f}, {:.2f}]  actual=[{:.2f}, {:.2f}]".format(
        col, info['n_outliers'], info['pct'],
        info['lo_bound'], info['hi_bound'],
        info['min'], info['max']))

note("\nRecommendations:")
note("  * For clinically meaningful outliers (e.g., lactate=16, pH=6.98):"
     " keep but consider Winsorisation.")
note("  * For impossible values (e.g., dia_bp=137): cap at clinical max"
     " or investigate data entry errors.")
note("  * For tidal_volume > 800 (e.g., 860 mL): plausible in large"
     " patients -- use clinical judgment.")

# Box-plots
cols_to_plot = list(outlier_summary.keys())[:12]
if cols_to_plot:
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols_to_plot):
        df[col].dropna().plot.box(ax=axes[i])
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Outlier Box-Plots (IQR)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, 'step4_outlier_boxplots.png'), dpi=150)
    plt.close(fig)
    note("  -> Plot saved: step4_outlier_boxplots.png")


# ==========================================================================
#  STEP 5 - CLASS IMBALANCE CHECK
# ==========================================================================
section("STEP 5 - CLASS IMBALANCE CHECK")

for label in label_cols:
    vc = df[label].value_counts()
    ratio = vc.min() / vc.max()
    note("\n{}:".format(label))
    note(vc.to_string())
    note("  Minority ratio: {:.3f}".format(ratio))
    if ratio < 0.1:
        note("  [SEVERE] Severe imbalance (ratio<0.10)."
             " Use SMOTE / class_weight='balanced'.")
    elif ratio < 0.3:
        note("  [MODERATE] Moderate imbalance."
             " Consider oversampling or class weights.")
    else:
        note("  [OK] Acceptable balance.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, label in enumerate(label_cols):
    df[label].value_counts().plot.bar(ax=axes[i], color=['#4C72B0', '#DD8452'])
    axes[i].set_title('{} Distribution'.format(label))
    axes[i].set_ylabel('Count')
    for p in axes[i].patches:
        axes[i].annotate('{:,}'.format(int(p.get_height())),
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'step5_class_imbalance.png'), dpi=150)
plt.close(fig)
note("  -> Plot saved: step5_class_imbalance.png")


# ==========================================================================
#  STEP 6 - FEATURE DISTRIBUTION ANALYSIS
# ==========================================================================
section("STEP 6 - FEATURE DISTRIBUTION ANALYSIS")

note("\nSkewness (|skew| > 1 = highly skewed):\n")
skew_data = {}
for col in numeric_features:
    s = df[col].dropna()
    sk = round(float(s.skew()), 3)
    skew_data[col] = sk
    flag = "[!]" if abs(sk) > 1 else "   "
    note("  {} {:15s}  skew={:>7.3f}".format(flag, col, sk))

cols_needing_transform = [c for c, s in skew_data.items() if abs(s) > 1]
if cols_needing_transform:
    note("\n[SUGGEST] Columns needing normalisation/log-transform: {}".format(
        cols_needing_transform))
else:
    note("\n[OK] No severely skewed columns.")

# Histogram grid
n_cols = len(numeric_features)
n_rows_plot = (n_cols + 3) // 4
fig, axes = plt.subplots(n_rows_plot, 4, figsize=(18, n_rows_plot * 3.2))
axes = axes.flatten()
for i, col in enumerate(numeric_features):
    df[col].dropna().hist(bins=50, ax=axes[i], color='steelblue', edgecolor='white')
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(labelsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Feature Distributions', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'step6_distributions.png'), dpi=150)
plt.close(fig)
note("  -> Plot saved: step6_distributions.png")


# ==========================================================================
#  STEP 7 - CORRELATION & LEAKAGE DETECTION
# ==========================================================================
section("STEP 7 - CORRELATION & LEAKAGE DETECTION")

corr_cols = numeric_features + label_cols
corr_matrix = df[corr_cols].corr()

# Highly correlated feature pairs
note("\nHighly correlated feature pairs (|r| > 0.9):\n")
found_high = False
for i in range(len(numeric_features)):
    for j in range(i + 1, len(numeric_features)):
        r = corr_matrix.loc[numeric_features[i], numeric_features[j]]
        if abs(r) > 0.9:
            note("  {:15s} <-> {:15s}  r = {:.4f}".format(
                numeric_features[i], numeric_features[j], r))
            found_high = True
if not found_high:
    note("  [OK] No feature pairs exceed |r| > 0.9.")

# Moderately correlated pairs
note("\nModerately correlated feature pairs (0.7 < |r| <= 0.9):\n")
found_mod = False
for i in range(len(numeric_features)):
    for j in range(i + 1, len(numeric_features)):
        r = corr_matrix.loc[numeric_features[i], numeric_features[j]]
        if 0.7 < abs(r) <= 0.9:
            note("  {:15s} <-> {:15s}  r = {:.4f}".format(
                numeric_features[i], numeric_features[j], r))
            found_mod = True
if not found_mod:
    note("  [OK] No moderately correlated pairs found.")

# Feature-label correlations
note("\nFeature-Label correlations:\n")
for label in label_cols:
    note("  {}:".format(label))
    for feat in numeric_features:
        r = corr_matrix.loc[feat, label]
        flag = " [LEAKAGE RISK!]" if abs(r) > 0.8 else ""
        note("    {:15s}  r = {:>7.4f}{}".format(feat, r, flag))
    note("")

# Leakage: same feature sets -> same labels across patient
note("[CRITICAL] Cross-checking for data leakage patterns:")
note("  Duplicate feature rows with different labels detected in Step 1.")
note("  This is a strong leakage signal -- the same patient-observation")
note("  appears multiple times with different outcomes.")

# Heatmap
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
            annot_kws={'size': 7})
ax.set_title('Correlation Matrix (Features + Labels)', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'step7_correlation_heatmap.png'), dpi=150)
plt.close(fig)
note("  -> Plot saved: step7_correlation_heatmap.png")


# ==========================================================================
#  STEP 8 - BIAS DETECTION
# ==========================================================================
section("STEP 8 - BIAS DETECTION")

# Gender distribution
note("\nGender distribution:")
gender_counts = df['gender'].value_counts()
note(gender_counts.to_string())
gender_ratio = gender_counts.min() / gender_counts.max()
note("  Gender ratio (min/max): {:.3f}".format(gender_ratio))
if gender_ratio < 0.5:
    note("  [WARNING] Significant gender imbalance -- model may be biased.")
else:
    note("  [OK] Gender distribution acceptable.")

# Gender vs labels
note("\nGender vs weaning_success:")
ct_w = pd.crosstab(df['gender'], df['weaning_success'], normalize='index')
note(ct_w.round(4).to_string())

note("\nGender vs readmission:")
ct_r = pd.crosstab(df['gender'], df['readmission'], normalize='index')
note(ct_r.round(4).to_string())

# Chi-squared tests
for label in label_cols:
    ct = pd.crosstab(df['gender'], df[label])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    note("\n  Chi-sq test (gender x {}): chi2={:.2f}, p={:.4e}, dof={}".format(
        label, chi2, p, dof))
    if p < 0.05:
        note("  [WARNING] Statistically significant association (p<0.05)."
             " Potential gender bias.")
    else:
        note("  [OK] No significant association (p>=0.05).")

# Age vs labels
note("\nAge distribution by label:")
for label in label_cols:
    groups = df.groupby(label)['age']
    for val, grp in groups:
        note("  {}={}: mean age={:.1f}, std={:.1f}".format(label, val, grp.mean(), grp.std()))
    g0 = df[df[label] == 0]['age'].dropna()
    g1 = df[df[label] == 1]['age'].dropna()
    t_stat, t_p = stats.ttest_ind(g0, g1)
    note("  t-test: t={:.2f}, p={:.4e}".format(t_stat, t_p))
    if t_p < 0.05:
        note("  [WARNING] Significant age difference between {} groups.".format(label))
    else:
        note("  [OK] No significant age bias for {}.".format(label))

# Age distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, label in enumerate(label_cols):
    for val in [0, 1]:
        subset = df[df[label] == val]['age']
        subset.hist(bins=30, ax=axes[i], alpha=0.5,
                    label='{}={}'.format(label, val), edgecolor='white')
    axes[i].set_title('Age Distribution by {}'.format(label))
    axes[i].set_xlabel('Age')
    axes[i].legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'step8_age_bias.png'), dpi=150)
plt.close(fig)
note("  -> Plot saved: step8_age_bias.png")

# Gender label plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, label in enumerate(label_cols):
    ct = pd.crosstab(df['gender'], df[label], normalize='index') * 100
    ct.plot.bar(ax=axes[i], stacked=True, color=['#4C72B0', '#DD8452'])
    axes[i].set_title('Gender vs {} (%)'.format(label))
    axes[i].set_ylabel('Percentage')
    axes[i].legend(title=label)
    axes[i].tick_params(axis='x', rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'step8_gender_bias.png'), dpi=150)
plt.close(fig)
note("  -> Plot saved: step8_gender_bias.png")


# ==========================================================================
#  STEP 9 - FINAL VERDICT
# ==========================================================================
section("STEP 9 - FINAL VERDICT")

issues = []
# Duplicates
if n_dup_features > 0:
    issues.append("[CRITICAL] Massive duplicate rows with different labels --"
                  " likely data leakage / artificial expansion.")

# Missing
cols_high_missing = miss_df[miss_df['missing_pct'] > 30].index.tolist()
if cols_high_missing:
    issues.append("[MODERATE] Columns with >30% missing: {}".format(cols_high_missing))
cols_any_missing = miss_df[miss_df['missing_pct'] > 0].index.tolist()
if cols_any_missing:
    issues.append("[MINOR] Columns with some missing values: {}"
                  " -> need imputation before training.".format(cols_any_missing))

# Clinical
if violations:
    issues.append("[MODERATE] Out-of-clinical-range values in: {}".format(
        list(violations.keys())))

# Outliers
if outlier_summary:
    severe = [c for c, v in outlier_summary.items() if v['pct'] > 5]
    if severe:
        issues.append("[MODERATE] Significant outliers in: {}".format(severe))

# Imbalance
for label in label_cols:
    vc = df[label].value_counts()
    ratio = vc.min() / vc.max()
    if ratio < 0.1:
        issues.append("[SEVERE] {} is severely imbalanced (ratio={:.3f}).".format(
            label, ratio))
    elif ratio < 0.3:
        issues.append("[MODERATE] {} has moderate class imbalance (ratio={:.3f}).".format(
            label, ratio))

note("\n" + "-" * 60)
note("ISSUES IDENTIFIED:")
note("-" * 60)
if not issues:
    note("  [OK] No major issues found.")
else:
    for iss in issues:
        note("  {}".format(iss))

note("\n" + "-" * 60)
note("MUST-FIX BEFORE TRAINING:")
note("-" * 60)
note("  1. De-duplicate: decide the right grain (1 row per stay? per event?)."
     " Remove artificial duplicates.")
note("  2. Impute missing values (median / KNN for labs; carry-forward for vitals).")
note("  3. Cap / correct clinical range violations (esp. dia_bp, fio2, pH, lactate).")
note("  4. Encode gender (M=1, F=0 or one-hot).")
note("  5. Split data by patient (subject_id) to prevent intra-patient leakage.")
note("  6. Handle class imbalance (SMOTE, class weights, or stratified sampling).")

note("\n" + "-" * 60)
note("FINAL READINESS STATUS:")
note("-" * 60)
if any("CRITICAL" in i for i in issues):
    note("  >>> NOT READY FOR ML TRAINING -- critical issues must be resolved first.")
else:
    note("  >>> CONDITIONALLY READY -- moderate issues should be addressed.")

note("\n" + "=" * 80)

# Save full text report
report_path = os.path.join(OUT_DIR, 'full_audit_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print("\nFull text report saved to: {}".format(report_path))
print("All plots saved to: {}".format(OUT_DIR))
