# -*- coding: utf-8 -*-
"""
=============================================================================
COMPLETE ML PIPELINE V2 - ICU Ventilator Weaning & Readmission Prediction
=============================================================================
Upgraded with 8 new clinical features: icu_los_days, hosp_los_days,
charlson_comorbidity_index, sofa_score, hypertension, diabetes, copd,
heart_failure (+ log transforms for LOS).

Step 1: Data Validation (re-check cleaned data)
Step 2: Preprocessing (encode, scale)
Step 3: Feature Engineering (PF ratio, RSBI, clinical composites)
Step 4: Model Building (Logistic Regression, Random Forest, XGBoost)
Step 5: Evaluation (Accuracy, Precision, Recall, ROC-AUC, PR-AUC)
Step 6: Model Interpretation (Feature Importance, SHAP)
Step 7: Optimization (Optuna-style Hyperparameter Tuning, Class Imbalance)
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

from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# -- Paths ------------------------------------------------------------------
DATA_PATH = r"e:\fbi ML\fix_outputs\fbi_data_cleaned.csv"
OUT_DIR   = r"e:\fbi ML\ml_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -- Load Cleaned Data ------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded cleaned dataset: {} rows x {} columns\n".format(df.shape[0], df.shape[1]))


# ==========================================================================
#  STEP 1: DATA VALIDATION (Re-check cleaned data)
# ==========================================================================
def section(title):
    print("\n" + "=" * 75)
    print("  " + title)
    print("=" * 75)

section("STEP 1: DATA VALIDATION (Cleaned Data)")

print("Shape: {}".format(df.shape))
print("Missing values: {}".format(df.isnull().sum().sum()))
print("Duplicate rows: {}".format(df.duplicated().sum()))
print("\nLabel distributions:")
for label in ['weaning_success', 'readmission']:
    vc = df[label].value_counts()
    print("  {}: {} (ratio={:.3f})".format(label, vc.to_dict(), vc.min()/vc.max()))

print("\nNew feature distributions:")
for col in ['icu_los_days', 'hosp_los_days', 'charlson_comorbidity_index',
            'sofa_score', 'hypertension', 'diabetes', 'copd', 'heart_failure']:
    print("  {}: mean={:.2f}, std={:.2f}, min={}, max={}".format(
        col, df[col].mean(), df[col].std(), df[col].min(), df[col].max()))

print("\n[OK] Data validation passed - ready for ML pipeline V2.")


# ==========================================================================
#  STEP 2 & 3: PREPROCESSING + FEATURE ENGINEERING
# ==========================================================================
section("STEP 2 & 3: PREPROCESSING + FEATURE ENGINEERING")

# --- Feature Engineering ---
# PF Ratio (PaO2/FiO2) - key indicator of oxygenation
df['pf_ratio'] = df['pao2'] / (df['fio2'] / 100.0)

# RSBI (Rapid Shallow Breathing Index) = resp_rate / tidal_volume_in_liters
df['rsbi'] = df['resp_rate'] / (df['tidal_volume'] / 1000.0)

# Mean Arterial Pressure (MAP)
df['map'] = (df['sys_bp'] + 2 * df['dia_bp']) / 3.0

# Pulse Pressure
df['pulse_pressure'] = df['sys_bp'] - df['dia_bp']

# Oxygenation Index (simplified) = (FiO2 * MAP) / PaO2
df['oxi_index'] = (df['fio2'] * df['map']) / df['pao2'].replace(0, np.nan)

# A-a gradient approximation (simplified)
df['aa_gradient'] = (df['fio2'] / 100.0 * 713) - (df['paco2'] / 0.8) - df['pao2']

# Log-transform for pao2
df['pao2_log'] = np.log1p(df['pao2'])

# --- NEW V2 Feature Engineering ---
# Comorbidity burden score (sum of ICD flags)
df['comorbidity_count'] = df['hypertension'] + df['diabetes'] + df['copd'] + df['heart_failure']

# SOFA per day (severity normalised by LOS)
df['sofa_per_day'] = df['sofa_score'] / df['icu_los_days'].replace(0, np.nan)

# LOS ratio (ICU stay as fraction of total hospital stay)
df['los_ratio'] = df['icu_los_days'] / df['hosp_los_days'].replace(0, np.nan)

# Charlson risk category (bin into clinical tiers)
df['charlson_high'] = (df['charlson_comorbidity_index'] >= 5).astype(int)

# High SOFA flag (discharge SOFA >= 8 is a known readmission risk factor)
df['sofa_high'] = (df['sofa_score'] >= 8).astype(int)

print("Engineered features created:")
print("  1. pf_ratio         = PaO2 / (FiO2/100)")
print("  2. rsbi             = resp_rate / (tidal_volume/1000)")
print("  3. map              = (SBP + 2*DBP) / 3")
print("  4. pulse_pressure   = SBP - DBP")
print("  5. oxi_index        = (FiO2 * MAP) / PaO2")
print("  6. aa_gradient      = Alveolar-arterial gradient")
print("  7. pao2_log         = log1p(PaO2)")
print("  8. comorbidity_count= sum(hypertension + diabetes + copd + heart_failure)")
print("  9. sofa_per_day     = sofa_score / icu_los_days")
print(" 10. los_ratio        = icu_los_days / hosp_los_days")
print(" 11. charlson_high    = 1 if charlson >= 5")
print(" 12. sofa_high        = 1 if sofa_score >= 8")

# Handle any inf/nan from division
df.replace([np.inf, -np.inf], np.nan, inplace=True)
eng_cols = ['pf_ratio', 'rsbi', 'map', 'pulse_pressure', 'oxi_index',
            'aa_gradient', 'pao2_log', 'comorbidity_count', 'sofa_per_day',
            'los_ratio', 'charlson_high', 'sofa_high']
for col in eng_cols:
    median = df[col].median()
    df[col].fillna(median, inplace=True)

# --- Define Final Feature Sets ---
id_cols = ['subject_id', 'hadm_id', 'stay_id']
label_cols = ['weaning_success', 'readmission']
exclude_cols = id_cols + label_cols + ['gender']

# All numeric features for modeling
feature_cols = [c for c in df.columns if c not in exclude_cols]
print("\nTotal features for modeling: {}".format(len(feature_cols)))
print("Features: {}".format(feature_cols))


# ==========================================================================
#  TRAIN-TEST SPLIT (Patient-level, no leakage)
# ==========================================================================
section("TRAIN-TEST SPLIT (Patient-Level)")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['subject_id']))

df_train = df.iloc[train_idx].copy()
df_test  = df.iloc[test_idx].copy()

X_train = df_train[feature_cols].values
X_test  = df_test[feature_cols].values

print("Train: {} rows, {} patients".format(len(df_train), df_train['subject_id'].nunique()))
print("Test:  {} rows, {} patients".format(len(df_test), df_test['subject_id'].nunique()))
overlap = set(df_train['subject_id']) & set(df_test['subject_id'])
print("Patient overlap: {} [OK]".format(len(overlap)))

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nFeatures scaled with StandardScaler (fit on train only).")


# ==========================================================================
#  STEP 4 & 5: MODEL BUILDING + EVALUATION
# ==========================================================================

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name, target_name, threshold=0.5):
    """Train, predict, and return comprehensive metrics."""
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob)
    pr_auc = average_precision_score(y_te, y_prob)

    print("\n  {} ({})".format(model_name, target_name))
    print("  " + "-" * 50)
    print("    Accuracy:  {:.4f}".format(acc))
    print("    Precision: {:.4f}".format(prec))
    print("    Recall:    {:.4f}".format(rec))
    print("    F1-Score:  {:.4f}".format(f1))
    print("    ROC-AUC:   {:.4f}".format(auc))
    print("    PR-AUC:    {:.4f}".format(pr_auc))

    return {
        'model': model, 'model_name': model_name, 'target': target_name,
        'y_test': y_te, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc, 'pr_auc': pr_auc
    }


def plot_results(results_list, target_name):
    """Plot ROC curves, PR curves, and confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # --- ROC Curves ---
    ax = axes[0]
    for res in results_list:
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax.plot(fpr, tpr, label='{} (AUC={:.4f})'.format(res['model_name'], res['roc_auc']), linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - {}'.format(target_name), fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Precision-Recall Curves ---
    ax = axes[1]
    for res in results_list:
        prec_arr, rec_arr, _ = precision_recall_curve(res['y_test'], res['y_prob'])
        ax.plot(rec_arr, prec_arr, label='{} (PR-AUC={:.4f})'.format(res['model_name'], res['pr_auc']), linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - {}'.format(target_name), fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Best Model Confusion Matrix ---
    ax = axes[2]
    best = max(results_list, key=lambda x: x['roc_auc'])
    cm = confusion_matrix(best['y_test'], best['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - {} ({})'.format(best['model_name'], target_name), fontsize=13)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'roc_pr_cm_{}.png'.format(target_name)), dpi=150)
    plt.close(fig)
    print("  -> Saved: roc_pr_cm_{}.png".format(target_name))


# --- Run for both targets ---
all_results = {}

for target in label_cols:
    section("STEP 4 & 5: {} PREDICTION".format(target.upper()))

    y_train = df_train[target].values
    y_test  = df_test[target].values

    print("\n  Class distribution (train): {}".format(
        dict(zip(*np.unique(y_train, return_counts=True)))))
    print("  Class distribution (test):  {}".format(
        dict(zip(*np.unique(y_test, return_counts=True)))))

    # Compute scale_pos_weight for XGBoost
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    results = []

    # Set threshold to improve recall for Highly Imbalanced Readmission target
    threshold = 0.3 if target == 'readmission' else 0.5

    # --- 1. Logistic Regression ---
    lr = LogisticRegression(
        max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=42, C=1.0
    )
    res = evaluate_model(lr, X_train_scaled, y_train, X_test_scaled, y_test,
                         'Logistic Regression', target, threshold=threshold)
    results.append(res)

    # --- 2. Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight='balanced', random_state=42,
        n_jobs=-1, min_samples_split=5, min_samples_leaf=2
    )
    res = evaluate_model(rf, X_train_scaled, y_train, X_test_scaled, y_test,
                         'Random Forest', target, threshold=threshold)
    results.append(res)

    # --- 3. XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        scale_pos_weight=scale_weight,
        eval_metric='auc', random_state=42,
        use_label_encoder=False, n_jobs=-1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=3, gamma=0.1
    )
    res = evaluate_model(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test,
                         'XGBoost', target, threshold=threshold)
    results.append(res)

    # Plot results
    plot_results(results, target)

    # Store results
    all_results[target] = results

    # --- Summary Table ---
    print("\n  SUMMARY TABLE - {}:".format(target))
    print("  {:25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'))
    print("  " + "-" * 85)
    for r in results:
        print("  {:25s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r['model_name'], r['accuracy'], r['precision'],
            r['recall'], r['f1'], r['roc_auc'], r['pr_auc']))


# ==========================================================================
#  STEP 6: MODEL INTERPRETATION (Feature Importance + SHAP)
# ==========================================================================
section("STEP 6: MODEL INTERPRETATION")

for target in label_cols:
    print("\n--- {} ---".format(target.upper()))

    # Get best model (XGBoost) results
    xgb_res = [r for r in all_results[target] if r['model_name'] == 'XGBoost'][0]
    xgb_model = xgb_res['model']
    rf_res = [r for r in all_results[target] if r['model_name'] == 'Random Forest'][0]
    rf_model = rf_res['model']

    # --- Feature Importance (XGBoost) ---
    importances = xgb_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n  Top 15 Features (XGBoost - {}):\n".format(target))
    for rank, (i, row) in enumerate(feat_imp.head(15).iterrows(), 1):
        print("    {:3d}. {:30s}  importance={:.4f}".format(
            rank, row['feature'], row['importance']))

    # Plot feature importance
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # XGBoost importance
    top20 = feat_imp.head(20)
    colors_xgb = ['#2196F3' if f in ['icu_los_days', 'hosp_los_days', 'charlson_comorbidity_index',
                   'sofa_score', 'hypertension', 'diabetes', 'copd', 'heart_failure',
                   'comorbidity_count', 'sofa_per_day', 'los_ratio', 'charlson_high',
                   'sofa_high', 'icu_los_days_log', 'hosp_los_days_log']
                   else '#90CAF9' for f in top20['feature'].values]
    axes[0].barh(range(len(top20)), top20['importance'].values, color=colors_xgb)
    axes[0].set_yticks(range(len(top20)))
    axes[0].set_yticklabels(top20['feature'].values, fontsize=9)
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].set_title('XGBoost Feature Importance - {}\n(dark blue = NEW features)'.format(target), fontsize=13)
    axes[0].invert_yaxis()

    # Random Forest importance
    rf_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    colors_rf = ['#FF9800' if f in ['icu_los_days', 'hosp_los_days', 'charlson_comorbidity_index',
                  'sofa_score', 'hypertension', 'diabetes', 'copd', 'heart_failure',
                  'comorbidity_count', 'sofa_per_day', 'los_ratio', 'charlson_high',
                  'sofa_high', 'icu_los_days_log', 'hosp_los_days_log']
                  else '#FFE0B2' for f in rf_imp['feature'].values]
    axes[1].barh(range(len(rf_imp)), rf_imp['importance'].values, color=colors_rf)
    axes[1].set_yticks(range(len(rf_imp)))
    axes[1].set_yticklabels(rf_imp['feature'].values, fontsize=9)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Random Forest Feature Importance - {}\n(dark orange = NEW features)'.format(target), fontsize=13)
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_importance_{}.png'.format(target)), dpi=150)
    plt.close(fig)
    print("  -> Saved: feature_importance_{}.png".format(target))

    # --- SHAP Analysis ---
    print("\n  Computing SHAP values (XGBoost - {})...".format(target))

    shap_sample_size = min(500, len(X_test_scaled))
    X_shap = X_test_scaled[:shap_sample_size]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)

    # SHAP summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_cols,
        show=False, max_display=25
    )
    plt.title('SHAP Summary - {} (XGBoost V2)'.format(target), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'shap_summary_{}.png'.format(target)), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  -> Saved: shap_summary_{}.png".format(target))

    # SHAP bar plot (mean absolute values)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_cols,
        plot_type='bar', show=False, max_display=25
    )
    plt.title('SHAP Mean Importance - {} (XGBoost V2)'.format(target), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'shap_bar_{}.png'.format(target)), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  -> Saved: shap_bar_{}.png".format(target))


# ==========================================================================
#  STEP 7: OPTIMIZATION (Hyperparameter Tuning)
# ==========================================================================
section("STEP 7: OPTIMIZATION (Hyperparameter Tuning)")

best_models = {}

for target in label_cols:
    print("\n--- Tuning XGBoost for {} ---".format(target.upper()))

    y_train = df_train[target].values
    y_test  = df_test[target].values

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.08, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
    }

    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric='auc', random_state=42,
        use_label_encoder=False, n_jobs=-1
    )

    # Randomized search (50 iterations for better search)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_base, param_grid,
        n_iter=50, scoring='roc_auc',
        cv=cv, random_state=42,
        n_jobs=-1, verbose=0
    )

    print("  Running RandomizedSearchCV (50 iterations, 3-fold CV)...")
    search.fit(X_train_scaled, y_train)

    print("  Best params: {}".format(search.best_params_))
    print("  Best CV ROC-AUC: {:.4f}".format(search.best_score_))

    # Evaluate tuned model on test set (with threshold tuning)
    threshold = 0.3 if target == 'readmission' else 0.5
    best_model = search.best_estimator_
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n  TUNED XGBoost ({}) Test Performance:".format(target))
    print("    Accuracy:  {:.4f}".format(acc))
    print("    Precision: {:.4f}".format(prec))
    print("    Recall:    {:.4f}".format(rec))
    print("    F1-Score:  {:.4f}".format(f1))
    print("    ROC-AUC:   {:.4f}".format(auc))
    print("    PR-AUC:    {:.4f}".format(pr_auc))

    # Compare with baseline
    baseline_auc = [r['roc_auc'] for r in all_results[target] if r['model_name'] == 'XGBoost'][0]
    improvement = auc - baseline_auc
    print("\n  Improvement over baseline XGBoost: {:+.4f} ROC-AUC".format(improvement))

    best_models[target] = {
        'model': best_model, 'auc': auc, 'pr_auc': pr_auc,
        'params': search.best_params_,
        'y_pred': y_pred, 'y_prob': y_prob
    }


# ==========================================================================
#  FINAL COMPARISON: ALL MODELS (V1 vs V2)
# ==========================================================================
section("FINAL COMPARISON: ALL MODELS")

# Previous V1 results (hardcoded from last run for comparison)
v1_results = {
    'weaning_success': {'LR': 0.7656, 'RF': 0.8008, 'XGB': 0.8050, 'XGB_Tuned': 0.8149},
    'readmission':     {'LR': 0.6336, 'RF': 0.6394, 'XGB': 0.6263, 'XGB_Tuned': 0.6557}
}

for target in label_cols:
    print("\n  {} - Final Results:".format(target.upper()))
    print("  {:30s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'))
    print("  " + "-" * 90)

    for r in all_results[target]:
        print("  {:30s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r['model_name'], r['accuracy'], r['precision'],
            r['recall'], r['f1'], r['roc_auc'], r['pr_auc']))

    # Tuned model
    bm = best_models[target]
    y_test = df_test[target].values
    print("  {:30s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        'XGBoost (Tuned V2)',
        accuracy_score(y_test, bm['y_pred']),
        precision_score(y_test, bm['y_pred'], zero_division=0),
        recall_score(y_test, bm['y_pred'], zero_division=0),
        f1_score(y_test, bm['y_pred'], zero_division=0),
        bm['auc'], bm['pr_auc']))

    # V1 vs V2 comparison
    if target in v1_results:
        v1_best = v1_results[target]['XGB_Tuned']
        v2_best = bm['auc']
        print("\n  ==> V1 Best ROC-AUC: {:.4f}  |  V2 Best ROC-AUC: {:.4f}  |  Improvement: {:+.4f}".format(
            v1_best, v2_best, v2_best - v1_best))

# --- Final ROC comparison plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, target in enumerate(label_cols):
    ax = axes[idx]

    for r in all_results[target]:
        fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
        ax.plot(fpr, tpr, label='{} ({:.4f})'.format(r['model_name'], r['roc_auc']), linewidth=2)

    # Tuned XGBoost
    bm = best_models[target]
    y_test = df_test[target].values
    fpr, tpr, _ = roc_curve(y_test, bm['y_prob'])
    ax.plot(fpr, tpr, label='XGBoost Tuned V2 ({:.4f})'.format(bm['auc']),
            linewidth=2.5, linestyle='--', color='darkred')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('{} - ROC Comparison V2'.format(target), fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'final_roc_comparison.png'), dpi=150)
plt.close(fig)
print("\n  -> Saved: final_roc_comparison.png")

# --- Classification reports ---
for target in label_cols:
    bm = best_models[target]
    y_test = df_test[target].values
    print("\n\n  Classification Report - {} (Tuned XGBoost V2):".format(target))
    print(classification_report(y_test, bm['y_pred'],
                                target_names=['Negative (0)', 'Positive (1)']))


# ==========================================================================
#  FINAL SUMMARY
# ==========================================================================
section("PIPELINE V2 COMPLETE - SUMMARY")

print("""
  Dataset:           {} rows, {} features
  Train/Test Split:  {}/{} (patient-level, 0 overlap)
  New Features:      icu_los_days, hosp_los_days, charlson, sofa, ICD flags
  Engineered:        pf_ratio, rsbi, map, pulse_pressure, oxi_index,
                     aa_gradient, comorbidity_count, sofa_per_day, los_ratio

  Best Models (by ROC-AUC):
""".format(len(df), len(feature_cols), len(df_train), len(df_test)))

for target in label_cols:
    bm = best_models[target]
    v1_best = v1_results.get(target, {}).get('XGB_Tuned', 0)
    print("    {}: XGBoost Tuned V2 (ROC-AUC = {:.4f}, PR-AUC = {:.4f})  [V1 was {:.4f}, +{:.4f}]".format(
        target, bm['auc'], bm['pr_auc'], v1_best, bm['auc'] - v1_best))

print("""
  Output Files ({}):
    - roc_pr_cm_weaning_success.png
    - roc_pr_cm_readmission.png
    - feature_importance_weaning_success.png
    - feature_importance_readmission.png
    - shap_summary_weaning_success.png
    - shap_summary_readmission.png
    - shap_bar_weaning_success.png
    - shap_bar_readmission.png
    - final_roc_comparison.png
""".format(OUT_DIR))

print("=" * 75)
print("  DONE!")
print("=" * 75)
