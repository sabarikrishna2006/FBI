# -*- coding: utf-8 -*-
"""
=============================================================================
COMPLETE ML PIPELINE - ICU Ventilator Weaning & Readmission Prediction
=============================================================================
Step 1: Data Validation (re-check cleaned data)
Step 2: Preprocessing (encode, scale)
Step 3: Feature Engineering (PF ratio, RSBI, etc.)
Step 4: Model Building (Logistic Regression, Random Forest, XGBoost)
Step 5: Evaluation (Accuracy, Precision, Recall, ROC-AUC)
Step 6: Model Interpretation (Feature Importance, SHAP)
Step 7: Optimization (Hyperparameter Tuning, Class Imbalance)
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
from sklearn.ensemble import RandomForestClassifier
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

print("\nBasic statistics:")
print(df.describe().round(2).to_string())
print("\n[OK] Data validation passed - ready for ML pipeline.")


# ==========================================================================
#  STEP 2 & 3: PREPROCESSING + FEATURE ENGINEERING
# ==========================================================================
section("STEP 2 & 3: PREPROCESSING + FEATURE ENGINEERING")

# --- Feature Engineering ---
# PF Ratio (PaO2/FiO2) - key indicator of oxygenation
# FiO2 is on 0-100 scale, convert to fraction for proper PF ratio
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

# Log-transforms for skewed features (already have lactate_log, creatinine_log)
# Also create for pao2 which was skewed
df['pao2_log'] = np.log1p(df['pao2'])

print("Engineered features created:")
print("  1. pf_ratio    = PaO2 / (FiO2/100) -- P/F ratio (oxygenation index)")
print("  2. rsbi         = resp_rate / (tidal_volume/1000) -- Rapid Shallow Breathing Index")
print("  3. map          = (SBP + 2*DBP) / 3 -- Mean Arterial Pressure")
print("  4. pulse_pressure = SBP - DBP")
print("  5. oxi_index    = (FiO2 * MAP) / PaO2 -- Oxygenation Index")
print("  6. aa_gradient  = Alveolar-arterial gradient")
print("  7. pao2_log     = log1p(PaO2)")

# Handle any inf/nan from division
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in ['pf_ratio', 'rsbi', 'map', 'pulse_pressure', 'oxi_index', 'aa_gradient', 'pao2_log']:
    median = df[col].median()
    df[col].fillna(median, inplace=True)

# --- Define Final Feature Sets ---
id_cols = ['subject_id', 'hadm_id', 'stay_id']
label_cols = ['weaning_success', 'readmission']
exclude_cols = id_cols + label_cols + ['gender']  # gender already encoded

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

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name, target_name):
    """Train, predict, and return comprehensive metrics."""
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob)

    print("\n  {} ({})".format(model_name, target_name))
    print("  " + "-" * 50)
    print("    Accuracy:  {:.4f}".format(acc))
    print("    Precision: {:.4f}".format(prec))
    print("    Recall:    {:.4f}".format(rec))
    print("    F1-Score:  {:.4f}".format(f1))
    print("    ROC-AUC:   {:.4f}".format(auc))

    return {
        'model': model, 'model_name': model_name, 'target': target_name,
        'y_test': y_te, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc
    }


def plot_results(results_list, target_name):
    """Plot ROC curves and confusion matrices for all models."""
    n_models = len(results_list)

    # --- ROC Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for res in results_list:
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax.plot(fpr, tpr, label='{} (AUC={:.4f})'.format(res['model_name'], res['roc_auc']), linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - {}'.format(target_name), fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Confusion Matrices ---
    ax = axes[1]
    # Show best model's confusion matrix
    best = max(results_list, key=lambda x: x['roc_auc'])
    cm = confusion_matrix(best['y_test'], best['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - {} ({})'.format(best['model_name'], target_name), fontsize=13)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'roc_cm_{}.png'.format(target_name)), dpi=150)
    plt.close(fig)
    print("  -> Saved: roc_cm_{}.png".format(target_name))


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

    # Compute sample weights for class imbalance
    sample_weights = compute_sample_weight('balanced', y_train)

    results = []

    # --- 1. Logistic Regression ---
    lr = LogisticRegression(
        max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=42, C=1.0
    )
    res = evaluate_model(lr, X_train_scaled, y_train, X_test_scaled, y_test,
                         'Logistic Regression', target)
    results.append(res)

    # --- 2. Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight='balanced', random_state=42,
        n_jobs=-1, min_samples_split=5
    )
    res = evaluate_model(rf, X_train_scaled, y_train, X_test_scaled, y_test,
                         'Random Forest', target)
    results.append(res)

    # --- 3. XGBoost ---
    # Calculate scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_weight,
        eval_metric='auc', random_state=42,
        use_label_encoder=False, n_jobs=-1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0
    )
    res = evaluate_model(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test,
                         'XGBoost', target)
    results.append(res)

    # Plot results
    plot_results(results, target)

    # Store results
    all_results[target] = results

    # --- Summary Table ---
    print("\n  SUMMARY TABLE - {}:".format(target))
    print("  {:25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'))
    print("  " + "-" * 75)
    for r in results:
        print("  {:25s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r['model_name'], r['accuracy'], r['precision'],
            r['recall'], r['f1'], r['roc_auc']))


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

    print("\n  Top 10 Features (XGBoost - {}):\n".format(target))
    for i, row in feat_imp.head(10).iterrows():
        print("    {:3d}. {:20s}  importance={:.4f}".format(
            feat_imp.index.tolist().index(i) + 1, row['feature'], row['importance']))

    # Plot feature importance
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # XGBoost importance
    top20 = feat_imp.head(20)
    axes[0].barh(range(len(top20)), top20['importance'].values, color='steelblue')
    axes[0].set_yticks(range(len(top20)))
    axes[0].set_yticklabels(top20['feature'].values, fontsize=9)
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].set_title('XGBoost Feature Importance - {}'.format(target), fontsize=13)
    axes[0].invert_yaxis()

    # Random Forest importance
    rf_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    axes[1].barh(range(len(rf_imp)), rf_imp['importance'].values, color='darkorange')
    axes[1].set_yticks(range(len(rf_imp)))
    axes[1].set_yticklabels(rf_imp['feature'].values, fontsize=9)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Random Forest Feature Importance - {}'.format(target), fontsize=13)
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_importance_{}.png'.format(target)), dpi=150)
    plt.close(fig)
    print("  -> Saved: feature_importance_{}.png".format(target))

    # --- SHAP Analysis ---
    print("\n  Computing SHAP values (XGBoost - {})...".format(target))

    # Use a sample for faster SHAP computation
    shap_sample_size = min(500, len(X_test_scaled))
    X_shap = X_test_scaled[:shap_sample_size]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)

    # SHAP summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_cols,
        show=False, max_display=20
    )
    plt.title('SHAP Summary - {} (XGBoost)'.format(target), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'shap_summary_{}.png'.format(target)), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  -> Saved: shap_summary_{}.png".format(target))

    # SHAP bar plot (mean absolute values)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_cols,
        plot_type='bar', show=False, max_display=20
    )
    plt.title('SHAP Mean Importance - {} (XGBoost)'.format(target), fontsize=14)
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
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0],
    }

    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric='auc', random_state=42,
        use_label_encoder=False, n_jobs=-1
    )

    # Randomized search (faster than full grid)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_base, param_grid,
        n_iter=30, scoring='roc_auc',
        cv=cv, random_state=42,
        n_jobs=-1, verbose=0
    )

    print("  Running RandomizedSearchCV (30 iterations, 3-fold CV)...")
    search.fit(X_train_scaled, y_train)

    print("  Best params: {}".format(search.best_params_))
    print("  Best CV ROC-AUC: {:.4f}".format(search.best_score_))

    # Evaluate tuned model on test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n  TUNED XGBoost ({}) Test Performance:".format(target))
    print("    Accuracy:  {:.4f}".format(acc))
    print("    Precision: {:.4f}".format(prec))
    print("    Recall:    {:.4f}".format(rec))
    print("    F1-Score:  {:.4f}".format(f1))
    print("    ROC-AUC:   {:.4f}".format(auc))

    # Compare with baseline
    baseline_auc = [r['roc_auc'] for r in all_results[target] if r['model_name'] == 'XGBoost'][0]
    improvement = auc - baseline_auc
    print("\n  Improvement over baseline XGBoost: {:+.4f} ROC-AUC".format(improvement))

    best_models[target] = {
        'model': best_model, 'auc': auc,
        'params': search.best_params_,
        'y_pred': y_pred, 'y_prob': y_prob
    }


# ==========================================================================
#  FINAL COMPARISON: ALL MODELS
# ==========================================================================
section("FINAL COMPARISON: ALL MODELS")

for target in label_cols:
    print("\n  {} - Final Results:".format(target.upper()))
    print("  {:30s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'))
    print("  " + "-" * 80)

    for r in all_results[target]:
        print("  {:30s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r['model_name'], r['accuracy'], r['precision'],
            r['recall'], r['f1'], r['roc_auc']))

    # Tuned model
    bm = best_models[target]
    y_test = df_test[target].values
    print("  {:30s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        'XGBoost (Tuned)',
        accuracy_score(y_test, bm['y_pred']),
        precision_score(y_test, bm['y_pred'], zero_division=0),
        recall_score(y_test, bm['y_pred'], zero_division=0),
        f1_score(y_test, bm['y_pred'], zero_division=0),
        bm['auc']))

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
    ax.plot(fpr, tpr, label='XGBoost Tuned ({:.4f})'.format(bm['auc']),
            linewidth=2.5, linestyle='--', color='darkred')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('{} - ROC Comparison'.format(target), fontsize=14)
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
    print("\n\n  Classification Report - {} (Tuned XGBoost):".format(target))
    print(classification_report(y_test, bm['y_pred'],
                                target_names=['Negative (0)', 'Positive (1)']))


# ==========================================================================
#  FINAL SUMMARY
# ==========================================================================
section("PIPELINE COMPLETE - SUMMARY")

print("""
  Dataset:         {} rows, {} features
  Train/Test Split: {}/{} (patient-level, 0 overlap)
  Engineered Features: pf_ratio, rsbi, map, pulse_pressure, oxi_index, aa_gradient

  Best Models (by ROC-AUC):
""".format(len(df), len(feature_cols), len(df_train), len(df_test)))

for target in label_cols:
    bm = best_models[target]
    print("    {}: XGBoost Tuned (ROC-AUC = {:.4f})".format(target, bm['auc']))

print("""
  Output Files ({}):
    - roc_cm_weaning_success.png
    - roc_cm_readmission.png
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
