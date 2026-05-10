#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Lightweight Screening
Train both XGBoost and Random Forest models and compare their performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import argparse
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def _optional_smote(X, y, method='smote', random_state=42):
    """Optional oversampling (SMOTE/ADASYN) on training data only. Returns (X_res, y_res)."""
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
    except ImportError:
        return X, y
    k = min(5, max(1, int(np.sum(y == 1)) - 1))
    if k < 1:
        return X, y
    if method == 'adasyn':
        sampler = ADASYN(random_state=random_state, n_neighbors=k)
    else:
        sampler = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


def user_level_split(df, test_size=0.2, random_state=42):
    """
    Split data at user level to prevent data leakage
    Same user should not appear in both train and test sets
    """
    print("Performing user-level split...")
    
    unique_users = df['file_user'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_users)
    
    n_test_users = int(len(unique_users) * test_size)
    test_users = set(unique_users[:n_test_users])
    train_users = set(unique_users[n_test_users:])
    
    train_mask = df['file_user'].isin(train_users)
    test_mask = df['file_user'].isin(test_users)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  Train users: {len(train_users)}, Train records: {len(train_df):,}")
    print(f"  Test users: {len(test_users)}, Test records: {len(test_df):,}")
    print(f"  Train malicious rate: {train_df['is_malicious'].mean()*100:.2f}%")
    print(f"  Test malicious rate: {test_df['is_malicious'].mean()*100:.2f}%")
    
    return train_df, test_df, train_users, test_users


def add_interaction_features(df):
    """Add cross features: ops_per_unique_file, off_hour_activity_ratio. Modifies df in place."""
    if 'user_total_ops' in df.columns and 'user_unique_files' in df.columns:
        df['ops_per_unique_file'] = df['user_total_ops'] / (df['user_unique_files'].replace(0, np.nan).fillna(1e-6))
    if 'is_after_hours' in df.columns and 'file_user' in df.columns:
        df['off_hour_activity_ratio'] = df.groupby('file_user')['is_after_hours'].transform('mean')
    elif 'logon_after_hours_ratio' in df.columns:
        df['off_hour_activity_ratio'] = df['logon_after_hours_ratio']
    return df


def _feature_matrix_from_df(df, feature_cols):
    """Numeric matrix only: no second copy of interaction logic (caller must have same cols)."""
    return df[list(feature_cols)].fillna(0).replace([np.inf, -np.inf], 0)


def train_xgb_model(X_train, y_train, random_state=42, fn_weight=2.0):
    """Train XGBoost: lighter default (fewer trees / shallower) for faster Stage 1; still cost-sensitive."""
    print("\nTraining XGBoost model (cost-sensitive, simplified config)...")
    pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=pos_weight * fn_weight,  # Cost-sensitive: higher FN cost
        min_child_weight=3,  # Stronger regularization
        gamma=0.1,  # Regularization to reduce overfitting
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  XGBoost training completed")
    return model


def train_xgb_model_with_cv(X_train, y_train, groups, n_folds=5, n_iter=10, random_state=42, fn_weight=2.0, use_group_cv=True):
    """Train XGBoost with CV + smaller random search (fewer trees/depth options = faster)."""
    print("\nTraining XGBoost with CV + compact hyperparameter search...")
    pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    scale_pos = pos_weight * fn_weight
    # n_jobs=1 here: RandomizedSearchCV already parallelizes trials; avoids oversubscription.
    base = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=scale_pos,
        tree_method='hist',
        n_jobs=1,
    )
    param_dist = {
        'n_estimators': [80, 120, 200],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.05, 0.08],
        'subsample': [0.75, 0.85],
        'colsample_bytree': [0.75, 0.85],
        'min_child_weight': [2, 4],
        'gamma': [0.1, 0.2],
        'reg_alpha': [0.1, 0.2],
        'reg_lambda': [1.0, 2.0],
    }
    cv = GroupKFold(n_splits=n_folds) if use_group_cv else KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='recall',
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    if use_group_cv:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    print(f"  Best CV recall: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    best = search.best_estimator_
    # Refit model can use all cores for final large fit
    if hasattr(best, 'set_params'):
        best.set_params(n_jobs=-1)
    return best


def train_rf_model(X_train, y_train, random_state=42, fn_weight=2.0):
    """Random Forest: fewer / shallower trees than before for faster training."""
    print("\nTraining Random Forest model (cost-sensitive, simplified config)...")
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    class_weight = {0: 1.0, 1: (neg_count / max(1, pos_count)) * fn_weight}
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=12,
        min_samples_leaf=6,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        max_features='sqrt',
    )
    model.fit(X_train, y_train)
    print("  Random Forest training completed")
    return model


def train_rf_model_with_cv(X_train, y_train, groups, n_folds=5, n_iter=10, random_state=42, fn_weight=2.0, use_group_cv=True):
    """Random Forest with CV + smaller random search for speed."""
    print("\nTraining Random Forest with CV + compact hyperparameter search...")
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    cw = {0: 1.0, 1: (neg_count / max(1, pos_count)) * fn_weight}
    base = RandomForestClassifier(
        class_weight=cw,
        random_state=random_state,
        n_jobs=1,
        max_features='sqrt',
    )
    param_dist = {
        'n_estimators': [80, 120, 200],
        'max_depth': [8, 12],
        'min_samples_split': [8, 16],
        'min_samples_leaf': [4, 8],
    }
    cv = GroupKFold(n_splits=n_folds) if use_group_cv else KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='recall',
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    if use_group_cv:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    print(f"  Best CV recall: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    best = search.best_estimator_
    if hasattr(best, 'set_params'):
        best.set_params(n_jobs=-1)
    return best


def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate model performance"""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def find_optimal_threshold_for_recall(y_true, y_pred_proba, target_recall=0.99, min_recall=0.95):
    """
    Find threshold that maximizes recall while maintaining minimum recall
    If target recall cannot be achieved, find threshold with maximum possible recall
    """
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find maximum achievable recall
    max_recall = recall_curve.max()
    max_recall_idx = np.argmax(recall_curve)
    
    # Try to achieve target recall
    if max_recall >= target_recall:
        # Find threshold that achieves at least target recall
        valid_indices = np.where(recall_curve >= target_recall)[0]
        if len(valid_indices) > 0:
            # Among valid indices, choose the one with highest precision
            best_idx = valid_indices[np.argmax(precision_curve[valid_indices])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            optimal_recall = recall_curve[best_idx]
            optimal_precision = precision_curve[best_idx]
        else:
            # Fall back to maximum recall
            optimal_threshold = thresholds[max_recall_idx] if max_recall_idx < len(thresholds) else thresholds[-1]
            optimal_recall = max_recall
            optimal_precision = precision_curve[max_recall_idx]
    else:
        # Use maximum achievable recall
        optimal_threshold = thresholds[max_recall_idx] if max_recall_idx < len(thresholds) else thresholds[-1]
        optimal_recall = max_recall
        optimal_precision = precision_curve[max_recall_idx]
        print(f"    Warning: Target recall {target_recall*100}% not achievable. Using maximum recall {max_recall*100:.2f}%")
    
    # Ensure minimum recall
    if optimal_recall < min_recall:
        # Find threshold that achieves at least min_recall
        valid_indices = np.where(recall_curve >= min_recall)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(precision_curve[valid_indices])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            optimal_recall = recall_curve[best_idx]
            optimal_precision = precision_curve[best_idx]
    
    return optimal_threshold, optimal_recall, optimal_precision


def filter_top_k_percent(df, risk_scores, k_percent=5, model_name="", threshold_mode='fixed_top_k', train_scores_ref=None):
    """
    Filter suspicious sequences. threshold_mode:
      - fixed_top_k: top k% by count (default)
      - percentile: all with score >= train 95th percentile (use train_scores_ref)
      - adaptive_k: k = 10% if mean(score) > 0.3 else 5%
    """
    df = df.copy()
    df['risk_score'] = risk_scores
    df_sorted = df.sort_values('risk_score', ascending=False)
    n_total = len(df_sorted)

    if threshold_mode == 'percentile' and train_scores_ref is not None:
        th = np.percentile(train_scores_ref, 95)
        top_sequences = df_sorted[df_sorted['risk_score'] >= th].copy()
        n_top = len(top_sequences)
        print(f"\nFiltering by score >= {th:.4f} (95th pct of train) ({model_name})...")
    elif threshold_mode == 'adaptive_k':
        mean_score = float(np.mean(risk_scores))
        k_percent = 10.0 if mean_score > 0.3 else 5.0
        n_top = int(n_total * k_percent / 100)
        top_sequences = df_sorted.head(n_top).copy()
        print(f"\nFiltering top {k_percent}% (adaptive: mean_score={mean_score:.3f}) ({model_name})...")
    else:
        n_top = int(n_total * k_percent / 100)
        top_sequences = df_sorted.head(n_top).copy()
        print(f"\nFiltering top {k_percent}% suspicious sequences ({model_name})...")

    malicious_in_top = top_sequences['is_malicious'].sum()
    total_malicious = df['is_malicious'].sum()
    recall_in_top = malicious_in_top / total_malicious if total_malicious > 0 else 0
    precision_in_top = malicious_in_top / n_top if n_top > 0 else 0

    print(f"  Total sequences: {n_total:,}")
    print(f"  Selected: {n_top:,}")
    print(f"  Malicious in selected: {malicious_in_top:,} / {total_malicious:,}")
    print(f"  Recall: {recall_in_top*100:.2f}%")
    print(f"  Precision: {precision_in_top*100:.2f}%")

    return top_sequences, {
        'n_top': n_top,
        'malicious_in_top': malicious_in_top,
        'total_malicious': total_malicious,
        'recall': recall_in_top,
        'precision': precision_in_top
    }


def top_k_recall_sweep(df, risk_scores, k_percent_list, model_name=""):
    """Report recall at multiple top-k% (e.g. 5%, 3%, 2%) for threshold moving analysis."""
    total_malicious = df['is_malicious'].sum()
    if total_malicious == 0:
        return
    n_total = len(df)
    df_sorted = df.copy()
    df_sorted['_score'] = risk_scores
    df_sorted = df_sorted.sort_values('_score', ascending=False)
    print(f"\n  Top-K recall sweep ({model_name}):")
    print(f"  {'k%':>6} {'selected':>10} {'malicious':>10} {'recall%':>8} {'prec%':>8}")
    for k in k_percent_list:
        n_top = int(n_total * k / 100)
        top = df_sorted.head(n_top)
        mal = top['is_malicious'].sum()
        rec = mal / total_malicious * 100
        prec = mal / n_top * 100 if n_top else 0
        print(f"  {k:>5.1f}% {n_top:>10,} {mal:>10,} {rec:>7.2f}% {prec:>7.2f}%")


def compare_models(results_dict):
    """Compare performance of different models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)
    
    comparison_df = pd.DataFrame(results_dict).T
    print("\nPerformance Metrics:")
    print(comparison_df.to_string())
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv')
    print("\nComparison saved to model_comparison.csv")
    
    return comparison_df


def run_5fold_user_evaluation(df, feature_cols, target_recall=0.99, random_state=42):
    """
    5-fold user-level cross-validation: each fold trains on 4 user groups
    and tests on the remaining 1, rotating five times; reports per-fold
    metrics together with mean ± std.
    """
    exclude_cols = ['file_user', 'file_date', 'is_malicious', 'malicious_scenario']
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['is_malicious'].astype(int)
    groups = df['file_user'].values

    cv = GroupKFold(n_splits=5)
    fold_metrics = {'XGBoost': [], 'RandomForest': []}  # list of dicts per fold

    print("\n" + "="*60)
    print("5-FOLD USER-LEVEL CROSS-VALIDATION EVALUATION")
    print("(4 user groups train, 1 user group test per fold)")
    print("="*60)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]

        # Train with fixed params (no HP search for speed)
        xgb_m = train_xgb_model(X_train, y_train, random_state=random_state)
        rf_m = train_rf_model(X_train, y_train, random_state=random_state)

        # Threshold from train
        xgb_proba = xgb_m.predict_proba(X_train)[:, 1]
        rf_proba = rf_m.predict_proba(X_train)[:, 1]
        xgb_th, _, _ = find_optimal_threshold_for_recall(y_train.values, xgb_proba, target_recall=target_recall, min_recall=0.90)
        rf_th, _, _ = find_optimal_threshold_for_recall(y_train.values, rf_proba, target_recall=target_recall, min_recall=0.90)

        # Test evaluation
        xgb_res = evaluate_model(xgb_m, X_test, y_test.values, threshold=xgb_th)
        rf_res = evaluate_model(rf_m, X_test, y_test.values, threshold=rf_th)

        fold_metrics['XGBoost'].append({
            'Precision': xgb_res['precision'], 'Recall': xgb_res['recall'],
            'F1': xgb_res['f1'], 'PR-AUC': xgb_res['pr_auc']
        })
        fold_metrics['RandomForest'].append({
            'Precision': rf_res['precision'], 'Recall': rf_res['recall'],
            'F1': rf_res['f1'], 'PR-AUC': rf_res['pr_auc']
        })
        print(f"  XGBoost  Test: P={xgb_res['precision']:.4f} R={xgb_res['recall']:.4f} F1={xgb_res['f1']:.4f} PR-AUC={xgb_res['pr_auc']:.4f}")
        print(f"  RF       Test: P={rf_res['precision']:.4f} R={rf_res['recall']:.4f} F1={rf_res['f1']:.4f} PR-AUC={rf_res['pr_auc']:.4f}")

    # Aggregate: mean ± std
    report = {}
    for model_name in ['XGBoost', 'RandomForest']:
        list_of_dicts = fold_metrics[model_name]
        report[model_name] = {}
        for key in list_of_dicts[0]:
            vals = [d[key] for d in list_of_dicts]
            report[model_name][f'{key}_mean'] = np.mean(vals)
            report[model_name][f'{key}_std'] = np.std(vals)

    print("\n" + "="*60)
    print("5-FOLD EVALUATION REPORT (mean ± std across 5 test folds)")
    print("="*60)
    for model_name in ['XGBoost', 'RandomForest']:
        print(f"\n{model_name}:")
        for key in ['Precision', 'Recall', 'F1', 'PR-AUC']:
            m, s = report[model_name][f'{key}_mean'], report[model_name][f'{key}_std']
            print(f"  {key}: {m:.4f} ± {s:.4f}")
    metrics = ['Precision', 'Recall', 'F1', 'PR-AUC']
    report_df = pd.DataFrame(
        [[report[m][f'{k}_mean'] for k in metrics] for m in ['XGBoost', 'RandomForest']],
        index=['XGBoost', 'RandomForest'],
        columns=metrics
    )
    for k in metrics:
        report_df[f'{k}_std'] = [report[m][f'{k}_std'] for m in report_df.index]
    report_df.to_csv('cv5_evaluation_report.csv')
    print("\nSaved cv5_evaluation_report.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Lightweight Screening - Compare XGBoost and Random Forest')
    _default_input = os.path.join(os.path.dirname(__file__), '..', 'features.csv')
    parser.add_argument('--input', '-i', type=str, default=_default_input,
                       help='Input features CSV file (default: project root features.csv)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (user-level split)')
    parser.add_argument('--top-k', type=float, default=5.0,
                       help='Top k% to forward to Stage 2 (default: 5)')
    parser.add_argument('--target-recall', type=float, default=0.99,
                       help='Target recall for threshold selection (default: 0.99, maximize recall)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds for GroupKFold CV (default: 5)')
    parser.add_argument('--hp-n-iter', type=int, default=10,
                       help='Random search trials per model (default: 10; use 20+ for deeper search)')
    parser.add_argument('--no-hp-tune', action='store_true',
                       help='Disable hyperparameter search; use fixed default params only')
    parser.add_argument('--eval-5fold', action='store_true',
                       help='Run 5-fold user-level CV evaluation and save cv5_evaluation_report.csv (mean ± std)')
    parser.add_argument('--smote', action='store_true',
                       help='Apply SMOTE oversampling on training set to address class imbalance')
    parser.add_argument('--adasyn', action='store_true',
                       help='Apply ADASYN oversampling on training set (alternative to --smote)')
    parser.add_argument('--fn-weight', type=float, default=2.0,
                       help='Cost weight for false negatives (scale_pos_weight / class_weight multiplier, default 2.0)')
    parser.add_argument('--threshold-mode', type=str, default='fixed_top_k',
                       choices=['fixed_top_k', 'percentile', 'adaptive_k'],
                       help='fixed_top_k=top k%%; percentile=score>=train 95pct; adaptive_k=k%% by mean score')
    parser.add_argument('--iso-forest', action='store_true',
                       help='Add Isolation Forest anomaly score as an extra feature (unsupervised signal)')
    parser.add_argument('--top-k-sweep', type=str, default='',
                       help='Comma-separated k%% values to report recall (e.g. 5,3,2); if set, print recall at each k%%')

    args = parser.parse_args()
    
    # Load data
    print(f"Loading features from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False, engine='c')
    print(f"  Loaded {len(df):,} records")
    
    if 'is_malicious' not in df.columns:
        raise ValueError("'is_malicious' column not found in input file")
    add_interaction_features(df)
    exclude_cols = ['file_user', 'file_date', 'is_malicious', 'malicious_scenario']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    # 5-fold user-level evaluation: each fold trains on 4 user groups and
    # tests on the remaining 1; reports the 5-fold mean ± std.
    if args.eval_5fold:
        run_5fold_user_evaluation(df, feature_cols, target_recall=args.target_recall)
    
    # User-level split
    train_df, test_df, train_users, test_users = user_level_split(
        df, test_size=args.test_size, random_state=42
    )
    
    # Train/test matrices: interaction cols already on df before split — avoid second copy/groupby
    print("Building train/test feature matrices...")
    X_train = _feature_matrix_from_df(train_df, feature_cols)
    X_test = _feature_matrix_from_df(test_df, feature_cols)
    print(f"  Using {len(feature_cols)} features × {len(X_train):,} train / {len(X_test):,} test rows")
    y_train = train_df['is_malicious'].astype(int)
    y_test = test_df['is_malicious'].astype(int)
    groups_train = train_df['file_user'].values

    # Optional: add Isolation Forest anomaly score as extra feature (unsupervised)
    if args.iso_forest:
        print("\nAdding Isolation Forest anomaly score as feature...")
        iso = IsolationForest(random_state=42, n_estimators=50, contamination=0.01, n_jobs=-1)
        iso.fit(X_train)
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['iso_score'] = iso.decision_function(X_train)
        X_test['iso_score'] = iso.decision_function(X_test)
        df_fill = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        df['iso_score'] = iso.decision_function(df_fill)
        feature_cols = list(feature_cols) + ['iso_score']
        print(f"  Feature set now has {len(feature_cols)} features (incl. iso_score)")

    # Optional oversampling (SMOTE/ADASYN) on training set only
    use_group_cv = True
    if args.smote or args.adasyn:
        method = 'smote' if args.smote else 'adasyn'
        print(f"\nApplying {method.upper()} on training set...")
        X_arr = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
        y_arr = np.asarray(y_train)
        X_res, y_res = _optional_smote(X_arr, y_arr, method=method)
        X_train = pd.DataFrame(X_res, columns=feature_cols)
        y_train = pd.Series(y_res)
        print(f"  Resampled train size: {len(y_train):,} (malicious: {y_train.sum():,})")
        use_group_cv = False  # synthetic samples have no user group

    # Dictionary to store all results
    all_results = {}

    # Train and evaluate both models
    models = {}

    # ========== XGBoost ==========
    print("\n" + "="*60)
    print("XGBOOST MODEL")
    print("="*60)

    if args.no_hp_tune:
        xgb_model = train_xgb_model(X_train, y_train, fn_weight=args.fn_weight)
    else:
        xgb_model = train_xgb_model_with_cv(
            X_train, y_train, groups_train,
            n_folds=args.n_folds, n_iter=args.hp_n_iter,
            fn_weight=args.fn_weight, use_group_cv=use_group_cv,
        )
    models['XGBoost'] = xgb_model
    
    # Evaluate on training set
    print("\n=== Training Set Evaluation (XGBoost) ===")
    xgb_train_results = evaluate_model(xgb_model, X_train, y_train)
    print(f"  Precision: {xgb_train_results['precision']:.4f}")
    print(f"  Recall: {xgb_train_results['recall']:.4f}")
    print(f"  F1-Score: {xgb_train_results['f1']:.4f}")
    print(f"  PR-AUC: {xgb_train_results['pr_auc']:.4f}")
    
    # Find optimal threshold (maximize recall)
    xgb_threshold, xgb_recall, xgb_precision = find_optimal_threshold_for_recall(
        y_train, xgb_train_results['y_pred_proba'], target_recall=args.target_recall, min_recall=0.90
    )
    print(f"\n  Optimal threshold (maximizing recall): {xgb_threshold:.4f}")
    print(f"  At this threshold - Precision: {xgb_precision:.4f}, Recall: {xgb_recall:.4f} ({xgb_recall*100:.2f}%)")
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation (XGBoost) ===")
    xgb_test_results = evaluate_model(xgb_model, X_test, y_test, threshold=xgb_threshold)
    print(f"  Precision: {xgb_test_results['precision']:.4f}")
    print(f"  Recall: {xgb_test_results['recall']:.4f}")
    print(f"  F1-Score: {xgb_test_results['f1']:.4f}")
    print(f"  PR-AUC: {xgb_test_results['pr_auc']:.4f}")
    
    # Filter: fixed top-k, percentile, or adaptive (single full-DF matrix for predict)
    X_full = _feature_matrix_from_df(df, feature_cols)
    xgb_risk_scores = xgb_model.predict_proba(X_full)[:, 1]
    train_proba_xgb = xgb_train_results['y_pred_proba']
    xgb_suspicious, xgb_filter_stats = filter_top_k_percent(
        df, xgb_risk_scores, k_percent=args.top_k, model_name="XGBoost",
        threshold_mode=args.threshold_mode, train_scores_ref=train_proba_xgb if args.threshold_mode == 'percentile' else None,
    )
    if args.top_k_sweep:
        k_list = [float(x.strip()) for x in args.top_k_sweep.split(',') if x.strip()]
        if k_list:
            top_k_recall_sweep(df, xgb_risk_scores, k_list, "XGBoost")
    
    # Store results
    all_results['XGBoost'] = {
        'Train_Precision': xgb_train_results['precision'],
        'Train_Recall': xgb_train_results['recall'],
        'Train_F1': xgb_train_results['f1'],
        'Train_PR-AUC': xgb_train_results['pr_auc'],
        'Test_Precision': xgb_test_results['precision'],
        'Test_Recall': xgb_test_results['recall'],
        'Test_F1': xgb_test_results['f1'],
        'Test_PR-AUC': xgb_test_results['pr_auc'],
        'Optimal_Threshold': xgb_threshold,
        'TopK_Recall': xgb_filter_stats['recall'],
        'TopK_Precision': xgb_filter_stats['precision'],
        'TopK_Malicious': xgb_filter_stats['malicious_in_top'],
        'TopK_Total': xgb_filter_stats['n_top']
    }
    
    # Save XGBoost model and results
    joblib.dump(xgb_model, 'stage1_xgb_model.pkl')
    xgb_suspicious.to_csv('suspicious_sequences_xgb.csv', index=False)
    
    # XGBoost feature importance
    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    xgb_importance.to_csv('feature_importance_xgb.csv', index=False)
    
    # ========== Random Forest ==========
    print("\n" + "="*60)
    print("RANDOM FOREST MODEL")
    print("="*60)

    if args.no_hp_tune:
        rf_model = train_rf_model(X_train, y_train, fn_weight=args.fn_weight)
    else:
        rf_model = train_rf_model_with_cv(
            X_train, y_train, groups_train,
            n_folds=args.n_folds, n_iter=args.hp_n_iter,
            fn_weight=args.fn_weight, use_group_cv=use_group_cv,
        )
    models['RandomForest'] = rf_model
    
    # Evaluate on training set
    print("\n=== Training Set Evaluation (Random Forest) ===")
    rf_train_results = evaluate_model(rf_model, X_train, y_train)
    print(f"  Precision: {rf_train_results['precision']:.4f}")
    print(f"  Recall: {rf_train_results['recall']:.4f}")
    print(f"  F1-Score: {rf_train_results['f1']:.4f}")
    print(f"  PR-AUC: {rf_train_results['pr_auc']:.4f}")
    
    # Find optimal threshold (maximize recall)
    rf_threshold, rf_recall, rf_precision = find_optimal_threshold_for_recall(
        y_train, rf_train_results['y_pred_proba'], target_recall=args.target_recall, min_recall=0.90
    )
    print(f"\n  Optimal threshold (maximizing recall): {rf_threshold:.4f}")
    print(f"  At this threshold - Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f} ({rf_recall*100:.2f}%)")
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation (Random Forest) ===")
    rf_test_results = evaluate_model(rf_model, X_test, y_test, threshold=rf_threshold)
    print(f"  Precision: {rf_test_results['precision']:.4f}")
    print(f"  Recall: {rf_test_results['recall']:.4f}")
    print(f"  F1-Score: {rf_test_results['f1']:.4f}")
    print(f"  PR-AUC: {rf_test_results['pr_auc']:.4f}")
    
    # Filter: fixed top-k, percentile, or adaptive
    rf_risk_scores = rf_model.predict_proba(X_full)[:, 1]
    train_proba_rf = rf_train_results['y_pred_proba']
    rf_suspicious, rf_filter_stats = filter_top_k_percent(
        df, rf_risk_scores, k_percent=args.top_k, model_name="Random Forest",
        threshold_mode=args.threshold_mode, train_scores_ref=train_proba_rf if args.threshold_mode == 'percentile' else None,
    )
    if args.top_k_sweep:
        k_list = [float(x.strip()) for x in args.top_k_sweep.split(',') if x.strip()]
        if k_list:
            top_k_recall_sweep(df, rf_risk_scores, k_list, "Random Forest")
    
    # Store results
    all_results['RandomForest'] = {
        'Train_Precision': rf_train_results['precision'],
        'Train_Recall': rf_train_results['recall'],
        'Train_F1': rf_train_results['f1'],
        'Train_PR-AUC': rf_train_results['pr_auc'],
        'Test_Precision': rf_test_results['precision'],
        'Test_Recall': rf_test_results['recall'],
        'Test_F1': rf_test_results['f1'],
        'Test_PR-AUC': rf_test_results['pr_auc'],
        'Optimal_Threshold': rf_threshold,
        'TopK_Recall': rf_filter_stats['recall'],
        'TopK_Precision': rf_filter_stats['precision'],
        'TopK_Malicious': rf_filter_stats['malicious_in_top'],
        'TopK_Total': rf_filter_stats['n_top']
    }
    
    # Save Random Forest model and results
    joblib.dump(rf_model, 'stage1_rf_model.pkl')
    rf_suspicious.to_csv('suspicious_sequences_rf.csv', index=False)
    
    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_importance.to_csv('feature_importance_rf.csv', index=False)
    
    # ========== Comparison ==========
    comparison_df = compare_models(all_results)
    
    # Print top features for both models
    print("\n" + "="*60)
    print("TOP 10 FEATURES - XGBoost")
    print("="*60)
    print(xgb_importance.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP 10 FEATURES - Random Forest")
    print("="*60)
    print(rf_importance.head(10).to_string(index=False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest Test F1-Score:")
    if all_results['XGBoost']['Test_F1'] > all_results['RandomForest']['Test_F1']:
        print(f"  XGBoost: {all_results['XGBoost']['Test_F1']:.4f} (Winner)")
        print(f"  Random Forest: {all_results['RandomForest']['Test_F1']:.4f}")
    else:
        print(f"  XGBoost: {all_results['XGBoost']['Test_F1']:.4f}")
        print(f"  Random Forest: {all_results['RandomForest']['Test_F1']:.4f} (Winner)")
    
    print(f"\nBest Test PR-AUC:")
    if all_results['XGBoost']['Test_PR-AUC'] > all_results['RandomForest']['Test_PR-AUC']:
        print(f"  XGBoost: {all_results['XGBoost']['Test_PR-AUC']:.4f} (Winner)")
        print(f"  Random Forest: {all_results['RandomForest']['Test_PR-AUC']:.4f}")
    else:
        print(f"  XGBoost: {all_results['XGBoost']['Test_PR-AUC']:.4f}")
        print(f"  Random Forest: {all_results['RandomForest']['Test_PR-AUC']:.4f} (Winner)")
    
    print(f"\nBest Test Recall (Most Important for Security):")
    if all_results['XGBoost']['Test_Recall'] > all_results['RandomForest']['Test_Recall']:
        print(f"  XGBoost: {all_results['XGBoost']['Test_Recall']*100:.2f}% (Winner - Fewer False Negatives)")
        print(f"  Random Forest: {all_results['RandomForest']['Test_Recall']*100:.2f}%")
    else:
        print(f"  XGBoost: {all_results['XGBoost']['Test_Recall']*100:.2f}%")
        print(f"  Random Forest: {all_results['RandomForest']['Test_Recall']*100:.2f}% (Winner - Fewer False Negatives)")
    
    print(f"\nBest Top-K Recall (Malicious captured in Top {args.top_k}%):")
    if all_results['XGBoost']['TopK_Recall'] > all_results['RandomForest']['TopK_Recall']:
        print(f"  XGBoost: {all_results['XGBoost']['TopK_Recall']*100:.2f}% (Winner)")
        print(f"  Random Forest: {all_results['RandomForest']['TopK_Recall']*100:.2f}%")
    else:
        print(f"  XGBoost: {all_results['XGBoost']['TopK_Recall']*100:.2f}%")
        print(f"  Random Forest: {all_results['RandomForest']['TopK_Recall']*100:.2f}% (Winner)")
    
    print("\n=== Stage 1 Screening Complete ===")
    print("\nGenerated Files:")
    print("  - stage1_xgb_model.pkl")
    print("  - stage1_rf_model.pkl")
    print("  - suspicious_sequences_xgb.csv")
    print("  - suspicious_sequences_rf.csv")
    print("  - feature_importance_xgb.csv")
    print("  - feature_importance_rf.csv")
    print("  - model_comparison.csv")
    print("\nNext step: Use suspicious sequences for Stage 2 LLM-based reasoning")


if __name__ == '__main__':
    main()
