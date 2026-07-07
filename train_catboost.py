import os
import argparse 
import numpy as np
import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from catboost import CatBoostClassifier, Pool, cv


def fine_tune_catboost(X_train, y_train, cat_features):
    base_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0,
        'task_type': 'CPU',
        'cat_features': cat_features,
        'auto_class_weights': 'Balanced'
    }

    candidates = [
        {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0},
        {'depth': 8, 'learning_rate': 0.02, 'l2_leaf_reg': 5.0},
        {'depth': 10, 'learning_rate': 0.015, 'l2_leaf_reg': 8.0},
    ]

    best_auc = -1
    best_params = None
    train_pool = Pool(X_train, y_train, cat_features=cat_features)

    for params in candidates:
        cv_data = cv(
            params={**base_params, **params, 'iterations': 2000},
            pool=train_pool,
            fold_count=3,
            partition_random_seed=42,
            early_stopping_rounds=200,
            shuffle=True,
            verbose=False
        )
        auc = cv_data['test-AUC-mean'].max()
        if auc > best_auc:
            best_auc = auc
            best_params = params

    print(f"Best CV AUC: {best_auc:.4f} with params {best_params}")
    return best_params or {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5.0}


def main():
    parser = argparse.ArgumentParser(description='Train CatBoost model on microbiology data')
    parser.add_argument('--antibiotic', default=None)
    parser.add_argument('--target-scheme', default='binary_rs', choices=['binary_rs', 'binary_ni'])
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()

    # ============================================================
    # Load single combined CSV
    # ============================================================
    df = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)

    # ============================================================
    # Target creation
    # ============================================================
    df = df.dropna(subset=['susceptibility'])

    if args.antibiotic:
        df = df[df['antibiotic'].astype(str) == args.antibiotic]

    sus = df['susceptibility'].astype(str).str.strip().str.title()

    if args.target_scheme == 'binary_rs':
        mask = sus.isin(['Susceptible', 'Resistant'])
        df = df[mask].copy()
        df['target'] = (sus[mask] == 'Resistant').astype(int)
    else:
        mask = sus.isin(['Susceptible', 'Resistant', 'Intermediate'])
        df = df[mask].copy()
        df['target'] = sus[mask].map({
            'Susceptible': 0,
            'Resistant': 1,
            'Intermediate': 1
        })

    # ============================================================
    # Feature columns (already in combined CSV)
    # ============================================================
    feature_cols = [
        'medication_category',
        'medication_name',
        'antibiotic_class',
        'ordering_mode',
        'culture_description',
        'organism',
        'antibiotic',
        'age',
        'gender',
        'prior_organism',
        'was_positive',
        'time_to_culturetime',
        'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre',
        'implied_susceptibility'
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['target'].values

    # ============================================================
    # Handle missing values
    # ============================================================
    X = X.fillna('Unknown')

    cat_features = [
        i for i, c in enumerate(X.columns)
        if X[c].dtype == 'object'
    ]

    # ============================================================
    # Train / Test split
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ============================================================
    # Optional tuning
    # ============================================================
    base_params = fine_tune_catboost(X_train, y_train, cat_features) if args.tune else {
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5.0
    }

    # ============================================================
    # Train model
    # ============================================================
    model = CatBoostClassifier(
        iterations=3000,
        **base_params,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        od_type='Iter',
        od_wait=200,
        auto_class_weights='Balanced',
        verbose=200
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_test, y_test, cat_features=cat_features)

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # ============================================================
    # Evaluation
    # ============================================================
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y_test, (probs >= t)) for t in thresholds]
    best_thr = thresholds[np.argmax(f1s)]

    preds = (probs >= best_thr).astype(int)

    metrics = {
        'best_threshold': float(best_thr),
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'auc': roc_auc_score(y_test, probs),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        'classification_report': classification_report(y_test, preds, output_dict=True)
    }

    print("\n=== Final Evaluation ===")
    print(json.dumps(metrics, indent=4))

    # ============================================================
    # Save model & metadata
    # ============================================================
    os.makedirs('models', exist_ok=True)

    model.save_model('models/catboost_resistance_model.cbm')

    with open('models/catboost_resistance_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/model_metadata.json', 'w') as f:
        json.dump({
            'feature_columns': feature_cols,
            'categorical_features': [X.columns[i] for i in cat_features],
            'target_scheme': args.target_scheme,
            'antibiotic_filter': args.antibiotic,
            'metrics': metrics,
            'parameters': model.get_all_params()
        }, f, indent=4)

    print("\nModel and metadata saved successfully!")


if __name__ == '__main__':
    main()
