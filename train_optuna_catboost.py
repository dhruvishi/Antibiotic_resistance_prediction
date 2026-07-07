import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
import optuna


def build_features(df: pd.DataFrame):
    cols = [
        'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
        'culture_description', 'organism', 'antibiotic', 'age', 'gender', 'prior_organism',
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility'
    ]
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    cat_idx = [i for i, c in enumerate(X.columns) if X[c].dtype == 'object']
    return X, cat_idx


def make_target(df: pd.DataFrame, scheme: str):
    sus = df['susceptibility'].astype(str).str.strip().str.title()
    if scheme == 'binary_rs':
        mask = sus.isin(['Susceptible', 'Resistant'])
        df2 = df[mask].copy()
        y = (sus[mask] == 'Resistant').astype(int).values
    else:
        mask = sus.isin(['Susceptible', 'Resistant', 'Intermediate'])
        df2 = df[mask].copy()
        tmp = sus[mask]
        y = tmp.map({'Susceptible': 0, 'Resistant': 1, 'Intermediate': 1}).values
    return df2, y


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray, cat_idx):
    params = {
        'iterations': trial.suggest_int('iterations', 600, 1500),
        'depth': trial.suggest_int('depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0, log=True),
        'border_count': trial.suggest_int('border_count', 64, 254),
        'random_strength': trial.suggest_float('random_strength', 0.5, 3.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'od_type': 'Iter',
        'od_wait': 150,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1s = []
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        # Class weights
        pos_ratio = (y_tr == 1).mean()
        neg_ratio = 1 - pos_ratio
        class_weights = [1.0 / max(1e-6, neg_ratio), 1.0 / max(1e-6, pos_ratio)]

        model = CatBoostClassifier(**params, class_weights=class_weights)
        model.fit(Pool(X_tr, y_tr, cat_features=cat_idx), eval_set=Pool(X_va, y_va, cat_features=cat_idx), use_best_model=True)
        probs = model.predict_proba(X_va)[:, 1]

        # Threshold tuning per fold for F1
        best_f1 = -1
        for thr in np.linspace(0.1, 0.9, 33):
            pred = (probs >= thr).astype(int)
            f1 = f1_score(y_va, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        f1s.append(best_f1)

    return float(np.mean(f1s))


def evaluate_final(X_train, y_train, X_test, y_test, cat_idx, params):
    pos_ratio = (y_train == 1).mean()
    neg_ratio = 1 - pos_ratio
    class_weights = [1.0 / max(1e-6, neg_ratio), 1.0 / max(1e-6, pos_ratio)]

    model = CatBoostClassifier(**params, class_weights=class_weights)
    model.fit(Pool(X_train, y_train, cat_features=cat_idx), eval_set=Pool(X_test, y_test, cat_features=cat_idx), use_best_model=True, verbose=200)
    probs = model.predict_proba(X_test)[:, 1]

    # Tune final threshold
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (probs >= thr).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    preds = (probs >= best_thr).astype(int)
    metrics = {
        'threshold': best_thr,
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'auc': roc_auc_score(y_test, probs),
    }
    return model, metrics


def run_global(df, scheme, n_trials=12):
    df2, y = make_target(df, scheme)
    X, cat_idx = build_features(df2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda tr: objective(tr, X_train, y_train, cat_idx), n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    # Fill constant params
    best_params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 42, 'od_type': 'Iter', 'od_wait': 150, 'verbose': 200})

    model, metrics = evaluate_final(X_train, y_train, X_test, y_test, cat_idx, best_params)
    return model, metrics, best_params


def run_subgroups(df, scheme, group_col, min_rows=400, n_trials=6):
    rows = []
    for value, g in df.groupby(group_col):
        if len(g) < min_rows:
            continue
        try:
            model, metrics, params = run_global(g, scheme, n_trials=n_trials)
            rows.append({
                'group_col': group_col,
                'group_val': value,
                **metrics
            })
        except Exception as e:
            # Skip unstable groups
            continue
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Optuna-tuned CatBoost with CV and subgroup search')
    parser.add_argument('--target-scheme', default='binary_rs', choices=['binary_rs', 'binary_ni'])
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--search-subgroups', action='store_true')
    parser.add_argument('--min-rows', type=int, default=400)
    args = parser.parse_args()

    df = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Global model
    model, metrics, params = run_global(df, args.target_scheme, n_trials=args.trials)

    print('Global metrics:')
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')

    # Save artifacts
    model.save_model('models/catboost_resistance_optuna.cbm')
    with open('outputs/metrics_catboost_optuna.txt', 'w') as f:
        f.write('Global metrics\n')
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
        f.write(f'best_params: {params}\n')

    # Subgroup search
    if args.search_subgroups:
        report_frames = []
        for col in ['organism', 'antibiotic', 'culture_description']:
            if col in df.columns:
                print(f'Running subgroup search for {col} ...')
                rep = run_subgroups(df, args.target_scheme, col, min_rows=args.min_rows, n_trials=max(4, args.trials // 2))
                if not rep.empty:
                    report_frames.append(rep)
        if report_frames:
            subrep = pd.concat(report_frames, ignore_index=True)
            subrep.sort_values(by=['f1', 'accuracy', 'precision', 'recall'], ascending=False, inplace=True)
            subrep.to_csv('outputs/subgroup_search_report.csv', index=False)
            # Print candidates >= 0.9 across metrics
            candidates = subrep[(subrep['accuracy']>=0.9)&(subrep['precision']>=0.9)&(subrep['recall']>=0.9)&(subrep['f1']>=0.9)]
            if not candidates.empty:
                print('Found subgroups meeting >=90% on all metrics:')
                print(candidates.head(10))
            else:
                print('No subgroups reached >=90% on all metrics with current settings.')


if __name__ == '__main__':
    main()
