import os
import argparse
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostClassifier
import logging

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Helper Functions
# -----------------------------
def build_targets(df: pd.DataFrame, target_scheme: str) -> pd.DataFrame:
    df = df.dropna(subset=['susceptibility']).copy()
    sus = df['susceptibility'].astype(str).str.strip().str.title()
    if target_scheme == 'binary_rs':
        mask = sus.isin(['Susceptible', 'Resistant'])
        df = df[mask].copy()
        df['target'] = (sus[mask] == 'Resistant').astype(int).values
        df['target_label'] = np.where(df['target'] == 1, 'Resistant', 'Susceptible')
    else:
        mask = sus.isin(['Susceptible', 'Resistant', 'Intermediate'])
        df = df[mask].copy()
        tmp = sus[mask]
        df['target'] = tmp.map({'Susceptible': 0, 'Resistant': 1, 'Intermediate': 1}).values
        df['target_label'] = np.where(df['target'] == 1, 'Non-susceptible', 'Susceptible')
    return df

def get_feature_cols(df: pd.DataFrame):
    feature_cols = [
        'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
        'culture_description', 'organism', 'antibiotic', 'age', 'gender', 'prior_organism',
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility'
    ]
    return [c for c in feature_cols if c in df.columns]

def load_model(model_path: str) -> CatBoostClassifier:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as pf:
            model = pickle.load(pf)
        logging.info(f"Loaded CatBoost model from {model_path}")
        return model
    cbm_fallback = os.path.splitext(model_path)[0] + '.cbm'
    if os.path.exists(cbm_fallback):
        model = CatBoostClassifier()
        model.load_model(cbm_fallback)
        logging.info(f"Loaded CatBoost model from fallback {cbm_fallback}")
        return model
    raise FileNotFoundError("No CatBoost model file found (.pkl or .cbm)")

def plot_boxplot(df: pd.DataFrame, outpath: str):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='target_label', y='prob_positive')
    sample = df.sample(min(1000, len(df)), random_state=42)
    sns.stripplot(data=sample, x='target_label', y='prob_positive', color='black', alpha=0.3, jitter=0.25)
    plt.title('Predicted Probability by Target')
    plt.xlabel('Target')
    plt.ylabel('Predicted P(positive)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logging.info(f"Saved boxplot to {outpath}")

def plot_scatter(df: pd.DataFrame, outpath: str):
    if 'age' not in df.columns:
        logging.warning("Skipping scatter plot: 'age' column not found")
        return
    scatter_df = df[['age', 'prob_positive', 'target_label']].dropna()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=scatter_df.sample(min(5000, len(scatter_df))), x='age', y='prob_positive', hue='target_label', alpha=0.5)
    plt.title('Predicted Probability vs Age')
    plt.xlabel('Age')
    plt.ylabel('Predicted P(positive)')
    plt.legend(title='Target')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logging.info(f"Saved scatter plot to {outpath}")

def plot_feature_importance(model: CatBoostClassifier, pool: Pool, feature_cols: list, outpath: str, top_n: int = 25):
    importances = model.get_feature_importance(pool)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x='importance', y='feature', orient='h')
    plt.title('Top Feature Importances (CatBoost)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logging.info(f"Saved feature importance plot to {outpath}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Visualize CatBoost model predictions and importances')
    parser.add_argument('--model-path', default='models/catboost_resistance_new.pkl', help='Path to pickled CatBoost model')
    parser.add_argument('--csv', default='microbiology_combined_clean.csv', help='Input CSV used for training')
    parser.add_argument('--antibiotic', default=None, help='Optional: filter to a specific antibiotic name')
    parser.add_argument('--target-scheme', default='binary_rs', choices=['binary_rs', 'binary_ni'], help='Target definition to use')
    parser.add_argument('--outdir', default='outputs/visualizations', help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    if args.antibiotic:
        df = df[df['antibiotic'].astype(str) == args.antibiotic].copy()
        logging.info(f"Filtered data to antibiotic: {args.antibiotic} ({len(df)} rows)")

    df = build_targets(df, args.target_scheme)
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    cat_features = [i for i, c in enumerate(X.columns) if X[c].dtype == 'object']
    pool = Pool(X, cat_features=cat_features)

    model = load_model(args.model_path)

    # Predict probabilities
    df['prob_positive'] = model.predict_proba(pool)[:, 1]

    # Save predictions
    pred_csv = os.path.join(args.outdir, 'predictions_with_probs.csv')
    df.to_csv(pred_csv, index=False)
    logging.info(f"Saved predictions CSV to {pred_csv}")

    # Plots
    plot_boxplot(df, os.path.join(args.outdir, 'boxplot_target_probs.png'))
    plot_scatter(df, os.path.join(args.outdir, 'scatter_age_probs.png'))
    plot_feature_importance(model, pool, feature_cols, os.path.join(args.outdir, 'bar_feature_importance.png'))

if __name__ == '__main__':
    main()
