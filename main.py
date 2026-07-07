import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "."
OUTPUT_DIR = "outputs/heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Helper Functions
# -----------------------------
def load_csvs(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")
    items = []
    for file in csv_files:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        items.append((os.path.basename(file), df))
    logging.info(f"Loaded {len(items)} CSV files")
    return items

def pick_join_keys(dfs, user_keys=None):
    if user_keys:
        keys = [k.strip().lower() for k in user_keys]
        missing = {k for k in keys if not all(k in df.columns for df in dfs)}
        if missing:
            raise ValueError(f"Join keys not present in all files: {sorted(missing)}")
        return keys

    candidates = [
        ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "order_time_jittered_utc"],
        ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"],
        ["anon_id"],
    ]
    for keys in candidates:
        if all(all(k in df.columns for k in keys) for df in dfs):
            return keys
    if "anon_id" in dfs[0].columns:
        return ["anon_id"]
    raise ValueError("Could not determine join keys shared across CSVs")

def deduplicate_by_keys(df, keys):
    existing = [k for k in keys if k in df.columns]
    if not existing:
        return df
    return df.groupby(existing, as_index=False).first()

def merge_dataframes(dfs, on_keys, how="inner"):
    merged = deduplicate_by_keys(dfs[0], on_keys)
    for df in dfs[1:]:
        right = deduplicate_by_keys(df, on_keys)
        right = right.loc[:, ~right.columns.duplicated()]
        right_non_keys = [c for c in right.columns if c not in on_keys and c not in merged.columns]
        if not right_non_keys:
            continue
        merged = pd.merge(
            merged,
            right[on_keys + right_non_keys],
            on=on_keys,
            how=how,
        )
        merged = merged.loc[:, ~merged.columns.duplicated()]
    logging.info(f"Merged DataFrame shape: {merged.shape}")
    return merged

def drop_sparse_columns(df, threshold=0.7):
    missing_ratio = df.isnull().mean()
    df = df.loc[:, missing_ratio < threshold]
    logging.info(f"After dropping sparse columns (>={threshold*100:.0f}% missing): {df.shape}")
    return df

# -----------------------------
# Plotting Functions
# -----------------------------
def plot_missing_heatmap(df, filename):
    if df.empty:
        logging.warning("Skipping missing heatmap: empty DataFrame")
        return
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    plt.figure(figsize=(14, 5))
    sns.heatmap(sample.isnull(), yticklabels=False, cbar=False)
    plt.title("Missing Value Heatmap (Sampled)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_correlation_heatmap(df, filename):
    numeric_df = df.select_dtypes(include=["int64", "int32", "float64", "float32"]).copy()
    if numeric_df.empty:
        logging.warning("Skipping correlation heatmap: no numeric columns")
        return
    if numeric_df.shape[1] > 25:
        numeric_df = numeric_df.iloc[:, :25]
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Top Numeric Features)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_descriptive_stats(df, filename):
    df.describe(include='all').transpose().to_csv(filename)

def save_missing_summary(df, filename):
    df.isnull().mean().sort_values(ascending=False).to_csv(filename, header=['missing_fraction'])

def save_categorical_counts(df, filename, top_n=20):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if not len(cat_cols):
        return
    with open(filename, 'w') as f:
        for col in cat_cols:
            f.write(f"Column: {col}\n")
            f.write(df[col].value_counts().head(top_n).to_string())
            f.write("\n\n")

def generate_per_file_eda(items):
    for name, df in items:
        safe = os.path.splitext(name)[0]
        plot_missing_heatmap(df, os.path.join(OUTPUT_DIR, f"{safe}_missing_values_heatmap.png"))
        plot_correlation_heatmap(df, os.path.join(OUTPUT_DIR, f"{safe}_correlation_heatmap.png"))
        save_descriptive_stats(df, os.path.join(OUTPUT_DIR, f"{safe}_descriptive_stats.csv"))
        save_missing_summary(df, os.path.join(OUTPUT_DIR, f"{safe}_missing_summary.csv"))
        save_categorical_counts(df, os.path.join(OUTPUT_DIR, f"{safe}_categorical_counts.txt"))

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Microbiology EDA and heatmaps")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Folder containing CSV files")
    parser.add_argument("--join-keys", default=None, help="Comma-separated list of join keys")
    parser.add_argument("--join-type", default="inner", choices=["inner", "left", "right", "outer"], help="Join type")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional sample N rows for merged plotting")
    parser.add_argument("--per-file-only", action="store_true", help="Skip merging, generate per-file EDA only")
    args = parser.parse_args()

    items = load_csvs(args.data_dir)

    if args.per_file_only:
        logging.info("Generating per-file EDA (skipping merge)...")
        generate_per_file_eda(items)
        logging.info("Per-file EDA completed successfully")
        return

    dfs = [df for _, df in items]
    user_keys = [k.strip() for k in args.join_keys.split(",")] if args.join_keys else None
    join_keys = pick_join_keys(dfs, user_keys)
    logging.info(f"Using join keys: {join_keys}; join type: {args.join_type}")

    merged_df = merge_dataframes(dfs, on_keys=join_keys, how=args.join_type)
    merged_df = drop_sparse_columns(merged_df)

    if args.sample_rows and len(merged_df) > args.sample_rows:
        merged_df = merged_df.sample(n=args.sample_rows, random_state=42)
        logging.info(f"Downsampled merged data to {len(merged_df)} rows for plotting")

    os.makedirs("outputs", exist_ok=True)
    merged_df.to_csv("outputs/microbiology_combined_clean.csv", index=False)
    merged_df.sample(n=min(5000, len(merged_df)), random_state=42).to_csv("outputs/microbiology_combined_sample.csv", index=False)

    plot_missing_heatmap(merged_df, os.path.join(OUTPUT_DIR, "missing_values_heatmap.png"))
    plot_correlation_heatmap(merged_df, os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    save_descriptive_stats(merged_df, os.path.join(OUTPUT_DIR, "merged_descriptive_stats.csv"))
    save_missing_summary(merged_df, os.path.join(OUTPUT_DIR, "merged_missing_summary.csv"))
    save_categorical_counts(merged_df, os.path.join(OUTPUT_DIR, "merged_categorical_counts.txt"))

    logging.info("Merged EDA completed successfully")

if __name__ == "__main__":
    main()
