import os
import argparse
import json
import math
import random
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    candidates = {
        'patient_id': ['anon_id', 'patient_id'],
        'order_id': ['order_proc_id_coded', 'order_id', 'encounter_id'],
        'datetime': ['culture_datetime', 'collection_datetime', 'order_datetime'],
        'organism': ['organism', 'organism_name'],
        'antibiotic': ['antibiotic', 'antibiotic_name'],
        'susceptibility': ['susceptibility', 'mic_sir'],
        'regression_target': ['resistant_time_to_culturetime', 'medication_time_to_culturetime', 'time_to_culturetime'],
    }
    resolved = {}
    for key, opts in candidates.items():
        found = None
        for col in opts:
            if col in df.columns:
                found = col
                break
        resolved[key] = found
    return resolved


def build_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Suggested features – include only those present
    cat_candidates = [
        'sex', 'gender', 'race', 'ward', 'icu_status', 'admission_type', 'hospital_unit',
        'specimen_type', 'infection_site', 'organism_family', 'organism', 'organism_name',
        'antibiotic_class', 'antibiotic', 'antibiotic_name', 'ordering_mode', 'prior_organism',
    ]
    num_candidates = [
        'age', 'length_of_stay', 'charlson_index',
        # Labs/vitals (examples)
        'wbc_last', 'wbc_mean', 'wbc_min', 'wbc_max',
        'creatinine_last', 'creatinine_mean', 'creatinine_min', 'creatinine_max',
        'heart_rate_last', 'heart_rate_mean', 'heart_rate_min', 'heart_rate_max',
        'systolic_bp_last', 'systolic_bp_mean', 'systolic_bp_min', 'systolic_bp_max',
        # Prior exposures
        'fluoroquinolone_days_30', 'fluoroquinolone_days_90',
        'beta_lactam_days_30', 'beta_lactam_days_90',
        'macrolide_days_30', 'macrolide_days_90',
        'received_same_class_before', 'time_since_last_exposure',
        # Context from merged dataset
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility',
    ]

    # Age can be numeric or categorical; include dynamically
    dynamic_cat = []
    if 'age' in df.columns and not pd.api.types.is_numeric_dtype(df['age']):
        dynamic_cat.append('age')
    cat_features = [c for c in cat_candidates if c in df.columns] + dynamic_cat
    # Numeric heuristics: include provided numeric candidates (only numeric dtype) and other numeric cols except targets
    base_num = [c for c in num_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    other_numeric = [
        c for c in df.columns
        if c not in base_num + cat_features
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Exclude obvious ID/time/targets from other_numeric
    exclude = {'anon_id', 'patient_id', 'order_proc_id_coded', 'order_id', 'encounter_id',
               'culture_datetime', 'collection_datetime', 'order_datetime'}
    other_numeric = [c for c in other_numeric if c not in exclude]
    # Ensure uniqueness
    num_features = list(dict.fromkeys(base_num + other_numeric))
    return cat_features, num_features


class TabularARMDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str],
                 label_col: str, reg_col: Optional[str],
                 cat_maps: Optional[Dict[str, Dict[str, int]]] = None,
                 num_scaler: Optional[StandardScaler] = None,
                 fit: bool = False):
        self.df = df.copy()
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.label_col = label_col
        self.reg_col = reg_col

        # Binary label from susceptibility
        sus = self.df[label_col].astype(str).str.strip().str.title()
        self.df['label_bin'] = sus.map({'Resistant': 1, 'Intermediate': 1, 'Susceptible': 0}).fillna(np.nan)
        self.df = self.df.dropna(subset=['label_bin'])
        self.df['label_bin'] = self.df['label_bin'].astype(int)

        # Regression target can be missing; keep even if NaN (masked later)
        if self.reg_col is not None and self.reg_col not in self.df.columns:
            self.reg_col = None

        # Prepare categorical maps
        if cat_maps is None:
            cat_maps = {}
        self.cat_maps = cat_maps
        if fit:
            for c in self.cat_cols:
                vals = self.df[c].astype(str).fillna('NA').unique().tolist()
                # Reserve 0 for unknown
                mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}
                mapping['<UNK>'] = 0
                self.cat_maps[c] = mapping

        # Encode categoricals
        for c in self.cat_cols:
            mapping = self.cat_maps[c]
            self.df[c] = self.df[c].astype(str).fillna('NA').map(lambda x: mapping.get(x, 0)).astype(int)

        # Numeric scaler
        if num_scaler is None:
            num_scaler = StandardScaler()
        self.num_scaler = num_scaler
        # Fill NaN with medians before scaling
        for c in self.num_cols:
            if c not in self.df.columns:
                self.df[c] = np.nan
            self.df[c] = self.df[c].fillna(self.df[c].median())
        num_mat = self.df[self.num_cols].to_numpy(dtype=np.float32)
        if fit:
            self.num_scaler.fit(num_mat)
        num_scaled = self.num_scaler.transform(num_mat)
        self.num_tensor = torch.from_numpy(num_scaled)

        # Store categorical tensors
        self.cat_tensors = [torch.from_numpy(self.df[c].to_numpy(dtype=np.int64)) for c in self.cat_cols]
        self.label_tensor = torch.from_numpy(self.df['label_bin'].to_numpy(dtype=np.float32))
        if self.reg_col is not None:
            reg_vals = self.df[self.reg_col].to_numpy(dtype=np.float32)
            # Fill NaN with median of available
            if np.isnan(reg_vals).any():
                med = np.nanmedian(reg_vals)
                reg_vals = np.where(np.isnan(reg_vals), med, reg_vals)
            self.reg_tensor = torch.from_numpy(reg_vals)
        else:
            self.reg_tensor = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cats = [t[idx] for t in self.cat_tensors] if self.cat_cols else []
        nums = self.num_tensor[idx]
        label = self.label_tensor[idx]
        reg = self.reg_tensor[idx] if self.reg_tensor is not None else torch.tensor(0.0, dtype=torch.float32)
        return cats, nums, label, reg


class ARMDTabularNet(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_dim: int,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList()
        emb_out_dims = []
        for card in cat_cardinalities:
            # Embedding dim rule of thumb
            emb_dim = min(50, (card + 1) // 2)
            self.embeddings.append(nn.Embedding(card + 1, emb_dim))
            emb_out_dims.append(emb_dim)

        total_in = sum(emb_out_dims) + num_dim
        self.backbone = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Heads
        self.cls_head = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, cats: List[torch.Tensor], nums: torch.Tensor):
        if len(self.embeddings) > 0:
            embs = [emb(c) for emb, c in zip(self.embeddings, cats)]
            emb_cat = torch.cat(embs, dim=1)
            x = torch.cat([emb_cat, nums], dim=1)
        else:
            x = nums
        h = self.backbone(x)
        cls_logit = self.cls_head(h)
        reg_out = self.reg_head(h)
        return cls_logit.squeeze(1), reg_out.squeeze(1)


def compute_metrics(y_true, y_prob, y_reg_true=None, y_reg_pred=None) -> Dict[str, float]:
    preds = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    out = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc
    }
    if y_reg_true is not None and y_reg_pred is not None:
        mae = mean_absolute_error(y_reg_true, y_reg_pred)
        rmse = math.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
        out.update({'mae': mae, 'rmse': rmse})
    return out


def permutation_importance(model: ARMDTabularNet, val_loader: DataLoader,
                           cat_cols: List[str], num_cols: List[str], device: str = 'cpu') -> pd.DataFrame:
    # Baseline AUC on validation
    model.eval()
    with torch.no_grad():
        probs = []
        labels = []
        for cats, nums, label, reg in val_loader:
            cats = [c.to(device) for c in cats]
            nums = nums.to(device)
            logit, _ = model(cats, nums)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.append(prob)
            labels.append(label.numpy())
        baseline_auc = roc_auc_score(np.concatenate(labels), np.concatenate(probs))

    # Permute each numeric feature independently and measure drop in AUC
    importances = []
    for i, f in enumerate(num_cols):
        probs_perm = []
        labels_perm = []
        for cats, nums, label, reg in val_loader:
            nums_perm = nums.clone()
            # Permute column i
            idx = torch.randperm(nums_perm.size(0))
            nums_perm[:, i] = nums_perm[idx, i]
            cats = [c.to(device) for c in cats]
            nums_perm = nums_perm.to(device)
            with torch.no_grad():
                logit, _ = model(cats, nums_perm)
                prob = torch.sigmoid(logit).cpu().numpy()
            probs_perm.append(prob)
            labels_perm.append(label.numpy())
        auc_perm = roc_auc_score(np.concatenate(labels_perm), np.concatenate(probs_perm))
        importances.append(baseline_auc - auc_perm)
    imp_df = pd.DataFrame({'feature': num_cols, 'importance': importances}).sort_values('importance', ascending=False)
    return imp_df


def suggest_alternatives(model: ARMDTabularNet, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str],
                         cat_maps: Dict[str, Dict[str, int]], scaler: StandardScaler,
                         cols: Dict[str, Optional[str]],
                         patient_id: str, organism: str, top_k: int = 5,
                         device: str = 'cpu') -> pd.DataFrame:
    # Filter rows for patient+organism
    pid_col = cols['patient_id']
    org_col = cols['organism']
    ab_col = cols['antibiotic']
    sub = df[(df[pid_col].astype(str) == str(patient_id)) & (df[org_col].astype(str) == str(organism))].copy()
    if sub.empty:
        raise ValueError('No rows found for given patient and organism')

    # Prepare features for each candidate antibiotic row
    # Encode categoricals
    def encode_cat_row(row):
        enc = []
        for c in cat_cols:
            val = str(row.get(c, 'NA'))
            enc.append(cat_maps[c].get(val, 0))
        # Return list of per-column tensors with batch size 1
        return [torch.tensor([v], dtype=torch.int64) for v in enc]

    # Scale numerics
    def encode_num_row(row):
        vals = [row.get(c, np.nan) for c in num_cols]
        vals = [v if pd.notna(v) else 0.0 for v in vals]
        arr = np.array(vals, dtype=np.float32).reshape(1, -1)
        arr = scaler.transform(arr)
        return torch.tensor(arr[0], dtype=torch.float32)

    model.eval()
    rows = []
    with torch.no_grad():
        for _, r in sub.iterrows():
            cats = [t.to(device) for t in encode_cat_row(r)]
            nums = encode_num_row(r).unsqueeze(0).to(device)
            logit, reg = model(cats, nums)
            prob_resistant = torch.sigmoid(logit).cpu().item()
            rows.append({
                'antibiotic': r[ab_col],
                'prob_resistant': prob_resistant,
                'prob_susceptible': 1.0 - prob_resistant,
                'time_to_resistance_pred_days': float(reg.cpu().item())
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(['prob_susceptible', 'time_to_resistance_pred_days'], ascending=[False, False]).head(top_k)
    return out


def main():
    parser = argparse.ArgumentParser(description='ARMD PyTorch Multitask Pipeline')
    parser.add_argument('--csv', default='microbiology_combined_clean.csv', help='Merged ARMD CSV file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--outdir', default='outputs', help='Outputs directory')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', choices=['train', 'infer', 'suggest', 'list'], default='train')
    parser.add_argument('--patient-id', default=None)
    parser.add_argument('--organism', default=None)
    parser.add_argument('--antibiotic', default=None)
    parser.add_argument('--contains', default=None, help='Substring search for organism or antibiotic in list mode')
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    cols = detect_columns(df)
    if cols['susceptibility'] is None:
        raise ValueError('Missing susceptibility or mic_sir column')

    # Train mode
    if args.mode == 'train':
        df_clean = df.dropna(subset=[cols['susceptibility']]).copy()
        cat_cols, num_cols = build_feature_lists(df_clean)

        # Train/valid split
        sus_bin = df_clean[cols['susceptibility']].astype(str).str.strip().str.title().map({'Resistant': 1, 'Intermediate': 1, 'Susceptible': 0}).fillna(np.nan)
        df_clean = df_clean.loc[sus_bin.notna()].copy()
        y_cls = sus_bin.loc[sus_bin.notna()].astype(int).values
        train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=y_cls)

        # Datasets
        train_ds = TabularARMDDataset(train_df, cat_cols, num_cols, cols['susceptibility'], cols['regression_target'], fit=True)
        val_ds = TabularARMDDataset(val_df, cat_cols, num_cols, cols['susceptibility'], cols['regression_target'],
                                    cat_maps=train_ds.cat_maps, num_scaler=train_ds.num_scaler, fit=False)

        # Save preprocessing artifacts
        with open(os.path.join(args.models_dir, 'dl_cat_maps.pkl'), 'wb') as f:
            pickle.dump(train_ds.cat_maps, f)
        with open(os.path.join(args.models_dir, 'dl_num_scaler.pkl'), 'wb') as f:
            pickle.dump(train_ds.num_scaler, f)
        with open(os.path.join(args.models_dir, 'dl_feature_cols.json'), 'w') as f:
            json.dump({'cat_cols': cat_cols, 'num_cols': num_cols, 'resolved_cols': cols}, f)

        # Model
        # Cardinalities from fitted categorical maps
        cat_cards = [len(train_ds.cat_maps[c]) for c in cat_cols]
        model = ARMDTabularNet(cat_cards, num_dim=len(num_cols), hidden_dim=args.hidden_dim, dropout=args.dropout)
        device = torch.device(args.device)
        model.to(device)

        # Class weight for imbalance
        pos_ratio = (train_ds.label_tensor.numpy() == 1).mean()
        neg_ratio = 1 - pos_ratio
        pos_weight = torch.tensor(neg_ratio / max(pos_ratio, 1e-6), dtype=torch.float32, device=device)

        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        def collate(batch):
            cats = list(zip(*[b[0] for b in batch])) if len(train_ds.cat_cols) > 0 else []
            cats = [torch.stack(c, dim=0) for c in cats]
            nums = torch.stack([b[1] for b in batch], dim=0)
            labels = torch.stack([b[2] for b in batch], dim=0)
            regs = torch.stack([b[3] for b in batch], dim=0)
            return cats, nums, labels, regs

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        best_auc = -1
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for cats, nums, labels, regs in train_loader:
                cats = [c.to(device) for c in cats]
                nums = nums.to(device)
                labels = labels.to(device)
                regs = regs.to(device)
                optimizer.zero_grad()
                logit, reg_out = model(cats, nums)
                loss_cls = bce(logit, labels)
                loss_reg = mse(reg_out, regs)
                loss = loss_cls + loss_reg
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * labels.size(0)

            # Validation
            model.eval()
            with torch.no_grad():
                probs = []
                labels_all = []
                regs_true = []
                regs_pred = []
                for cats, nums, labels, regs in val_loader:
                    cats = [c.to(device) for c in cats]
                    nums = nums.to(device)
                    logit, reg_out = model(cats, nums)
                    prob = torch.sigmoid(logit).cpu().numpy()
                    probs.append(prob)
                    labels_all.append(labels.numpy())
                    regs_true.append(regs.numpy())
                    regs_pred.append(reg_out.cpu().numpy())
                y_prob = np.concatenate(probs)
                y_true = np.concatenate(labels_all)
                y_reg_t = np.concatenate(regs_true)
                y_reg_p = np.concatenate(regs_pred)
                metrics = compute_metrics(y_true, y_prob, y_reg_t, y_reg_p)

            print(f"Epoch {epoch}/{args.epochs} loss={epoch_loss/len(train_ds):.4f} "
                  f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} RMSE={metrics.get('rmse', float('nan')):.3f}")

            # Save best by AUC
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                torch.save(model.state_dict(), os.path.join(args.models_dir, 'armd_tabular_net.pt'))
                with open(os.path.join(args.outdir, 'metrics_dl.txt'), 'w') as f:
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")

        # Visualizations (boxplot, scatter, bar importance)
        # Boxplot: predicted prob by label
        plt.figure(figsize=(8, 5))
        df_val_vis = val_ds.df.copy()
        with torch.no_grad():
            probs_list = []
            for i in range(len(val_ds)):
                cats_i, nums_i, _, _ = val_ds[i]
                cats_i = [c.unsqueeze(0).to(device) for c in cats_i]
                nums_i = nums_i.unsqueeze(0).to(device)
                logit, _ = model(cats_i, nums_i)
                p = torch.sigmoid(logit).cpu().item()
                probs_list.append(p)
        df_val_vis['prob_positive'] = probs_list
        df_val_vis['target_label'] = np.where(df_val_vis['label_bin'] == 1, 'Resistant/Non-susceptible', 'Susceptible')
        sns.boxplot(data=df_val_vis, x='target_label', y='prob_positive')
        sns.stripplot(data=df_val_vis.sample(min(1000, len(df_val_vis))), x='target_label', y='prob_positive', color='black', alpha=0.3, jitter=0.25)
        plt.title('DL Model: Predicted Probability by Target')
        plt.tight_layout()
        box_path = os.path.join(args.outdir, 'visualizations_dl_boxplot.png')
        os.makedirs(args.outdir, exist_ok=True)
        plt.savefig(box_path, dpi=150)
        plt.close()

        # Scatter: age vs probability
        if 'age' in df_val_vis.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=df_val_vis.sample(min(5000, len(df_val_vis))), x='age', y='prob_positive', hue='target_label', alpha=0.5)
            plt.title('DL Model: Probability vs Age')
            plt.tight_layout()
            scatter_path = os.path.join(args.outdir, 'visualizations_dl_scatter_age.png')
            plt.savefig(scatter_path, dpi=150)
            plt.close()

        # Bar: permutation importance (numeric features)
        val_loader_eval = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: (
            [torch.stack(list(zip(*[x[0] for x in b]))[i], dim=0) for i in range(len(cat_cols))] if len(cat_cols) > 0 else [],
            torch.stack([x[1] for x in b], dim=0),
            torch.stack([x[2] for x in b], dim=0),
            torch.stack([x[3] for x in b], dim=0)
        ))
        imp_df = permutation_importance(model, val_loader_eval, cat_cols, num_cols, device=device)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp_df.head(25), x='importance', y='feature', orient='h')
        plt.title('DL Model: Permutation Importance (AUC drop)')
        plt.tight_layout()
        bar_path = os.path.join(args.outdir, 'visualizations_dl_importance.png')
        plt.savefig(bar_path, dpi=150)
        plt.close()

        print('Saved visualizations:')
        print(f' - {box_path}')
        if 'age' in df_val_vis.columns:
            print(f' - {scatter_path}')
        print(f' - {bar_path}')

    elif args.mode == 'suggest':
        # Load artifacts and model, then suggest alternatives
        with open(os.path.join(args.models_dir, 'dl_cat_maps.pkl'), 'rb') as f:
            cat_maps = pickle.load(f)
        with open(os.path.join(args.models_dir, 'dl_num_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(args.models_dir, 'dl_feature_cols.json'), 'r') as f:
            meta = json.load(f)
        cat_cols = meta['cat_cols']
        num_cols = meta['num_cols']
        cols = meta['resolved_cols']

        df_full = pd.read_csv(args.csv, low_memory=False)
        cat_cards = [len(cat_maps[c]) for c in cat_cols]
        model = ARMDTabularNet(cat_cards, num_dim=len(num_cols))
        model.load_state_dict(torch.load(os.path.join(args.models_dir, 'armd_tabular_net.pt'), map_location=args.device))
        device = torch.device(args.device)
        model.to(device)

        if args.patient_id is None or args.organism is None:
            raise ValueError('--patient-id and --organism are required for suggest mode')
        suggestions = suggest_alternatives(model, df_full, cat_cols, num_cols, cat_maps, scaler, cols,
                                           args.patient_id, args.organism, device=args.device)
        out_path = os.path.join(args.outdir, 'alternative_antibiotics.csv')
        suggestions.to_csv(out_path, index=False)
        print(f'Saved suggestions to {out_path}')

    elif args.mode == 'list':
        # List and search available IDs / organisms / antibiotics
        df_full = pd.read_csv(args.csv, low_memory=False)
        cols = detect_columns(df_full)
        pid_col, org_col, ab_col = cols['patient_id'], cols['organism'], cols['antibiotic']
        if any(v is None for v in [pid_col, org_col, ab_col]):
            raise ValueError('Missing key columns for listing (patient_id, organism, antibiotic)')

        subset = df_full[[pid_col, org_col, ab_col]].copy()
        subset = subset.dropna()

        # Apply filters if provided
        if args.patient_id is not None:
            subset = subset[subset[pid_col].astype(str) == str(args.patient_id)]
        if args.organism is not None:
            # Exact or case-insensitive match
            mask_exact = subset[org_col].astype(str) == str(args.organism)
            mask_ci = subset[org_col].astype(str).str.lower() == str(args.organism).lower()
            subset = subset[mask_exact | mask_ci]
        if args.antibiotic is not None:
            mask_exact = subset[ab_col].astype(str) == str(args.antibiotic)
            mask_ci = subset[ab_col].astype(str).str.lower() == str(args.antibiotic).lower()
            subset = subset[mask_exact | mask_ci]
        if args.contains is not None:
            q = str(args.contains).lower()
            subset = subset[subset[org_col].astype(str).str.lower().str.contains(q) |
                            subset[ab_col].astype(str).str.lower().str.contains(q)]

        # Print summary and examples
        print('=== Summary ===')
        print(f"Rows: {len(df_full):,}")
        print(f"Unique patients: {df_full[pid_col].nunique():,}")
        print(f"Unique organisms: {df_full[org_col].nunique():,}")
        print(f"Unique antibiotics: {df_full[ab_col].nunique():,}")

        print('\n=== Top patient IDs (by count) ===')
        print(df_full[pid_col].value_counts().head(10).to_string())

        print('\n=== Sample organisms ===')
        print(pd.Series(sorted(df_full[org_col].astype(str).unique())[:10]).to_string(index=False))

        print('\n=== Sample antibiotics ===')
        print(pd.Series(sorted(df_full[ab_col].astype(str).unique())[:10]).to_string(index=False))

        print('\n=== Matching rows preview ===')
        print(subset.head(20).to_string(index=False))

        print('\nTip: Use values exactly as shown above with --patient-id, --organism, and --antibiotic in infer/suggest modes.')

    else:
        # infer mode for single triple
        with open(os.path.join(args.models_dir, 'dl_cat_maps.pkl'), 'rb') as f:
            cat_maps = pickle.load(f)
        with open(os.path.join(args.models_dir, 'dl_num_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(args.models_dir, 'dl_feature_cols.json'), 'r') as f:
            meta = json.load(f)
        cat_cols = meta['cat_cols']
        num_cols = meta['num_cols']
        cols = meta['resolved_cols']

        df_full = pd.read_csv(args.csv, low_memory=False)
        cat_cards = [len(cat_maps[c]) for c in cat_cols]
        model = ARMDTabularNet(cat_cards, num_dim=len(num_cols))
        model.load_state_dict(torch.load(os.path.join(args.models_dir, 'armd_tabular_net.pt'), map_location=args.device))
        device = torch.device(args.device)
        model.to(device)

        if args.patient_id is None or args.organism is None or args.antibiotic is None:
            raise ValueError('--patient-id, --organism, and --antibiotic are required for infer mode')

        # Choose a row matching the triple
        pid_col, org_col, ab_col = cols['patient_id'], cols['organism'], cols['antibiotic']
        sub = df_full[(df_full[pid_col].astype(str) == str(args.patient_id)) &
                      (df_full[org_col].astype(str) == str(args.organism)) &
                      (df_full[ab_col].astype(str) == str(args.antibiotic))]
        if sub.empty:
            raise ValueError('No matching row found for given triple')
        row = sub.iloc[0]

        # Encode
        cats = [cat_maps[c].get(str(row.get(c, 'NA')), 0) for c in cat_cols]
        nums = [row.get(c, np.nan) for c in num_cols]
        nums = [v if pd.notna(v) else 0.0 for v in nums]
        nums_scaled = scaler.transform(np.array(nums, dtype=np.float32).reshape(1, -1))

        with torch.no_grad():
            cat_list = [torch.tensor([cats[i]], dtype=torch.int64).to(device) for i in range(len(cat_cols))]
            num_tensor = torch.tensor(nums_scaled[0], dtype=torch.float32).unsqueeze(0).to(device)
            cls_logit, reg_out = model(cat_list, num_tensor)
            prob_resistant = torch.sigmoid(cls_logit).cpu().item()
            time_to_resistance = float(reg_out.cpu().item())

        print(json.dumps({
            'patient_id': args.patient_id,
            'organism': args.organism,
            'antibiotic': args.antibiotic,
            'prob_resistant': prob_resistant,
            'prob_susceptible': 1.0 - prob_resistant,
            'time_to_resistance_days': time_to_resistance
        }, indent=2))


if __name__ == '__main__':
    main()
