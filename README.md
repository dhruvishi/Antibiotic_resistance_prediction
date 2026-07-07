# ARMD Antibiotic Resistance Prediction Pipeline

This workspace includes:

- `train_catboost.py`: CatBoost baseline (classification) with `.cbm` and `.pkl` saves and metrics.
- `visualize_model.py`: Generates boxplot, scatterplot, and feature-importance bar chart from the CatBoost model.
- `train_pytorch_armd.py`: PyTorch multitask pipeline with two heads:
  - Classification: probability the organism is resistant to a given antibiotic now.
  - Regression: estimated days until resistance (time-to-resistance).
  - Also provides alternative antibiotic suggestions ranked by higher susceptibility probability and longer time-to-resistance.

## Quick Start

Ensure Python environment is active and dependencies installed (see `requirements.txt`).

### 1) CatBoost baseline

```powershell
cd C:\Users\ASUS\Downloads\doi_10_5061_dryad_jq2bvq8kp__v20251022
python train_catboost.py
```

Visualize CatBoost predictions and importances:

```powershell
python visualize_model.py --model-path models/catboost_resistance_new.pkl --csv microbiology_combined_clean.csv --target-scheme binary_rs --outdir outputs/visualizations
```

### 2) PyTorch multitask training

```powershell
python train_pytorch_armd.py --csv microbiology_combined_clean.csv --epochs 5 --batch-size 512 --device cpu
```

Artifacts:
- Model: `models/armd_tabular_net.pt`
- Preprocessing: `models/dl_cat_maps.pkl`, `models/dl_num_scaler.pkl`, `models/dl_feature_cols.json`
- Metrics: `outputs/metrics_dl.txt`
- Visualizations: `outputs/visualizations_dl_boxplot.png`, `outputs/visualizations_dl_scatter_age.png`, `outputs/visualizations_dl_importance.png`

### 3) Inference for a specific triple

```powershell
python train_pytorch_armd.py --mode infer --csv microbiology_combined_clean.csv --patient-id <PID> --organism <OrganismName> --antibiotic <AntibioticName>
```

### 4) Alternative antibiotic suggestions

```powershell
python train_pytorch_armd.py --mode suggest --csv microbiology_combined_clean.csv --patient-id <PID> --organism <OrganismName>
```

Outputs saved to `outputs/alternative_antibiotics.csv`.

## Notes
- The script auto-detects key columns (patient ID, organism, antibiotic, susceptibility, time-to-resistance) and builds feature sets from available columns.
- Categorical features are embedded; numeric features are standardized.
- Class imbalance handled via `pos_weight` in BCE.
- Permutation importance (AUC drop) estimates numeric feature importance for the DL model.

## Frontend: Manual Entry Predictor (Flask)

Run a simple web form to manually enter features and get a resistance prediction using the CatBoost model:

```powershell
python app.py
```

Then open http://127.0.0.1:5000 and fill in available fields. Leave unknowns empty. The app uses a 0.5 threshold by default.
