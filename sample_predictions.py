import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def get_feature_cols(df: pd.DataFrame):
    feature_cols = [
        'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
        'culture_description', 'organism', 'antibiotic', 'age', 'gender', 'prior_organism',
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility'
    ]
    return [c for c in feature_cols if c in df.columns]


def load_model():
    pkl = os.path.join('models', 'catboost_resistance_new.pkl')
    cbm = os.path.join('models', 'catboost_resistance_new.cbm')
    if os.path.exists(pkl):
        import pickle
        with open(pkl, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(cbm):
        m = CatBoostClassifier()
        m.load_model(cbm)
        return m
    else:
        raise FileNotFoundError('Model not found. Run train_catboost.py first.')


def main():
    df = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)
    feature_cols = get_feature_cols(df)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    gender_is_numeric = ('gender' in df.columns) and pd.api.types.is_numeric_dtype(df['gender'])
    cat_features_idx = [i for i, c in enumerate(feature_cols) if c not in numeric_cols]

    # 15 sample test sets (you can paste these into the Flask form too)
    samples = [
        {"id": 1,  "medication_category": "CIP", "medication_name": "Ciprofloxacin", "antibiotic_class": "Fluoroquinolones", "ordering_mode": "Inpatient", "culture_description": "Urine culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Ciprofloxacin", "age": 68, "gender": "Female", "prior_organism": "ESCHERICHIA COLI", "was_positive": 1, "time_to_culturetime": 0, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 30, "implied_susceptibility": "Susceptible"},
        {"id": 2,  "medication_category": "PIP", "medication_name": "Piperacillin/Tazobactam", "antibiotic_class": "Beta-lactam/BLI", "ordering_mode": "Inpatient", "culture_description": "Blood culture", "organism": "KLEBSIELLA PNEUMONIAE", "antibiotic": "Piperacillin/Tazobactam", "age": 72, "gender": "Male", "prior_organism": "KLEBSIELLA PNEUMONIAE", "was_positive": 1, "time_to_culturetime": 1, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 14, "implied_susceptibility": "Intermediate"},
        {"id": 3,  "medication_category": "NIT", "medication_name": "Nitrofurantoin", "antibiotic_class": "Nitrofurans", "ordering_mode": "Outpatient", "culture_description": "Urine culture", "organism": "PROTEUS MIRABILIS", "antibiotic": "Nitrofurantoin", "age": 55, "gender": "Male", "prior_organism": "PROTEUS MIRABILIS", "was_positive": 0, "time_to_culturetime": 2, "medication_time_to_culturetime": 1, "prior_infecting_organism_days_to_culutre": 45, "implied_susceptibility": "Resistant"},
        {"id": 4,  "medication_category": "LEV", "medication_name": "Levofloxacin", "antibiotic_class": "Fluoroquinolones", "ordering_mode": "Inpatient", "culture_description": "Urine culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Levofloxacin", "age": 64, "gender": "Female", "prior_organism": "ESCHERICHIA COLI", "was_positive": 1, "time_to_culturetime": 0, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 60, "implied_susceptibility": "Susceptible"},
        {"id": 5,  "medication_category": "CEF", "medication_name": "Cefazolin", "antibiotic_class": "Cephalosporins", "ordering_mode": "Inpatient", "culture_description": "Blood culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Cefazolin", "age": 48, "gender": "Male", "prior_organism": "ESCHERICHIA COLI", "was_positive": 1, "time_to_culturetime": 0, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 20, "implied_susceptibility": "Intermediate"},
        {"id": 6,  "medication_category": "VAN", "medication_name": "Vancomycin", "antibiotic_class": "Glycopeptides", "ordering_mode": "Inpatient", "culture_description": "Blood culture", "organism": "ENTEROCOCCUS SPECIES", "antibiotic": "Vancomycin", "age": 70, "gender": "Male", "prior_organism": "ENTEROCOCCUS SPECIES", "was_positive": 1, "time_to_culturetime": 1, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 10, "implied_susceptibility": "Susceptible"},
        {"id": 7,  "medication_category": "AMO", "medication_name": "Amoxicillin/Clavulanic Acid", "antibiotic_class": "Penicillins", "ordering_mode": "Outpatient", "culture_description": "Urine culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Amoxicillin/Clavulanic Acid", "age": 36, "gender": "Female", "prior_organism": "ESCHERICHIA COLI", "was_positive": 0, "time_to_culturetime": 3, "medication_time_to_culturetime": 2, "prior_infecting_organism_days_to_culutre": 90, "implied_susceptibility": "Susceptible"},
        {"id": 8,  "medication_category": "TRI", "medication_name": "Trimethoprim/Sulfamethoxazole", "antibiotic_class": "Sulfonamides", "ordering_mode": "Outpatient", "culture_description": "Urine culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Trimethoprim/Sulfamethoxazole", "age": 29, "gender": "Female", "prior_organism": "ESCHERICHIA COLI", "was_positive": 0, "time_to_culturetime": 1, "medication_time_to_culturetime": 1, "prior_infecting_organism_days_to_culutre": 30, "implied_susceptibility": "Intermediate"},
        {"id": 9,  "medication_category": "GEN1", "medication_name": "Tobramycin", "antibiotic_class": "Aminoglycosides", "ordering_mode": "Inpatient", "culture_description": "Respiratory culture", "organism": "PSEUDOMONAS AERUGINOSA", "antibiotic": "Tobramycin", "age": 60, "gender": "Male", "prior_organism": "PSEUDOMONAS AERUGINOSA", "was_positive": 1, "time_to_culturetime": 1, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 15, "implied_susceptibility": "Resistant"},
        {"id": 10, "medication_category": "ERT", "medication_name": "Ertapenem", "antibiotic_class": "Carbapenems", "ordering_mode": "Inpatient", "culture_description": "Blood culture", "organism": "KLEBSIELLA PNEUMONIAE", "antibiotic": "Ertapenem", "age": 67, "gender": "Female", "prior_organism": "KLEBSIELLA PNEUMONIAE", "was_positive": 1, "time_to_culturetime": 0, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 7,  "implied_susceptibility": "Susceptible"},
        {"id": 11, "medication_category": "LEV2", "medication_name": "Levofloxacin", "antibiotic_class": "Fluoroquinolones", "ordering_mode": "Inpatient", "culture_description": "Respiratory culture", "organism": "ACINETOBACTER BAUMANNII", "antibiotic": "Levofloxacin", "age": 59, "gender": "Male", "prior_organism": "ACINETOBACTER BAUMANNII", "was_positive": 1, "time_to_culturetime": 2, "medication_time_to_culturetime": 1, "prior_infecting_organism_days_to_culutre": 21, "implied_susceptibility": "Resistant"},
        {"id": 12, "medication_category": "CEF3", "medication_name": "Cefepime", "antibiotic_class": "Cephalosporins", "ordering_mode": "Inpatient", "culture_description": "Blood culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Cefepime", "age": 44, "gender": "Female", "prior_organism": "ESCHERICHIA COLI", "was_positive": 1, "time_to_culturetime": 0, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 40, "implied_susceptibility": "Susceptible"},
        {"id": 13, "medication_category": "MIN", "medication_name": "Minocycline", "antibiotic_class": "Tetracyclines", "ordering_mode": "Outpatient", "culture_description": "Skin/soft tissue culture", "organism": "STAPHYLOCOCCUS AUREUS", "antibiotic": "Minocycline", "age": 34, "gender": "Female", "prior_organism": "STAPHYLOCOCCUS AUREUS", "was_positive": 0, "time_to_culturetime": 1, "medication_time_to_culturetime": 1, "prior_infecting_organism_days_to_culutre": 10, "implied_susceptibility": "Intermediate"},
        {"id": 14, "medication_category": "COL", "medication_name": "Colistin", "antibiotic_class": "Polymyxins", "ordering_mode": "Inpatient", "culture_description": "Respiratory culture", "organism": "KLEBSIELLA PNEUMONIAE", "antibiotic": "Colistin", "age": 75, "gender": "Male", "prior_organism": "KLEBSIELLA PNEUMONIAE", "was_positive": 1, "time_to_culturetime": 0, "medication_time_to_culturetime": 0, "prior_infecting_organism_days_to_culutre": 5,  "implied_susceptibility": "Resistant"},
        {"id": 15, "medication_category": "AMO1", "medication_name": "Amoxicillin/Clavulanic Acid", "antibiotic_class": "Penicillins", "ordering_mode": "Outpatient", "culture_description": "Urine culture", "organism": "ESCHERICHIA COLI", "antibiotic": "Amoxicillin/Clavulanic Acid", "age": 52, "gender": "Female", "prior_organism": "ESCHERICHIA COLI", "was_positive": 0, "time_to_culturetime": 2, "medication_time_to_culturetime": 1, "prior_infecting_organism_days_to_culutre": 30, "implied_susceptibility": "Susceptible"},
    ]

    # Build rows to match dataset feature types
    rows = []
    for s in samples:
        row = {}
        for c in feature_cols:
            if c in s:
                v = s[c]
                if c == 'gender' and gender_is_numeric:
                    row[c] = {"Male": 1, "Female": 2, "Unknown": 0}.get(v, np.nan) if isinstance(v, str) else v
                elif c in numeric_cols:
                    row[c] = float(v)
                else:
                    row[c] = str(v)
            else:
                # Fill missing values
                if c in numeric_cols:
                    row[c] = np.nan
                else:
                    row[c] = 'NA'
        rows.append(row)

    X = pd.DataFrame(rows, columns=feature_cols)
    pool = Pool(X, cat_features=cat_features_idx)
    model = load_model()
    probs = model.predict_proba(pool)[:, 1]
    thr = 0.50
    preds = (probs >= thr).astype(int)

    out_df = pd.DataFrame({
        'id': [s['id'] for s in samples],
        'organism': [s['organism'] for s in samples],
        'antibiotic': [s['antibiotic'] for s in samples],
        'prob_resistant': probs,
        'class': np.where(preds == 1, 'Resistant', 'Non-Resistant')
    })

    os.makedirs('outputs', exist_ok=True)
    out_csv = os.path.join('outputs', 'sample_predictions.csv')
    out_json = os.path.join('outputs', 'sample_predictions.json')
    out_df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(out_df.to_dict(orient='records'), f, indent=2)
    print('Saved:', out_csv)
    print('Saved:', out_json)


if __name__ == '__main__':
    main()
