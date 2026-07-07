import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template_string
from catboost import CatBoostClassifier, Pool
import torch
import pickle
from sklearn.preprocessing import StandardScaler


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Antibiotic Resistance Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    h1 { margin-bottom: 0.5rem; }
    form { display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 12px; }
    .field { display: flex; flex-direction: column; }
    label { font-weight: 600; margin-bottom: 4px; }
    input, select { padding: 8px; font-size: 14px; }
    .actions { grid-column: 1 / -1; margin-top: 1rem; }
    .result { margin-top: 1.5rem; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
    .resist { color: #b00020; font-weight: bold; }
    .suscept { color: #006400; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Antibiotic Resistance Predictor (CatBoost)</h1>
  <p style="margin:0.5rem 0 1rem;">
    New features: <a href="{{ url_for('dl_infer') }}">DL Inference</a> (probability + time-to-resistance)
    and <a href="{{ url_for('dl_suggest') }}">DL Suggestions</a> (ranked alternative antibiotics).
  </p>
  <p>Enter patient-organism-antibiotic context and relevant features. Leave unknowns empty.</p>
  <form method="post" action="{{ url_for('predict') }}">
    {% for col in feature_cols %}
      <div class="field">
        <label for="{{ col }}">{{ col }}</label>
        {% if col == 'gender' %}
          <select name="gender" id="gender">
            <option value="">-- Select --</option>
            {% for opt in gender_choices %}
              <option value="{{ opt }}" {% if form_values.get('gender','') == opt %}selected{% endif %}>{{ opt }}</option>
            {% endfor %}
          </select>
        {% elif col in numeric_cols %}
          <input type="number" step="any" name="{{ col }}" id="{{ col }}" value="{{ form_values.get(col, '') }}" />
        {% else %}
          <input type="text" name="{{ col }}" id="{{ col }}" value="{{ form_values.get(col, '') }}" />
        {% endif %}
      </div>
    {% endfor %}
    <div class="actions">
      <button type="submit">Predict</button>
    </div>
  </form>

  {% if result %}
  <div class="result">
    <div>Predicted probability of resistance: <strong>{{ '{:.4f}'.format(result.prob) }}</strong></div>
    <div>Status: {% if result.is_resistant %}<span class="resist">Resistant</span>{% else %}<span class="suscept">Non-Resistant</span>{% endif %}</div>
    <div>Threshold used: {{ '{:.2f}'.format(result.threshold) }}</div>
  </div>
  {% endif %}
</body>
</html>
"""


DL_INFER_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>DL Inference: Resistance + Time-to-Resistance</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    .field { margin-bottom: 10px; display: flex; gap: 8px; align-items: center; }
    label { width: 180px; font-weight: 600; }
    input { padding: 8px; font-size: 14px; flex: 1; }
    .result { margin-top: 1.5rem; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
  </style>
  </head>
<body>
  <h1>DL Inference</h1>
  <p>Enter a patient ID, organism, and antibiotic present in the dataset.</p>
  <form method=\"post\">
    <div class=\"field\"><label>Patient ID</label><input type=\"text\" name=\"patient_id\" value=\"{{ form_values.get('patient_id','') }}\" /></div>
    <div class=\"field\"><label>Organism</label><input type=\"text\" name=\"organism\" value=\"{{ form_values.get('organism','') }}\" /></div>
    <div class=\"field\"><label>Antibiotic</label><input type=\"text\" name=\"antibiotic\" value=\"{{ form_values.get('antibiotic','') }}\" /></div>
    <button type=\"submit\">Predict</button>
  </form>
  {% if result %}
  <div class=\"result\">
    <div>Probability resistant: <strong>{{ '{:.4f}'.format(result.prob_resistant) }}</strong></div>
    <div>Probability susceptible: <strong>{{ '{:.4f}'.format(result.prob_susceptible) }}</strong></div>
    <div>Time to resistance (days): <strong>{{ '{:.2f}'.format(result.time_to_resistance_days) }}</strong></div>
  </div>
  {% endif %}
  <p><a href=\"{{ url_for('index') }}\">Back to CatBoost form</a> | <a href=\"{{ url_for('dl_suggest') }}\">Alternative suggestions</a></p>
</body>
</html>
"""


DL_SUGGEST_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>DL Suggestions: Alternative Antibiotics</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    .field { margin-bottom: 10px; display: flex; gap: 8px; align-items: center; }
    label { width: 180px; font-weight: 600; }
    input { padding: 8px; font-size: 14px; flex: 1; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background: #f0f0f0; }
  </style>
  </head>
<body>
  <h1>Alternative Antibiotic Suggestions</h1>
  <p>Enter a patient ID and organism to get ranked alternatives.</p>
  <form method=\"post\">
    <div class=\"field\"><label>Patient ID</label><input type=\"text\" name=\"patient_id\" value=\"{{ form_values.get('patient_id','') }}\" /></div>
    <div class=\"field\"><label>Organism</label><input type=\"text\" name=\"organism\" value=\"{{ form_values.get('organism','') }}\" /></div>
    <button type=\"submit\">Suggest</button>
  </form>
  {% if table_rows %}
  <table>
    <tr><th>Antibiotic</th><th>Prob Susceptible</th><th>Prob Resistant</th><th>Time-to-Resistance (days)</th></tr>
    {% for r in table_rows %}
      <tr>
        <td>{{ r['antibiotic'] }}</td>
        <td>{{ '{:.4f}'.format(r['prob_susceptible']) }}</td>
        <td>{{ '{:.4f}'.format(r['prob_resistant']) }}</td>
        <td>{{ '{:.2f}'.format(r['time_to_resistance_pred_days']) }}</td>
      </tr>
    {% endfor %}
  </table>
  {% endif %}
  <p><a href=\"{{ url_for('index') }}\">Back to CatBoost form</a> | <a href=\"{{ url_for('dl_infer') }}\">DL Inference</a></p>
</body>
</html>
"""


def get_feature_cols(df: pd.DataFrame):
    feature_cols = [
        'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
        'culture_description', 'organism', 'antibiotic', 'age', 'gender', 'prior_organism',
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    return feature_cols


def build_app():
    app = Flask(__name__)

    # Load model
    model = None
    pkl_path = os.path.join('models', 'catboost_resistance_new.pkl')
    cbm_path = os.path.join('models', 'catboost_resistance_new.cbm')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as pf:
            model = pickle.load(pf)
    elif os.path.exists(cbm_path):
        model = CatBoostClassifier()
        model.load_model(cbm_path)
    else:
        raise FileNotFoundError('Model file not found. Train with train_catboost.py first.')

    # Load CSV to infer feature columns and numeric types
    df = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)
    feature_cols = get_feature_cols(df)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_features_idx = [i for i, c in enumerate(feature_cols) if c not in numeric_cols]
    gender_is_numeric = ('gender' in df.columns) and pd.api.types.is_numeric_dtype(df['gender'])
    gender_choices = ['Male', 'Female', 'Unknown']

    # Lazy load DL artifacts
    _dl_loaded = {'model': None, 'cat_maps': None, 'scaler': None, 'meta': None}

    def load_dl():
      if _dl_loaded['model'] is not None:
        return _dl_loaded
      models_dir = 'models'
      with open(os.path.join(models_dir, 'dl_cat_maps.pkl'), 'rb') as f:
        cat_maps = pickle.load(f)
      with open(os.path.join(models_dir, 'dl_num_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
      with open(os.path.join(models_dir, 'dl_feature_cols.json'), 'r') as f:
        meta = pickle.load(f) if f.name.endswith('.pkl') else __import__('json').load(f)
      cat_cols = meta['cat_cols']
      num_cols_dl = meta['num_cols']
      cat_cards = [len(cat_maps[c]) for c in cat_cols]

      # Build model matching training
      class ARMDTabularNet(torch.nn.Module):
        def __init__(self, cat_cardinalities, num_dim, hidden_dim=256, dropout=0.2):
          super().__init__()
          self.embeddings = torch.nn.ModuleList()
          emb_out_dims = []
          for card in cat_cardinalities:
            emb_dim = min(50, (card + 1) // 2)
            self.embeddings.append(torch.nn.Embedding(card + 1, emb_dim))
            emb_out_dims.append(emb_dim)
          total_in = sum(emb_out_dims) + num_dim
          self.backbone = torch.nn.Sequential(
            torch.nn.Linear(total_in, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
          )
          self.cls_head = torch.nn.Linear(256, 1)
          self.reg_head = torch.nn.Linear(256, 1)
        def forward(self, cats, nums):
          if len(self.embeddings) > 0:
            embs = [emb(c) for emb, c in zip(self.embeddings, cats)]
            x = torch.cat([torch.cat(embs, dim=1), nums], dim=1)
          else:
            x = nums
          h = self.backbone(x)
          cls_logit = self.cls_head(h)
          reg_out = self.reg_head(h)
          return cls_logit.squeeze(1), reg_out.squeeze(1)

      model = ARMDTabularNet(cat_cards, num_dim=len(num_cols_dl))
      state_path = os.path.join(models_dir, 'armd_tabular_net.pt')
      model.load_state_dict(torch.load(state_path, map_location='cpu'))
      model.eval()
      _dl_loaded.update({'model': model, 'cat_maps': cat_maps, 'scaler': scaler, 'meta': meta})
      return _dl_loaded

    @app.route('/', methods=['GET'])
    def index():
      return render_template_string(TEMPLATE, feature_cols=feature_cols, numeric_cols=numeric_cols, gender_choices=gender_choices, form_values={}, result=None)

    @app.route('/predict', methods=['POST'])
    def predict():
        # Collect inputs
        form_values = {}
        for c in feature_cols:
            v = request.form.get(c)
            form_values[c] = v if v is not None else ''

        # Build one-row DataFrame, coerce numeric where known
        row = {}
        for c in feature_cols:
          val = form_values[c]
          if c == 'gender':
            if gender_is_numeric:
              # Map UI to dataset numeric coding (assume Male=1, Female=2, Unknown=0)
              row[c] = {'Male': 1, 'Female': 2, 'Unknown': 0}.get(val, np.nan)
            else:
              row[c] = val if val not in (None, '') else 'NA'
          elif c in numeric_cols:
            try:
              row[c] = float(val) if val not in (None, '', 'NA') else np.nan
            except Exception:
              row[c] = np.nan
          else:
            row[c] = val if val not in (None, '') else 'NA'

        X = pd.DataFrame([row], columns=feature_cols)
        pool = Pool(X, cat_features=cat_features_idx)
        prob = float(model.predict_proba(pool)[:, 1][0])
        thr = 0.50  # default threshold; adjust if you have an optimal threshold saved
        is_resistant = prob >= thr

        result = type('Res', (), {'prob': prob, 'threshold': thr, 'is_resistant': is_resistant})
        return render_template_string(TEMPLATE, feature_cols=feature_cols, numeric_cols=numeric_cols, gender_choices=gender_choices, form_values=form_values, result=result)

    @app.route('/dl/infer', methods=['GET', 'POST'])
    def dl_infer():
        form_values = {}
        if request.method == 'POST':
          form_values = {k: request.form.get(k, '') for k in ['patient_id', 'organism', 'antibiotic']}
          dl = load_dl()
          cat_cols = dl['meta']['cat_cols']
          num_cols_dl = dl['meta']['num_cols']
          cat_maps = dl['cat_maps']
          scaler = dl['scaler']
          model = dl['model']
          df_full = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)
          pid_col = dl['meta']['resolved_cols']['patient_id']
          org_col = dl['meta']['resolved_cols']['organism']
          ab_col = dl['meta']['resolved_cols']['antibiotic']
          sub = df_full[(df_full[pid_col].astype(str) == str(form_values['patient_id'])) &
                  (df_full[org_col].astype(str) == str(form_values['organism'])) &
                  (df_full[ab_col].astype(str) == str(form_values['antibiotic']))]
          result = None
          if not sub.empty:
            row = sub.iloc[0]
            cats = [cat_maps[c].get(str(row.get(c, 'NA')), 0) for c in cat_cols]
            nums = [row.get(c, 0.0) if pd.notna(row.get(c, np.nan)) else 0.0 for c in num_cols_dl]
            nums_scaled = scaler.transform(np.array(nums, dtype=np.float32).reshape(1, -1))
            with torch.no_grad():
              cat_list = [torch.tensor([cats[i]], dtype=torch.int64) for i in range(len(cat_cols))]
              num_tensor = torch.tensor(nums_scaled[0], dtype=torch.float32).unsqueeze(0)
              cls_logit, reg_out = model(cat_list, num_tensor)
              prob_resistant = torch.sigmoid(cls_logit).cpu().item()
              time_to_resistance = float(reg_out.cpu().item())
            result = type('DLRes', (), {
              'prob_resistant': prob_resistant,
              'prob_susceptible': 1.0 - prob_resistant,
              'time_to_resistance_days': time_to_resistance,
            })
          return render_template_string(DL_INFER_TEMPLATE, form_values=form_values, result=result)
        return render_template_string(DL_INFER_TEMPLATE, form_values=form_values, result=None)

    @app.route('/dl/suggest', methods=['GET', 'POST'])
    def dl_suggest():
        form_values = {}
        table_rows = None
        if request.method == 'POST':
          form_values = {k: request.form.get(k, '') for k in ['patient_id', 'organism']}
          dl = load_dl()
          cat_cols = dl['meta']['cat_cols']
          num_cols_dl = dl['meta']['num_cols']
          cat_maps = dl['cat_maps']
          scaler = dl['scaler']
          model = dl['model']
          df_full = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)
          pid_col = dl['meta']['resolved_cols']['patient_id']
          org_col = dl['meta']['resolved_cols']['organism']
          ab_col = dl['meta']['resolved_cols']['antibiotic']
          sub = df_full[(df_full[pid_col].astype(str) == str(form_values['patient_id'])) &
                  (df_full[org_col].astype(str) == str(form_values['organism']))]
          rows = []
          with torch.no_grad():
            for _, r in sub.iterrows():
              cats = [cat_maps[c].get(str(r.get(c, 'NA')), 0) for c in cat_cols]
              nums = [r.get(c, 0.0) if pd.notna(r.get(c, np.nan)) else 0.0 for c in num_cols_dl]
              nums_scaled = scaler.transform(np.array(nums, dtype=np.float32).reshape(1, -1))
              cat_list = [torch.tensor([cats[i]], dtype=torch.int64) for i in range(len(cat_cols))]
              num_tensor = torch.tensor(nums_scaled[0], dtype=torch.float32).unsqueeze(0)
              cls_logit, reg_out = model(cat_list, num_tensor)
              prob_resistant = torch.sigmoid(cls_logit).cpu().item()
              rows.append({
                'antibiotic': r[ab_col],
                'prob_resistant': float(prob_resistant),
                'prob_susceptible': float(1.0 - prob_resistant),
                'time_to_resistance_pred_days': float(reg_out.cpu().item()),
              })
          if rows:
            # Rank by higher prob_susceptible and longer time-to-resistance
            table_rows = sorted(rows, key=lambda x: (x['prob_susceptible'], x['time_to_resistance_pred_days']), reverse=True)[:10]
          return render_template_string(DL_SUGGEST_TEMPLATE, form_values=form_values, table_rows=table_rows)
        return render_template_string(DL_SUGGEST_TEMPLATE, form_values=form_values, table_rows=table_rows)

    return app


if __name__ == '__main__':
    app = build_app()
    app.run(host='127.0.0.1', port=5000, debug=False)
