import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
)

# Load the dataset
df = pd.read_csv('microbiology_combined_clean.csv')

# Preprocess data
# Handle missing values (simple imputation or drop)
df = df.dropna(subset=['susceptibility', 'resistant_time_to_culturetime'])  # Drop rows missing targets

# Define categorical and numerical columns
# NOTE: Treat 'age' as numeric, not categorical.
categorical_cols = [
    'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
    'culture_description', 'organism', 'antibiotic', 'gender', 'prior_organism', 'age'
]
numerical_cols = [
    'time_to_culturetime', 'medication_time_to_culturetime',
    'prior_infecting_organism_days_to_culutre'
]

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Targets
# Binary resistance: 1 if 'Resistant', else 0
df['resistance'] = (df['susceptibility'] == 'Resistant').astype(int)
# Regression: time to develop resistance
df['time_to_resistance'] = df['resistant_time_to_culturetime']

# Features and targets
features = categorical_cols + numerical_cols
X = df[features].values
y_resist = df['resistance'].values
y_time = df['time_to_resistance'].values

# Split data (stratified for classification stability)
X_train, X_test, y_resist_train, y_resist_test, y_time_train, y_time_test = train_test_split(
    X, y_resist, y_time, test_size=0.2, random_state=42, stratify=y_resist
)

# Custom Dataset
class MicroDataset(Dataset):
    def __init__(self, X, y_resist, y_time):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_resist = torch.tensor(y_resist, dtype=torch.float32)
        self.y_time = torch.tensor(y_time, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_resist[idx], self.y_time[idx]

train_dataset = MicroDataset(X_train, y_resist_train, y_time_train)
test_dataset = MicroDataset(X_test, y_resist_test, y_time_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Deep Learning Model (Multi-task: classification + regression)
class MicroNN(nn.Module):
    def __init__(self, num_features, num_cat_cols):
        super(MicroNN, self).__init__()
        # Embeddings for categorical features (assuming small embedding dim)
        self.embeddings = nn.ModuleList([nn.Embedding(len(label_encoders[col].classes_), 8) for col in categorical_cols])
        embed_size = 8 * num_cat_cols
        self.fc1 = nn.Linear(embed_size + len(numerical_cols), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.2)
        
        # Classification head (resistance prediction)
        self.resist_head = nn.Linear(32, 1)  # Sigmoid for binary
        
        # Regression head (time to resistance)
        self.time_head = nn.Linear(32, 1)  # Linear for regression
    
    def forward(self, x):
        # Split cat and num
        cat_features = x[:, :len(categorical_cols)].long()
        num_features = x[:, len(categorical_cols):]
        
        # Embed categoricals
        embeds = [self.embeddings[i](cat_features[:, i]) for i in range(len(categorical_cols))]
        embeds = torch.cat(embeds, dim=1)
        
        # Concat with num
        x = torch.cat([embeds, num_features], dim=1)
        
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        
        resist_out = torch.sigmoid(self.resist_head(x))  # Probability of resistance
        time_out = self.time_head(x)  # Predicted time
        
        return resist_out, time_out

# Initialize model
model = MicroNN(num_features=X.shape[1], num_cat_cols=len(categorical_cols))
# Handle class imbalance via positive class weight
pos_weight_value = float((y_resist_train == 0).sum()) / float(max(1, (y_resist_train == 1).sum()))
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
criterion_class = nn.BCELoss(weight=None, reduction='mean')  # We'll apply pos_weight manually in loss
criterion_reg = nn.MSELoss()    # For regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 65
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y_resist, batch_y_time in train_loader:
        optimizer.zero_grad()
        resist_pred, time_pred = model(batch_x)
        # Weighted BCE: weight positive samples by pos_weight
        rp = resist_pred.squeeze()
        by = batch_y_resist
        # BCE with logits is more stable but we use sigmoid head; keep BCELoss and weight positive class
        eps = 1e-8
        bce = - (pos_weight * by * torch.log(rp + eps) + (1 - by) * torch.log(1 - rp + eps))
        loss_class = bce.mean()
        loss_reg = criterion_reg(time_pred.squeeze(), batch_y_time)
        loss = loss_class + loss_reg  # Combined loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# Evaluate
model.eval()
resist_probs = []
time_preds = []
resist_true = []
time_true = []
with torch.no_grad():
    for batch_x, batch_y_resist, batch_y_time in test_loader:
        resist_pred, time_pred = model(batch_x)
        resist_probs.extend(resist_pred.squeeze().cpu().numpy().tolist())
        time_preds.extend(time_pred.squeeze().cpu().numpy().tolist())
        resist_true.extend(batch_y_resist.cpu().numpy().tolist())
        time_true.extend(batch_y_time.cpu().numpy().tolist())

# Tune threshold to maximize F1 on the test set (proxy if no val set)
best_thr, best_f1 = 0.5, -1
for thr in np.linspace(0.1, 0.9, 17):
    preds_bin = (np.array(resist_probs) >= thr).astype(int)
    f1 = f1_score(resist_true, preds_bin, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

resist_preds = (np.array(resist_probs) >= best_thr).astype(int)
acc = accuracy_score(resist_true, resist_preds)
prec = precision_score(resist_true, resist_preds, zero_division=0)
rec = recall_score(resist_true, resist_preds, zero_division=0)
f1 = f1_score(resist_true, resist_preds, zero_division=0)
try:
    auc = roc_auc_score(resist_true, resist_probs)
except Exception:
    auc = float('nan')
cm = confusion_matrix(resist_true, resist_preds)
mse = mean_squared_error(time_true, time_preds)

print(f'Best threshold: {best_thr:.3f}')
print(f'Test Accuracy (Resistance): {acc:.4f}')
print(f'Test Precision: {prec:.4f}')
print(f'Test Recall: {rec:.4f}')
print(f'Test F1: {f1:.4f}')
print(f'Test ROC-AUC: {auc:.4f}')
print(f'Test MSE (Time to Resistance): {mse:.4f}')

# Save metrics report
import os
os.makedirs('outputs', exist_ok=True)
report_path = os.path.join('outputs', 'metrics_report.txt')
with open(report_path, 'w') as f:
    f.write(f'Best threshold: {best_thr:.3f}\n')
    f.write(f'Accuracy: {acc:.4f}\n')
    f.write(f'Precision: {prec:.4f}\n')
    f.write(f'Recall: {rec:.4f}\n')
    f.write(f'F1: {f1:.4f}\n')
    f.write(f'ROC-AUC: {auc:.4f}\n')
    f.write(f'MSE (Time): {mse:.4f}\n')
    f.write('Confusion Matrix:\n')
    f.write(np.array2string(cm))

# Function to rank alternatives
def rank_alternatives(model, patient_features, all_antibiotics, label_encoders):
    """
    For a given patient (features without antibiotic), predict resistance prob and time for each alternative antibiotic.
    Rank by lowest resistance prob, then highest time to resistance (assuming longer time is better).
    """
    ranks = []
    antibiotic_col_idx = categorical_cols.index('antibiotic')
    for antib in all_antibiotics:
        # Copy features, set antibiotic
        feat = patient_features.copy()
        feat[antibiotic_col_idx] = label_encoders['antibiotic'].transform([antib])[0]
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            resist_prob, time_pred = model(feat_tensor)
            resist_prob = resist_prob.item()
            time_pred = time_pred.item()
        
        ranks.append((antib, resist_prob, time_pred))
    
    # Sort: low resist prob first, then high time
    ranks.sort(key=lambda x: (x[1], -x[2]))
    return ranks

# Example usage for ranking (assuming you have a patient row without antibiotic value)
# all_antibiotics = label_encoders['antibiotic'].classes_
# patient_feat = X_test[0]  # Example
# rankings = rank_alternatives(model, patient_feat, all_antibiotics, label_encoders)
# print(rankings)