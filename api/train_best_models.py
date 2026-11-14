"""
Script Pelatihan & Ekspor Model Terbaik AntiBurnout
Berdasarkan Grand Comparison dari notebook PBL_Perbandingan_Model_Terbaik.ipynb

Champion Model: Gaussian Naive Bayes (GaussianNB) - Akurasi: 98.90%, F1-Score: 0.9850
Alternatif 1: Logistic Regression L1 (Lasso) - Akurasi: 98.90%, F1-Score: 0.9850
Alternatif 2: Random Forest Gini - Akurasi: 80.40%, F1-Score: 0.7854
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

# 1. Lokasi dataset
dataset_path = BASE_DIR.parent.parent / 'dataset' / 'mental_health_burnout_prediction_dataset.csv'
if not dataset_path.exists():
    dataset_path = BASE_DIR.parent / 'LogisticRegression' / 'burnout_dataset1.csv'

print("=" * 70)
print(f">> [Trainer] Membaca dataset dari: {dataset_path}")
df_full = pd.read_csv(dataset_path)

# Menggunakan sampling n=5000 random_state=42 persis seperti di notebook
df = df_full.sample(n=5000, random_state=42).reset_index(drop=True)

# Seleksi kolom fitur
numeric_columns = list(
    df.select_dtypes(include='number').columns.drop(['Person_ID', 'Burnout_Score'], errors='ignore')
)
categorical_columns = list(
    df.select_dtypes(include=['object', 'category']).columns.drop(
        ['Gender', 'Country', 'Occupation', 'AI_Wellness_Recommendation', 'Burnout_Risk', 'Education_Level', 'Employment_Status'],
        errors='ignore'
    )
)

selected_columns = numeric_columns + categorical_columns

# Encoding mapping standar notebook
encoding_maps = {
    'Remote_Work': {'No': 0, 'Hybrid': 1, 'Yes': 2},
    'Sleep_Quality': {'Excellent': 0, 'Good': 1, 'Average': 2, 'Poor': 3},
    'Stress_Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Exercise_Frequency': {'Daily': 0, 'Weekly': 1, 'Rarely': 2, 'Never': 3},
    'Alcohol_Consumption': {'No': 0, 'Yes': 1},
    'Smoking': {'No': 0, 'Yes': 1},
    'Healthy_Diet': {'No': 1, 'Yes': 0},
    'Chronic_Stress': {'No': 0, 'Yes': 1},
    'Family_History_Mental_Illness': {'No': 0, 'Yes': 1},
    'Therapy_Attendance': {'No': 1, 'Yes': 0},
    'Support_System': {'Poor': 3, 'Average': 2, 'Good': 1, 'Excellent': 0},
    'Mental_Health_Status': {'Healthy': 0, 'Needs Attention': 1, 'Critical': 2}
}

target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
inv_target_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}

X = df[selected_columns].copy()
y = df['Burnout_Risk'].map(target_mapping)

for col, mapping in encoding_maps.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

mean_values = X.mean().to_dict()
X = X.fillna(mean_values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Pelatihan Champion Model: GaussianNB
print(">> [Trainer] Melatih Champion Model (Gaussian Naive Bayes)...")
model_nb = GaussianNB(var_smoothing=1e-9)
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='macro')
print(f"   [OK] GaussianNB -> Akurasi: {acc_nb * 100:.2f}% | F1-Macro: {f1_nb:.4f}")

# 3. Pelatihan Alternatif 1: Logistic Regression L1
print(">> [Trainer] Melatih Alternatif 1 (Logistic Regression L1 Lasso)...")
model_lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')
print(f"   [OK] LogReg L1  -> Akurasi: {acc_lr * 100:.2f}% | F1-Macro: {f1_lr:.4f}")

# 4. Pelatihan Alternatif 2: Random Forest GINI
print(">> [Trainer] Melatih Alternatif 2 (Random Forest GINI)...")
model_rf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
print(f"   [OK] RF GINI    -> Akurasi: {acc_rf * 100:.2f}% | F1-Macro: {f1_rf:.4f}")

# 5. Ekspor artefak model ke models/
models_dir = BASE_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model_nb, models_dir / 'best_model.pkl')
joblib.dump(model_lr, models_dir / 'logistic_regression_l1.pkl')
joblib.dump(model_rf, models_dir / 'random_forest_gini.pkl')
joblib.dump(scaler, models_dir / 'scaler.pkl')

metadata = {
    "champion_model": "Gaussian Naive Bayes (GaussianNB)",
    "comparison_source": "PBL_Perbandingan_Model_Terbaik.ipynb",
    "evaluation_metrics": {
        "gaussian_nb": {
            "name": "Gaussian Naive Bayes",
            "accuracy": round(float(acc_nb), 4),
            "f1_score_macro": round(float(f1_nb), 4),
            "is_champion": True
        },
        "logistic_regression_l1": {
            "name": "Logistic Regression L1 (Lasso)",
            "accuracy": round(float(acc_lr), 4),
            "f1_score_macro": round(float(f1_lr), 4),
            "is_champion": False
        },
        "random_forest_gini": {
            "name": "Random Forest (GINI)",
            "accuracy": round(float(acc_rf), 4),
            "f1_score_macro": round(float(f1_rf), 4),
            "is_champion": False
        }
    },
    "feature_columns": selected_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "encoding_maps": encoding_maps,
    "target_mapping": target_mapping,
    "inv_target_mapping": inv_target_mapping,
    "mean_imputation": mean_values
}

with open(models_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("=" * 70)
print(f">> [Trainer] Seluruh artefak model berhasil diekspor ke: {models_dir}")
print("   - best_model.pkl (Champion: GaussianNB)")
print("   - logistic_regression_l1.pkl (Alternatif 1)")
print("   - random_forest_gini.pkl (Alternatif 2)")
print("   - scaler.pkl (StandardScaler)")
print("   - model_metadata.json (Fitur, Encoding & Metrik)")
print("=" * 70)

if __name__ == "__main__":
    pass
