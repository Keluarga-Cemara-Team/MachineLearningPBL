"""
Script Pelatihan & Ekspor Model Terbaik AntiBurnout (Standar 50.000 Data Bebas Leakage)
Berdasarkan Grand Comparison Final dari notebook PBL_Perbandingan_Model_Terbaik.ipynb:

- Champion Model : Logistic Regression L2 (Ridge, C=0.1) - Akurasi: ~81.67%, F1-Macro: ~0.7982
- Alternatif 1   : Random Forest GINI (n_estimators=100) - Akurasi: ~81.45%, F1-Macro: ~0.7959
- Alternatif 2   : Multinomial Naive Bayes (MinMaxScaler) - Akurasi: ~80.63%, F1-Macro: ~0.7826

Standar Anti-Leakage:
1. Menggunakan penuh 50.000 baris dataset (40.000 latih / 10.000 uji, tanpa sampling 5.000).
2. Menghapus kolom bocor 'Mental_Health_Status', 'Burnout_Score', dan 'AI_Wellness_Recommendation' (30 fitur bersih).
3. train_test_split dilakukan SEBELUM SimpleImputer, StandardScaler, dan MinMaxScaler.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent

# 1. Lokasi dataset (mencari otomatis di beberapa path standar proyek)
candidate_paths = [
    BASE_DIR.parent / 'LogisticRegression' / 'burnout_dataset1.csv',
    BASE_DIR.parent / 'NaiveBayes' / 'burnout_dataset1.csv',
    BASE_DIR.parent / 'burnout_dataset1.csv',
    BASE_DIR.parent.parent / 'dataset' / 'mental_health_burnout_prediction_dataset.csv',
]
dataset_path = next((p for p in candidate_paths if p.exists()), None)
if dataset_path is None:
    raise FileNotFoundError(
        "File dataset burnout_dataset1.csv tidak ditemukan di direktori proyek!"
    )

print("=" * 75)
print(f">> [Trainer] Membaca dataset penuh dari: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f">> [Trainer] Total baris dataset: {len(df)} (tanpa sampling)")

# 2. Seleksi kolom fitur (30 fitur bersih, tanpa target & tanpa fitur bocor Mental_Health_Status)
excluded_numeric = ['Person_ID', 'Burnout_Score']
excluded_categorical = [
    'Gender',
    'Country',
    'Occupation',
    'AI_Wellness_Recommendation',
    'Burnout_Risk',
    'Education_Level',
    'Employment_Status',
    'Mental_Health_Status',
]

numeric_columns = list(
    df.select_dtypes(include='number').columns.drop(excluded_numeric, errors='ignore')
)
categorical_columns = list(
    df.select_dtypes(include=['object', 'category', 'string']).columns.drop(
        excluded_categorical, errors='ignore'
    )
)
selected_columns = numeric_columns + categorical_columns
print(f">> [Trainer] Jumlah fitur terpilih: {len(selected_columns)} ({len(numeric_columns)} numerik + {len(categorical_columns)} kategorikal)")

# 3. Encoding kategorikal standar (tanpa Mental_Health_Status)
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
}

target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
inv_target_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}

X = df[selected_columns].copy()
y = df['Burnout_Risk'].map(target_mapping)

for col, mapping in encoding_maps.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

# 4. Split data mentah TERLEBIH DAHULU (bebas preprocessing data leakage)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f">> [Trainer] Split data: {len(X_train_raw)} train / {len(X_test_raw)} test")

# 5. Fit Imputer & Scalers HANYA pada X_train
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

mean_values = {col: float(val) for col, val in zip(selected_columns, imputer.statistics_)}

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_imp)
X_test_std = scaler.transform(X_test_imp)

minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train_imp)
X_test_mm = minmax_scaler.transform(X_test_imp)


def _eval_model(clf, X_te, y_te):
    y_pred = clf.predict(X_te)
    return {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_te, y_pred, average='macro')), 4),
        "recall_macro": round(float(recall_score(y_te, y_pred, average='macro')), 4),
        "f1_score_macro": round(float(f1_score(y_te, y_pred, average='macro')), 4),
    }


# 6. Pelatihan Champion Model: Logistic Regression L2 (Ridge)
print(">> [Trainer] Melatih Champion Model (Logistic Regression L2, C=0.1)...")
model_lr = LogisticRegression(
    penalty='l2', solver='lbfgs', C=0.1, max_iter=1000, random_state=42
)
model_lr.fit(X_train_std, y_train)
metrics_lr = _eval_model(model_lr, X_test_std, y_test)
print(f"   [OK] Logistic Regression L2 -> Akurasi: {metrics_lr['accuracy']*100:.2f}% | F1-Macro: {metrics_lr['f1_score_macro']:.4f}")

# 7. Pelatihan Alternatif 1: Random Forest GINI
print(">> [Trainer] Melatih Alternatif 1 (Random Forest GINI, n_estimators=100)...")
model_rf = RandomForestClassifier(
    criterion='gini', n_estimators=100, random_state=42, n_jobs=-1
)
model_rf.fit(X_train_std, y_train)
metrics_rf = _eval_model(model_rf, X_test_std, y_test)
print(f"   [OK] Random Forest GINI     -> Akurasi: {metrics_rf['accuracy']*100:.2f}% | F1-Macro: {metrics_rf['f1_score_macro']:.4f}")

# 8. Pelatihan Alternatif 2: Multinomial Naive Bayes (MinMaxScaler)
print(">> [Trainer] Melatih Alternatif 2 (Multinomial Naive Bayes, MinMaxScaler)...")
model_nb = MultinomialNB(alpha=1.0)
model_nb.fit(X_train_mm, y_train)
metrics_nb = _eval_model(model_nb, X_test_mm, y_test)
print(f"   [OK] Naive Bayes Multinomial -> Akurasi: {metrics_nb['accuracy']*100:.2f}% | F1-Macro: {metrics_nb['f1_score_macro']:.4f}")

# 9. Ekspor artefak model ke models/
models_dir = BASE_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model_lr, models_dir / 'best_model.pkl')
joblib.dump(model_rf, models_dir / 'random_forest_gini.pkl')
joblib.dump(model_nb, models_dir / 'naive_bayes_multinomial.pkl')
joblib.dump(imputer, models_dir / 'imputer.pkl')
joblib.dump(scaler, models_dir / 'scaler.pkl')
joblib.dump(minmax_scaler, models_dir / 'minmax_scaler.pkl')

# Hapus artefak lama yang tidak lagi dipakai agar folder models/ bersih
legacy_lr_l1 = models_dir / 'logistic_regression_l1.pkl'
if legacy_lr_l1.exists():
    legacy_lr_l1.unlink()

metadata = {
    "champion_model": "Logistic Regression L2 (Ridge, C=0.1)",
    "comparison_source": "PBL_Perbandingan_Model_Terbaik.ipynb",
    "dataset_info": {
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train_raw)),
        "test_rows": int(len(X_test_raw)),
        "total_features": len(selected_columns),
        "excluded_leakage_columns": ["Mental_Health_Status", "Burnout_Score", "AI_Wellness_Recommendation"],
        "preprocessing_leakage_free": True,
    },
    "evaluation_metrics": {
        "logistic_regression_l2": {
            "name": "Logistic Regression L2 (Ridge)",
            **metrics_lr,
            "scaler": "StandardScaler",
            "is_champion": True,
        },
        "random_forest_gini": {
            "name": "Random Forest (GINI)",
            **metrics_rf,
            "scaler": "StandardScaler",
            "is_champion": False,
        },
        "naive_bayes_multinomial": {
            "name": "Multinomial Naive Bayes (MultinomialNB)",
            **metrics_nb,
            "scaler": "MinMaxScaler",
            "is_champion": False,
        },
    },
    "grand_comparison_50k": [
        {"rank": 1, "model": "Logistic Regression L2", "accuracy": 0.8167, "f1_score_macro": 0.7982, "cv_accuracy": 0.8143},
        {"rank": 2, "model": "SVM RBF", "accuracy": 0.8136, "f1_score_macro": 0.7964, "cv_accuracy": 0.8112},
        {"rank": 3, "model": "Random Forest GINI", "accuracy": 0.8145, "f1_score_macro": 0.7959, "cv_accuracy": 0.8124},
        {"rank": 4, "model": "Naive Bayes Multinomial", "accuracy": 0.8063, "f1_score_macro": 0.7826, "cv_accuracy": 0.8050},
        {"rank": 5, "model": "KNN Manhattan", "accuracy": 0.7831, "f1_score_macro": 0.7628, "cv_accuracy": 0.7851},
    ],
    "feature_columns": selected_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "encoding_maps": encoding_maps,
    "target_mapping": target_mapping,
    "inv_target_mapping": inv_target_mapping,
    "mean_imputation": mean_values,
}

with open(models_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("=" * 75)
print(f">> [Trainer] Seluruh artefak 50.000 data berhasil diekspor ke: {models_dir}")
print("   - best_model.pkl (Champion: Logistic Regression L2)")
print("   - random_forest_gini.pkl (Alternatif 1: Random Forest GINI)")
print("   - naive_bayes_multinomial.pkl (Alternatif 2: MultinomialNB)")
print("   - imputer.pkl (SimpleImputer mean fit on X_train)")
print("   - scaler.pkl (StandardScaler fit on X_train)")
print("   - minmax_scaler.pkl (MinMaxScaler fit on X_train)")
print("   - model_metadata.json (30 Fitur Bersih, Encoding & Metrik 50k)")
print("=" * 75)
