# AntiBurnout Machine Learning Repository & Production API Wrapper

Repository Project AntiBurnout untuk evaluasi, grand comparison, dan deployment model Machine Learning dalam deteksi dini risiko burnout mahasiswa.

**Kelompok 4 | TI-3H:**
- Muhammad Anka Reza (10)
- Esatovin Ebenhaezer Victoria (05)
- Louis Judia B Sinaga (09)
- Raditya Zandra Fadhillah (19)
- Sandy Kurniawan (24)

---

## 1. Hasil Grand Comparison Model Terbaik (50.000 Data, Bebas Data Leakage)
Berdasarkan analisis komprehensif pada notebook [`PBL_Perbandingan_Model_Terbaik.ipynb`](./PBL_Perbandingan_Model_Terbaik.ipynb) menggunakan seluruh **50.000 baris dataset** (40.000 latih / 10.000 uji), **30 fitur bersih** (tanpa kolom bocor `Mental_Health_Status`), dan *preprocessing* bebas *data leakage*:

| Peringkat | Algoritma & Varian | Akurasi Data Uji | F1-Score Macro | 5-Fold CV Mean | Waktu Latih | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Logistic Regression (L2 Ridge, C=0.1)** | **81.67%** | **0.7982** | **81.43%** | 56.44 s | **Model Jawara (Champion API)** |
| 2 | **SVM (Kernel RBF)** | 81.36% | 0.7964 | 81.12% | 85.80 s | Evaluasi Kernel |
| 3 | **Random Forest (Kriteria GINI)** | 81.45% | 0.7959 | 81.24% | 5.89 s | Alternatif 1 (API) |
| 4 | **Naive Bayes (MultinomialNB, MinMaxScaler)** | 80.63% | 0.7826 | 80.50% | **0.0312 s** | Alternatif 2 (API) |
| 5 | **KNN (Metrik Manhattan)** | 78.31% | 0.7628 | 78.51% | **0.0312 s** | Evaluasi Jarak |

### Mengapa Logistic Regression L2 Terpilih sebagai Champion API?
1. **Akurasi & F1-Score Tertinggi**: Memimpin pada data bersih 50.000 baris dengan Akurasi **81.67%** dan F1-Score Macro **0.7982**.
2. **Inferensi Instan & Ringan**: Prediksi real-time hanya berupa perkalian bobot linear (< 0.5 ms per request) dengan ukuran file `.pkl` sangat kecil.
3. **Explainable AI**: Koefisien fitur linear mudah diinterpretasikan untuk menjelaskan kontribusi faktor risiko burnout mahasiswa.

---

## 2. Arsitektur Integrasi End-to-End

```
+-------------------+           +-------------------+           +-----------------------+
|    MobilePBL      |           |      WebPBL       |           |  MachineLearningPBL   |
| (Flutter Android/ |  HTTP     |  (Flask Clean     |  HTTP     | (FastAPI Microservice |
|   iOS / Web)      | --------> |   Architecture)   | --------> |   Port 8000)          |
|                   |  JSON     |   Port 5000       |  JSON     |                       |
+-------------------+           +-------------------+           +-----------------------+
         |                                |                                 |
         |                                v                                 v
         |                      +-------------------+            +---------------------+
         +--------------------> |  Cloud Firestore  |            |   best_model.pkl    |
           (Simpan/Ambil Riwayat|    (Firebase)     |            |  (LogReg L2 81.7%) |
            Asesmen Mahasiswa)  +-------------------+            +---------------------+
```

---

## 3. Cara Menjalankan API Server

### Opsi A: Menggunakan Batch Script (Windows)
Cukup klik ganda file:
```cmd
run_ml.bat
```
*(atau masuk ke folder `api/` dan jalankan `run_ml.bat`)*

### Opsi B: Menggunakan Perintah Terminal / Python Laragon
```bash
# Jalankan server API dari folder api (Port 8000)
cd c:\laragon\www\AntiBurnout\MachineLearningPBL\api
C:\laragon\bin\python\python-3.10\python.exe run.py
```

Server akan aktif di:
- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Docs**: `http://127.0.0.1:8000/redoc`

---

## 4. Spesifikasi Endpoint

### A. Health Check & Info
- **URL**: `GET /` atau `GET /health`
- **Respon**:
```json
{
  "status": "online",
  "service": "AntiBurnout MachineLearningPBL API",
  "api_version": "2.0.0",
  "champion_model": "Logistic Regression L2 (Ridge, C=0.1)",
  "accuracy": "81.67%",
  "f1_score_macro": 0.7982,
  "dataset_rows": 50000,
  "total_features": 30,
  "leakage_free": true
}
```

### B. Daftar Model Perbandingan
- **URL**: `GET /models`
- **Respon**: Ringkasan seluruh metrik evaluasi 5 model dari Grand Comparison 50.000 data.

### C. Prediksi Risiko Burnout (Utama)
- **URL**: `POST /predict`
- **Query Param Opsional**: `?model_choice=champion` (atau `logistic_regression`, `random_forest`, `naive_bayes`)
- **Headers**: `Content-Type: application/json`
- **Contoh Request Payload (Bisa Lengkap / Parsial)**:
```json
{
  "work_hours_per_week": 55.0,
  "sleep_hours": 4.5,
  "sleep_quality": "Poor",
  "stress_level": "High",
  "screen_time_hours": 10.0,
  "coffee_cups_per_day": 4,
  "meditation_minutes": 5.0,
  "mood_score": 30.0,
  "chronic_stress": "Yes"
}
```
- **Contoh Response JSON**:
```json
{
  "status": "success",
  "is_simulation": false,
  "model_name": "Logistic Regression L2 (Ridge)",
  "burnout_score": 98.0,
  "burnout_risk": "High",
  "burnout_level": "High (Logistic Regression L2 (Ridge))",
  "mental_health_status": "Critical",
  "ai_wellness_recommendation": "Tingkat risiko burnout tinggi terdeteksi oleh AI. Segera kurangi jam kerja/belajar, ambil jeda pemulihan, dan jadwalkan sesi konseling dengan profesional.",
  "badge_class": "danger",
  "probabilities": {
    "low": 0.0,
    "moderate": 0.0001,
    "high": 0.9999
  }
}
```

---

## 5. Struktur Folder Microservice
```text
MachineLearningPBL/
├── api/                            # 🚀 Folder Khusus Microservice API
│   ├── app.py                      # FastAPI Application Controller & Preprocessing
│   ├── run.py                      # Script Runner Uvicorn
│   ├── run_ml.bat                  # Windows Quick Launcher (di dalam folder api)
│   ├── requirements.txt            # Spesifikasi Dependencies Python
│   ├── train_best_models.py        # Pipeline Pelatihan & Ekspor Model
│   └── models/                     # Artefak Model Serialisasi (50.000 Data)
│       ├── best_model.pkl          # Champion Model (Logistic Regression L2)
│       ├── random_forest_gini.pkl  # Model Alternatif 1 (Random Forest GINI)
│       ├── naive_bayes_multinomial.pkl # Model Alternatif 2 (MultinomialNB)
│       ├── imputer.pkl             # SimpleImputer (mean fit on X_train)
│       ├── scaler.pkl              # StandardScaler (fit on X_train)
│       ├── minmax_scaler.pkl       # MinMaxScaler (fit on X_train untuk NB)
│       └── model_metadata.json     # Metadata 30 Fitur Bersih & Metrik 50k
├── run_ml.bat                      # 🚀 Quick Launcher Utama (memanggil api/run.py)
├── PBL_Perbandingan_Model_Terbaik.ipynb # Notebook Riset & Evaluasi Utama
├── NaiveBayes/                     # Riset & Varian Model Naive Bayes
├── RandomForest/                   # Riset & Varian Model Random Forest
├── SVM/                            # Riset & Varian Model SVM
├── LogisticRegression/             # Riset & Varian Model Logistic Regression
└── KNN/                            # Riset & Varian Model KNN
```
