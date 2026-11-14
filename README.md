# AntiBurnout Machine Learning Repository & Production API Wrapper

Repository Project AntiBurnout untuk evaluasi, grand comparison, dan deployment model Machine Learning dalam deteksi dini risiko burnout mahasiswa.

**Kelompok 4 | TI-3H:**
- Muhammad Anka Reza (10)
- Esatovin Ebenhaezer Victoria (05)
- Louis Judia B Sinaga (09)
- Raditya Zandra Fadhillah (19)
- Sandy Kurniawan (24)

---

## 1. Hasil Grand Comparison Model Terbaik
Berdasarkan analisis komprehensif pada notebook [`PBL_Perbandingan_Model_Terbaik.ipynb`](./PBL_Perbandingan_Model_Terbaik.ipynb):

| Peringkat | Algoritma & Varian | Akurasi Data Uji | F1-Score Macro | 5-Fold CV Mean | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🏆 **1** | **Naive Bayes (GaussianNB)** | **98.90%** | **0.9850** | **98.60% (±0.21%)** | **Model Jawara (Champion)** |
| 🥈 2 | **Logistic Regression (L1 Lasso)** | 98.10% - 98.90% | 0.9738 - 0.9850 | 98.53% (±0.44%) | Alternatif 1 |
| 🥉 3 | **SVM (Kernel RBF)** | 98.90% | 0.9850 | 98.60% (±0.50%) | Alternatif 2 |
| 4 | **Random Forest (Kriteria GINI)** | 80.40% - 98.90% | 0.7854 - 0.9850 | 80.92% (±1.10%) | Alternatif 3 |
| 5 | **KNN (Metrik Manhattan)** | 89.80% | 0.8820 | 88.78% (±0.78%) | Evaluasi Jarak |

### Mengapa Naive Bayes Gaussian Terpilih sebagai Champion API?
1. **Akurasi Puncak**: Akurasi 98.90% dan F1-Score 0.9850 pada data uji.
2. **Latensi Inferensi Super Cepat**: Waktu inferensi < 0.2 ms per request, sangat efisien untuk melayani ratusan request mahasiswa secara serentak.
3. **Ukuran Memori Sangat Ringan**: Ukuran model serialisasi hanya ~2.3 KB, sangat hemat resource server.

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
           (Simpan/Ambil Riwayat|    (Firebase)     |            |  (GaussianNB 98.9%) |
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
  "champion_model": "Gaussian Naive Bayes (GaussianNB)",
  "accuracy": "98.90%",
  "f1_score_macro": 0.985
}
```

### B. Daftar Model Perbandingan
- **URL**: `GET /models`
- **Respon**: Ringkasan seluruh metrik evaluasi model dari Grand Comparison.

### C. Prediksi Risiko Burnout (Utama)
- **URL**: `POST /predict`
- **Query Param Opsional**: `?model_choice=champion` (atau `logistic_regression`, `random_forest`)
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
  "model_name": "Gaussian Naive Bayes (GaussianNB)",
  "burnout_score": 98.0,
  "burnout_risk": "High",
  "burnout_level": "High (Gaussian Naive Bayes (GaussianNB))",
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
│   └── models/                     # Artefak Model Serialisasi
│       ├── best_model.pkl          # Champion Model (GaussianNB)
│       ├── logistic_regression_l1.pkl # Model Alternatif 1
│       ├── random_forest_gini.pkl  # Model Alternatif 2
│       ├── scaler.pkl              # StandardScaler Fitur Dataset
│       └── model_metadata.json     # Metadata Fitur & Metrik Evaluasi
├── run_ml.bat                      # 🚀 Quick Launcher Utama (memanggil api/run.py)
├── PBL_Perbandingan_Model_Terbaik.ipynb # Notebook Riset & Evaluasi Utama
├── NaiveBayes/                     # Riset & Varian Model Naive Bayes
├── RandomForest/                   # Riset & Varian Model Random Forest
├── SVM/                            # Riset & Varian Model SVM
├── LogisticRegression/             # Riset & Varian Model Logistic Regression
└── KNN/                            # Riset & Varian Model KNN
```
