"""
AntiBurnout Machine Learning Microservice API
Wrapper API Resmi untuk Model Terbaik dari PBL_Perbandingan_Model_Terbaik.ipynb

Champion Model: Gaussian Naive Bayes (GaussianNB)
Akurasi: 98.90% | F1-Score Macro: 0.9850

Framework: FastAPI (Port 8000)
Dokumentasi Interaktif Swagger UI: http://127.0.0.1:8000/docs
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("MachineLearningPBL")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 1. Inisialisasi FastAPI
app = FastAPI(
    title="AntiBurnout ML Prediction API",
    description=(
        "Microservice inferensi Machine Learning untuk deteksi dini risiko burnout mahasiswa. "
        "Menggunakan Champion Model (Gaussian Naive Bayes 98.90% Akurasi) dari Grand Comparison PBL."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 2. CORS Middleware agar kompatibel dengan Web & Mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Muat Artefak Model & Metadata
CHAMPION_MODEL = None
SCALER = None
METADATA = {}
ALTERNATIVE_MODELS = {}

def load_models():
    global CHAMPION_MODEL, SCALER, METADATA, ALTERNATIVE_MODELS
    
    meta_path = MODELS_DIR / "model_metadata.json"
    best_model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not (meta_path.exists() and best_model_path.exists() and scaler_path.exists()):
        logger.warning("Artefak model belum lengkap di models/. Melatih ulang model terbaik...")
        import subprocess
        subprocess.run([sys.executable, str(BASE_DIR / "train_best_models.py")], check=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        METADATA = json.load(f)

    CHAMPION_MODEL = joblib.load(best_model_path)
    SCALER = joblib.load(scaler_path)
    logger.info(">> Champion Model (GaussianNB) & Scaler berhasil dimuat.")

    lr_path = MODELS_DIR / "logistic_regression_l1.pkl"
    rf_path = MODELS_DIR / "random_forest_gini.pkl"
    if lr_path.exists():
        ALTERNATIVE_MODELS["logistic_regression"] = joblib.load(lr_path)
    if rf_path.exists():
        ALTERNATIVE_MODELS["random_forest"] = joblib.load(rf_path)

load_models()

FEATURE_COLUMNS = METADATA.get("feature_columns", [])
ENCODING_MAPS = METADATA.get("encoding_maps", {})
MEAN_IMPUTATION = METADATA.get("mean_imputation", {})
TARGET_MAP = {0: "Low", 1: "Moderate", 2: "High"}


# 4. Preprocessing Cerdas dengan Korelasi Psikometrik Dataset
def _preprocess_input(raw_dict: Dict[str, Any]) -> pd.DataFrame:
    """Mengonversi dictionary mentah menjadi DataFrame terstandarisasi siap inferensi."""
    normalized_input = {}
    for k, v in raw_dict.items():
        clean_key = str(k).strip()
        normalized_input[clean_key] = v
        normalized_input[clean_key.lower()] = v
        normalized_input[clean_key.upper()] = v
        normalized_input[clean_key.replace("_", "").lower()] = v

    def _get(key, default=None):
        for candidate in (key, key.lower(), key.upper(), key.replace("_", "").lower()):
            if candidate in normalized_input and normalized_input[candidate] not in (None, ""):
                return normalized_input[candidate]
        return default

    # Ekstraksi fitur utama untuk kalkulasi korelasi cerdas
    stress_val = str(_get("stress_level", "Moderate")).capitalize()
    sleep_hrs = float(_get("sleep_hours", 7.0))
    mood_score = float(_get("mood_score", 60.0))
    work_hrs = float(_get("work_hours_per_week", 40.0))
    chronic = str(_get("chronic_stress", "No")).capitalize()

    # Estimasi cerdas untuk atribut klinis jika tidak disertakan dalam survei ringkas
    inferred_mental_status = _get("mental_health_status")
    if inferred_mental_status is None:
        if stress_val == "High" or mood_score < 40.0 or chronic == "Yes" or (work_hrs > 55 and sleep_hrs < 5.5):
            inferred_mental_status = "Critical"
        elif stress_val == "Moderate" or mood_score < 65.0 or work_hrs > 45 or sleep_hrs < 6.5:
            inferred_mental_status = "Needs Attention"
        else:
            inferred_mental_status = "Healthy"

    inferred_anxiety = _get("anxiety_score")
    if inferred_anxiety is None:
        if stress_val == "High":
            inferred_anxiety = 75.0 if chronic == "Yes" else 65.0
        elif stress_val == "Moderate":
            inferred_anxiety = 40.0
        else:
            inferred_anxiety = 18.0

    inferred_depression = _get("depression_score")
    if inferred_depression is None:
        if mood_score < 35.0 or chronic == "Yes":
            inferred_depression = 70.0
        elif mood_score < 60.0:
            inferred_depression = 35.0
        else:
            inferred_depression = 15.0

    inferred_work_life_balance = _get("work_life_balance")
    if inferred_work_life_balance is None:
        if work_hrs > 55 or sleep_hrs < 5.5:
            inferred_work_life_balance = 2.5
        elif work_hrs > 42:
            inferred_work_life_balance = 5.0
        else:
            inferred_work_life_balance = 8.0

    inferred_job_sat = _get("job_satisfaction")
    if inferred_job_sat is None:
        inferred_job_sat = 3.0 if stress_val == "High" else (6.0 if stress_val == "Moderate" else 8.5)

    smart_defaults = {
        "Mental_Health_Status": inferred_mental_status,
        "Anxiety_Score": inferred_anxiety,
        "Depression_Score": inferred_depression,
        "Work_Life_Balance": inferred_work_life_balance,
        "Job_Satisfaction": inferred_job_sat,
        "Emotional_Stability": mood_score,
        "Life_Satisfaction": inferred_work_life_balance,
    }

    row_data = {}
    for col in FEATURE_COLUMNS:
        val = _get(col)
        if val is None and col in smart_defaults:
            val = smart_defaults[col]

        # Proses kolom kategorikal
        if col in ENCODING_MAPS:
            mapping = ENCODING_MAPS[col]
            if val is not None:
                str_val = str(val).strip()
                matched_code = None
                for map_k, map_code in mapping.items():
                    if str(map_k).lower() == str_val.lower():
                        matched_code = map_code
                        break
                row_data[col] = matched_code if matched_code is not None else MEAN_IMPUTATION.get(col, 0.0)
            else:
                row_data[col] = MEAN_IMPUTATION.get(col, 0.0)
        else:
            # Proses kolom numerik
            try:
                row_data[col] = float(val) if val not in (None, "") else MEAN_IMPUTATION.get(col, 0.0)
            except (ValueError, TypeError):
                row_data[col] = MEAN_IMPUTATION.get(col, 0.0)

    input_df = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)
    input_scaled = SCALER.transform(input_df)
    return input_scaled


# 5. Endpoint Health & Informasi Model
@app.get("/", tags=["Info"])
@app.get("/health", tags=["Info"])
def health_check():
    """Mengecek status ketersediaan API dan informasi model jawara."""
    return {
        "status": "online",
        "service": "AntiBurnout MachineLearningPBL API",
        "champion_model": METADATA.get("champion_model", "Gaussian Naive Bayes (GaussianNB)"),
        "accuracy": f"{METADATA.get('evaluation_metrics', {}).get('gaussian_nb', {}).get('accuracy', 0.9890) * 100:.2f}%",
        "f1_score_macro": METADATA.get('evaluation_metrics', {}).get('gaussian_nb', {}).get('f1_score_macro', 0.9850),
        "total_features": len(FEATURE_COLUMNS),
        "endpoints": {
            "predict": "POST /predict",
            "models_comparison": "GET /models",
            "documentation": "GET /docs"
        }
    }


@app.get("/models", tags=["Model Comparison"])
def get_models_comparison():
    """Mengembalikan ringkasan grand perbandingan model terbaik dari notebook PBL."""
    return {
        "grand_comparison_title": "Grand Comparison Model Supervised Learning Terbaik PBL AntiBurnout",
        "source_notebook": "PBL_Perbandingan_Model_Terbaik.ipynb",
        "champion_model": METADATA.get("champion_model"),
        "models": METADATA.get("evaluation_metrics", {}),
        "available_inference_models": ["champion (GaussianNB)", "logistic_regression", "random_forest"]
    }


# 6. Endpoint Prediksi Utama (POST /predict)
@app.post("/predict", tags=["Inference"])
async def predict_burnout(
    request: Request,
    model_choice: Optional[str] = Query("champion", description="Pilih model: champion, logistic_regression, random_forest")
):
    """
    Melakukan prediksi risiko burnout mahasiswa menggunakan model terbaik terlatih.
    
    Output kompatibel penuh dengan WebPBL dan MobilePBL:
    - `burnout_score`: Nilai numerik 0.0 - 100.0
    - `burnout_risk`: "Low", "Moderate", "High"
    - `mental_health_status`: "Healthy", "Needs Attention", "Critical"
    - `ai_wellness_recommendation`: Rekomendasi tindakan personal
    - `probabilities`: Probabilitas tiap kelas risiko
    """
    try:
        raw_body = await request.json()
    except Exception:
        raw_body = {}

    if not isinstance(raw_body, dict):
        raise HTTPException(status_code=400, detail="Payload harus berupa objek JSON.")

    active_model = CHAMPION_MODEL
    active_model_name = "Gaussian Naive Bayes (GaussianNB)"

    if model_choice == "logistic_regression" and "logistic_regression" in ALTERNATIVE_MODELS:
        active_model = ALTERNATIVE_MODELS["logistic_regression"]
        active_model_name = "Logistic Regression L1 (Lasso)"
    elif model_choice == "random_forest" and "random_forest" in ALTERNATIVE_MODELS:
        active_model = ALTERNATIVE_MODELS["random_forest"]
        active_model_name = "Random Forest (GINI)"

    try:
        input_scaled = _preprocess_input(raw_body)

        pred_idx = int(active_model.predict(input_scaled)[0])
        probas = active_model.predict_proba(input_scaled)[0]

        pred_risk = TARGET_MAP.get(pred_idx, "Moderate")

        p_low = float(probas[0]) if len(probas) > 0 else 0.33
        p_mod = float(probas[1]) if len(probas) > 1 else 0.33
        p_high = float(probas[2]) if len(probas) > 2 else 0.33

        calc_score = (p_mod * 50.0) + (p_high * 100.0)
        if pred_risk == "Low":
            calc_score = min(calc_score, 35.0)
        elif pred_risk == "High":
            calc_score = max(calc_score, 75.0)
        burnout_score = round(float(np.clip(calc_score, 5.0, 98.0)), 1)

        if pred_risk == "High":
            status = "Critical"
            badge = "danger"
            rec = "Tingkat risiko burnout tinggi terdeteksi oleh AI. Segera kurangi jam kerja/belajar, ambil jeda pemulihan, dan jadwalkan sesi konseling dengan profesional."
        elif pred_risk == "Moderate":
            status = "Needs Attention"
            badge = "warning"
            rec = "Terindikasi kelelahan kerja tingkat sedang. Luangkan waktu jeda berkala di sela aktivitas, lakukan latihan pernapasan/meditasi, dan batasi lembur harian."
        else:
            status = "Healthy"
            badge = "success"
            rec = "Kondisi kesehatan mental dan ritme kerja tergolong prima. Pertahankan konsistensi istirahat teratur dan keseimbangan aktivitas fisik harian."

        return {
            "status": "success",
            "is_simulation": False,
            "model_name": active_model_name,
            "burnout_score": burnout_score,
            "burnout_risk": pred_risk,
            "burnout_level": f"{pred_risk} ({active_model_name})",
            "mental_health_status": status,
            "ai_wellness_recommendation": rec,
            "badge_class": badge,
            "probabilities": {
                "low": round(p_low, 4),
                "moderate": round(p_mod, 4),
                "high": round(p_high, 4)
            }
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal melakukan inferensi ML: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f">> [ML Server] Menjalankan AntiBurnout ML API di http://127.0.0.1:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
