"""
AntiBurnout Machine Learning Microservice API
Wrapper API Resmi untuk Model Terbaik dari PBL_Perbandingan_Model_Terbaik.ipynb
(Standar 50.000 Data Penuh, 30 Fitur Bersih Tanpa Mental_Health_Status, Bebas Data Leakage)

Champion Model : Logistic Regression L2 (Ridge, C=0.1) — Akurasi: ~81.67% | F1-Macro: ~0.7982
Alternatif 1   : Random Forest (GINI) — Akurasi: ~81.45% | F1-Macro: ~0.7959
Alternatif 2   : Multinomial Naive Bayes (MinMaxScaler) — Akurasi: ~80.63% | F1-Macro: ~0.7826

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
        "Menggunakan Champion Model (Logistic Regression L2 pada 50.000 data bebas data leakage) "
        "dari Grand Comparison PBL."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 2. CORS Middleware agar kompatibel dengan WebPBL & MobilePBL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Muat Artefak Model & Metadata
CHAMPION_MODEL = None
IMPUTER = None
SCALER = None
MINMAX_SCALER = None
METADATA = {}
ALTERNATIVE_MODELS = {}


def load_models():
    global CHAMPION_MODEL, IMPUTER, SCALER, MINMAX_SCALER, METADATA, ALTERNATIVE_MODELS

    meta_path = MODELS_DIR / "model_metadata.json"
    best_model_path = MODELS_DIR / "best_model.pkl"
    imputer_path = MODELS_DIR / "imputer.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    minmax_path = MODELS_DIR / "minmax_scaler.pkl"

    required_files = [meta_path, best_model_path, imputer_path, scaler_path, minmax_path]
    if not all(p.exists() for p in required_files):
        logger.warning("Artefak model 50k belum lengkap di models/. Melatih ulang model terbaik...")
        import subprocess
        subprocess.run([sys.executable, str(BASE_DIR / "train_best_models.py")], check=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        METADATA = json.load(f)

    CHAMPION_MODEL = joblib.load(best_model_path)
    IMPUTER = joblib.load(imputer_path)
    SCALER = joblib.load(scaler_path)
    MINMAX_SCALER = joblib.load(minmax_path)
    logger.info(">> Champion Model (Logistic Regression L2), Imputer, & Scalers berhasil dimuat.")

    rf_path = MODELS_DIR / "random_forest_gini.pkl"
    nb_path = MODELS_DIR / "naive_bayes_multinomial.pkl"
    ALTERNATIVE_MODELS = {}
    if rf_path.exists():
        ALTERNATIVE_MODELS["random_forest"] = joblib.load(rf_path)
    if nb_path.exists():
        ALTERNATIVE_MODELS["naive_bayes"] = joblib.load(nb_path)


load_models()

FEATURE_COLUMNS = METADATA.get("feature_columns", [])
NUMERIC_COLUMNS = set(METADATA.get("numeric_columns", []))
ENCODING_MAPS = METADATA.get("encoding_maps", {})
TARGET_MAP = {0: "Low", 1: "Moderate", 2: "High"}
VALID_MODEL_CHOICES = {"champion", "logistic_regression", "random_forest", "naive_bayes"}


# 4. Preprocessing Cerdas 30 Fitur Bersih (Tanpa Mental_Health_Status)
def _preprocess_input(raw_dict: Dict[str, Any], use_minmax: bool = False) -> np.ndarray:
    """Mengonversi dictionary mentah menjadi array terstandarisasi siap inferensi (30 fitur)."""
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

    def _parse_float(key, default_val):
        raw_val = _get(key)
        if raw_val is None:
            return default_val
        if isinstance(raw_val, bool):
            raise HTTPException(
                status_code=422,
                detail=f"Format nilai tidak valid untuk fitur numerik '{key}': tidak boleh boolean.",
            )
        try:
            parsed = float(raw_val)
            if not np.isfinite(parsed):
                raise ValueError("non-finite")
            return parsed
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail=f"Format nilai tidak valid untuk fitur numerik '{key}': '{raw_val}' bukan angka.",
            )

    # Validasi dan ekstraksi fitur utama untuk estimasi korelasi cerdas saat survei ringkas
    stress_val = str(_get("stress_level", "Moderate")).strip().capitalize()
    sleep_hrs = _parse_float("sleep_hours", 7.0)
    mood_score = _parse_float("mood_score", 60.0)
    work_hrs = _parse_float("work_hours_per_week", 40.0)
    chronic = str(_get("chronic_stress", "No")).strip().capitalize()

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

        if col in ENCODING_MAPS:
            mapping = ENCODING_MAPS[col]
            if val is not None:
                str_val = str(val).strip()
                matched_code = None
                for map_k, map_code in mapping.items():
                    if str(map_k).lower() == str_val.lower():
                        matched_code = map_code
                        break
                if matched_code is None:
                    allowed = list(mapping.keys())
                    raise HTTPException(
                        status_code=422,
                        detail=f"Nilai tidak valid untuk fitur kategorikal '{col}': '{val}'. Nilai yang diterima: {allowed}.",
                    )
                row_data[col] = matched_code
            else:
                row_data[col] = np.nan
        else:
            if val is not None:
                if isinstance(val, bool):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Format nilai tidak valid untuk fitur numerik '{col}': tidak boleh boolean.",
                    )
                try:
                    num_val = float(val)
                    if not np.isfinite(num_val):
                        raise ValueError("non-finite")
                    row_data[col] = num_val
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Format nilai tidak valid untuk fitur numerik '{col}': '{val}' bukan angka.",
                    )
            else:
                row_data[col] = np.nan

    input_df = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)
    input_imputed = IMPUTER.transform(input_df)
    if use_minmax:
        input_scaled = np.clip(MINMAX_SCALER.transform(input_imputed), 0.0, None)
    else:
        input_scaled = SCALER.transform(input_imputed)
    return input_scaled


# 5. Endpoint Health & Informasi Model
@app.get("/", tags=["Info"])
@app.get("/health", tags=["Info"])
def health_check():
    """Mengecek status ketersediaan API dan informasi model jawara (50.000 data)."""
    champ_metrics = METADATA.get("evaluation_metrics", {}).get("logistic_regression_l2", {})
    dataset_info = METADATA.get("dataset_info", {})
    return {
        "status": "online",
        "service": "AntiBurnout MachineLearningPBL API",
        "api_version": "2.0.0",
        "champion_model": METADATA.get("champion_model", "Logistic Regression L2 (Ridge, C=0.1)"),
        "accuracy": f"{champ_metrics.get('accuracy', 0.8167) * 100:.2f}%",
        "f1_score_macro": champ_metrics.get("f1_score_macro", 0.7982),
        "dataset_rows": dataset_info.get("total_rows", 50000),
        "total_features": len(FEATURE_COLUMNS),
        "leakage_free": dataset_info.get("preprocessing_leakage_free", True),
        "endpoints": {
            "predict": "POST /predict",
            "models_comparison": "GET /models",
            "documentation": "GET /docs",
        },
    }


@app.get("/models", tags=["Model Comparison"])
def get_models_comparison():
    """Mengembalikan ringkasan grand perbandingan model terbaik (50.000 data bebas leakage)."""
    return {
        "grand_comparison_title": "Grand Comparison Model Supervised Learning Terbaik PBL AntiBurnout (50.000 Data)",
        "source_notebook": "PBL_Perbandingan_Model_Terbaik.ipynb",
        "dataset_info": METADATA.get("dataset_info", {}),
        "champion_model": METADATA.get("champion_model"),
        "grand_comparison_50k": METADATA.get("grand_comparison_50k", []),
        "models": METADATA.get("evaluation_metrics", {}),
        "available_inference_models": [
            "champion (Logistic Regression L2)",
            "logistic_regression (Logistic Regression L2)",
            "random_forest (Random Forest GINI)",
            "naive_bayes (Multinomial Naive Bayes)",
        ],
    }


# 6. Endpoint Prediksi Utama (POST /predict)
@app.post("/predict", tags=["Inference"])
async def predict_burnout(
    request: Request,
    model_choice: Optional[str] = Query(
        "champion",
        description="Pilih model: champion, logistic_regression, random_forest, naive_bayes",
    ),
):
    """
    Melakukan prediksi risiko burnout mahasiswa menggunakan model terbaik terlatih (50.000 data, 30 fitur bersih).

    Output kompatibel penuh dengan WebPBL dan MobilePBL:
    - `burnout_score`: Nilai numerik 5.0 - 98.0
    - `burnout_risk`: "Low", "Moderate", "High"
    - `mental_health_status`: "Healthy", "Needs Attention", "Critical"
    - `ai_wellness_recommendation`: Rekomendasi tindakan personal
    - `probabilities`: Probabilitas tiap kelas risiko
    """
    choice = (model_choice or "champion").strip().lower()
    if choice not in VALID_MODEL_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Pilihan model '{model_choice}' tidak dikenali. Pilihan valid: {sorted(VALID_MODEL_CHOICES)}.",
        )

    try:
        raw_body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Format body request tidak valid. Pastikan mengirim payload JSON yang valid.",
        )

    if not isinstance(raw_body, dict):
        raise HTTPException(
            status_code=400,
            detail="Payload harus berupa objek JSON (key-value).",
        )

    if len(raw_body) == 0 or all(v in (None, "") for v in raw_body.values()):
        raise HTTPException(
            status_code=422,
            detail="Payload tidak boleh kosong. Sertakan minimal satu fitur asesmen mahasiswa.",
        )

    active_model = CHAMPION_MODEL
    active_model_name = "Logistic Regression L2 (Ridge)"
    use_minmax = False

    if choice == "random_forest" and "random_forest" in ALTERNATIVE_MODELS:
        active_model = ALTERNATIVE_MODELS["random_forest"]
        active_model_name = "Random Forest (GINI)"
    elif choice == "naive_bayes" and "naive_bayes" in ALTERNATIVE_MODELS:
        active_model = ALTERNATIVE_MODELS["naive_bayes"]
        active_model_name = "Multinomial Naive Bayes (MultinomialNB)"
        use_minmax = True

    input_scaled = _preprocess_input(raw_body, use_minmax=use_minmax)

    try:
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
                "high": round(p_high, 4),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal melakukan inferensi ML: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f">> [ML Server] Menjalankan AntiBurnout ML API v2.0 di http://127.0.0.1:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
