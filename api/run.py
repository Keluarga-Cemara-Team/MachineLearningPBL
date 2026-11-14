import os
import sys
from pathlib import Path

# Set path root
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    print("=" * 70)
    print(">> ANTIBURNOUT MACHINE LEARNING API MICROSERVICE")
    print(f">> Server berjalan di : http://{host}:{port}")
    print(f">> Swagger Docs UI   : http://{host}:{port}/docs")
    print(f">> Endpoint Prediksi : http://{host}:{port}/predict")
    print("=" * 70)
    uvicorn.run("app:app", host=host, port=port, reload=False, log_level="info")
