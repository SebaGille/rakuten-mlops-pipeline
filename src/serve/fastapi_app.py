"""
Rakuten Product Classification API
Python 3.11+ compatible
"""
import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import time
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest


# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "rakuten-baseline")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
LOG_DIR = Path(os.getenv("INFERENCE_LOG_DIR", "data/monitoring"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_LOG = LOG_DIR / "inference_log.csv"

# --- Prometheus metrics ---
PREDICTION_COUNTER = Counter("rakuten_predictions_total", "Total predictions by class", ["prdtypecode"])
PREDICTION_LATENCY = Histogram("rakuten_prediction_latency_seconds", "Prediction latency (seconds)")
TEXT_LEN_HIST = Histogram("rakuten_text_len_chars", "Length of input text (characters)")


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- FastAPI app ---
app = FastAPI(title="Rakuten Product Classification API")

# --- Input schema ---
class ProductInput(BaseModel):
    designation: str
    description: str

# --- Model cache ---
_model_cache = {"model": None, "error": None}

# --- Load model from MLflow Registry ---
def load_production_model():
    """Lazy load the model from MLflow Registry"""
    if _model_cache["model"] is not None:
        return _model_cache["model"]
    
    if _model_cache["error"] is not None:
        raise _model_cache["error"]
    
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading model from {model_uri} ...")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
        _model_cache["model"] = model
        return model
    except Exception as e:
        error_msg = f"Failed to load model '{MODEL_NAME}' at stage '{MODEL_STAGE}': {str(e)}"
        print(error_msg)
        _model_cache["error"] = HTTPException(status_code=503, detail=error_msg)
        raise _model_cache["error"]

# --- API endpoints ---
@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        # Try to load the model
        load_production_model()
        return {
            "status": "healthy",
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
            "model_loaded": True
        }
    except HTTPException:
        return {
            "status": "degraded",
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
            "model_loaded": False,
            "message": "Model not available. Please register and deploy a model first."
        }

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(product: ProductInput):
    """Predict product category from designation and description"""
    # Start latency measurement
    start_time = time.time()
    
    # Load model (will use cache if already loaded)
    model = load_production_model()
    
    # Prepare input text (as list of strings for the Pipeline)
    text = f"{product.designation or ''} {product.description or ''}".strip()
    
    # Observe text length
    TEXT_LEN_HIST.observe(len(text))

    # Predict - pass as list of strings (not DataFrame)
    prediction = model.predict([text])
    pred = int(prediction[0])
    
    # Record prediction latency
    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)
    
    # Count prediction by class
    PREDICTION_COUNTER.labels(prdtypecode=str(pred)).inc()

    # --- append inference log (CSV append-only) ---
    print(f"[DEBUG] About to log prediction. INFERENCE_LOG={INFERENCE_LOG}")
    try:
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "text_len": len(text),
            "designation": product.designation,
            "description": product.description,
            "predicted_prdtypecode": pred,
        }])
        # write header only if file does not exist
        header = not INFERENCE_LOG.exists()
        print(f"[DEBUG] File exists: {INFERENCE_LOG.exists()}, header: {header}")
        row.to_csv(INFERENCE_LOG, mode="a", header=header, index=False)
        print(f"[DEBUG] Successfully wrote to log")
    except Exception as e:
        # soft-fail logging (n'interrompt pas la prédiction)
        print(f"[warn] failed to append inference log: {e}")

    return {"predicted_prdtypecode": pred}
