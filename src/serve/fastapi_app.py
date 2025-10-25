import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "rakuten-baseline")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

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

@app.post("/predict")
def predict(product: ProductInput):
    """Predict product category from designation and description"""
    # Load model (will use cache if already loaded)
    model = load_production_model()
    
    # Prepare input dataframe
    text = f"{product.designation or ''} {product.description or ''}"
    df = pd.DataFrame({"text": [text]})

    # Predict
    prediction = model.predict(df)
    return {"predicted_prdtypecode": int(prediction[0])}