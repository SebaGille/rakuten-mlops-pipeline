import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "rakuten-baseline"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- FastAPI app ---
app = FastAPI(title="Rakuten Product Classification API")

# --- Input schema ---
class ProductInput(BaseModel):
    designation: str
    description: str

# --- Load model from MLflow Registry ---
def load_production_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Loading model from {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
    return model

model = load_production_model()

# --- API endpoint ---
@app.post("/predict")
def predict(product: ProductInput):
    # Prepare input dataframe
    text = f"{product.designation or ''} {product.description or ''}"
    df = pd.DataFrame({"text": [text]})

    # Predict
    prediction = model.predict(df)
    return {"predicted_prdtypecode": int(prediction[0])}