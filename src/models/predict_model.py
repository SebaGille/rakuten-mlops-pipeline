import pandas as pd
import joblib
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODELS_DIR / "baseline_model.pkl"
INPUT_FILE = DATA_RAW / "X_test.csv"
OUTPUT_FILE = DATA_PROCESSED / "predictions.csv"

def clean_text(text):
    import re
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    print("Loading model and vectorizer...")
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    vectorizer = saved["vectorizer"]

    print("Loading test data...")
    X_test = pd.read_csv(INPUT_FILE)
    print(f"Loaded shape: {X_test.shape}")

    # preprocess text
    X_test["designation_clean"] = X_test["designation"].apply(clean_text)
    X_test["description_clean"] = X_test["description"].apply(clean_text)
    X_test["text"] = X_test["designation_clean"].fillna("") + " " + X_test["description_clean"].fillna("")

    print("Vectorizing and predicting...")
    X_vec = vectorizer.transform(X_test["text"])
    preds = model.predict(X_vec)

    X_test["predicted_prdtypecode"] = preds

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_test[["productid", "predicted_prdtypecode"]].to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
