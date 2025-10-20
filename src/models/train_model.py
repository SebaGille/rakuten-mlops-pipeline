import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"

INPUT_FILE = DATA_PROCESSED / "train_features.csv"
MODEL_FILE = MODELS_DIR / "baseline_model.pkl"

def main():
    print("Loading processed data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded shape: {df.shape}")

    # combine designation and description
    df["text"] = df["designation_clean"].fillna("") + " " + df["description_clean"].fillna("")

    X = df["text"]
    y = df["prdtypecode"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training baseline model (Logistic Regression)...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_FILE)
    print(f"Saved model → {MODEL_FILE}")

    with open(METRICS_FILE, "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)
    print(f"Saved metrics → {METRICS_FILE}")

if __name__ == "__main__":
    main()
