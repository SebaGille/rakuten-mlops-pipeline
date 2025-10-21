import json
import os
from pathlib import Path

import boto3
import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# --- Paths & Config ---
def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(_project_root())))
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = MODELS_DIR / "metrics.json"
INPUT_FILE = DATA_PROCESSED / "train_features.csv"
MODEL_FILE = MODELS_DIR / "baseline_model.pkl"
EXPERIMENT_NAME = "rakuten-baseline-s3"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main():
    print("Loading processed data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded shape: {df.shape}")

    # Validate required columns
    required = {"designation_clean", "description_clean", "prdtypecode"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Combine text
    df["text"] = df["designation_clean"].fillna("") + " " + df["description_clean"].fillna("")
    X = df["text"]
    y = df["prdtypecode"]

    # Split (fallback to non-stratified if class counts are problematic)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}); falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

    # --- MLflow setup ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    vec_params = {"max_features": 20000, "ngram_range": (1, 2)}
    model_params = {"model": "LogisticRegression", "max_iter": 200, "random_state": 42}

    with mlflow.start_run(run_name="baseline-logreg-tfidf"):
        # Log params
        mlflow.log_params(
            {
                "vec_max_features": vec_params["max_features"],
                "vec_ngram_range": str(vec_params["ngram_range"]),
                **model_params,
            }
        )

        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=vec_params["max_features"],
            ngram_range=vec_params["ngram_range"],
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        print("Training baseline model (Logistic Regression)...")
        model = LogisticRegression(
            max_iter=model_params["max_iter"],
            random_state=model_params["random_state"],
            solver="lbfgs",  # multinomial-capable
            multi_class="auto",
        )
        model.fit(X_train_vec, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        # Save local artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_FILE)
        with open(METRICS_FILE, "w") as f:
            json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

        # Log artifacts to MLflow
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        if not METRICS_FILE.exists():
            raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

        mlflow.log_artifact(str(MODEL_FILE))   # baseline_model.pkl (model + vectorizer)
        mlflow.log_artifact(str(METRICS_FILE))

        print(f"Saved model → {MODEL_FILE}")
        print(f"Saved metrics → {METRICS_FILE}")
        print("Run logged to MLflow.")


if __name__ == "__main__":
    main()
