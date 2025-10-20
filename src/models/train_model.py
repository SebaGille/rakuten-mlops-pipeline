import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"

INPUT_FILE = DATA_PROCESSED / "train_features.csv"
MODEL_FILE = MODELS_DIR / "baseline_model.pkl"

EXPERIMENT_NAME = "rakuten-baseline"

def main():
    print("Loading processed data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded shape: {df.shape}")

    # combine designation and description
    df["text"] = df["designation_clean"].fillna("") + " " + df["description_clean"].fillna("")

    X = df["text"]
    y = df["prdtypecode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- MLflow setup (local server via Docker) ---
    # Prefer env var if provided; otherwise default to local MLflow server
    tracking_uri = mlflow.get_tracking_uri()
    if not tracking_uri or tracking_uri == "file://":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Params for logging
    vec_params = {"max_features": 20000, "ngram_range": (1, 2)}
    model_params = {"model": "LogisticRegression", "max_iter": 200, "random_state": 42}

    with mlflow.start_run(run_name="baseline-logreg-tfidf"):
        # Log parameters
        mlflow.log_params({
            "vec_max_features": vec_params["max_features"],
            "vec_ngram_range": str(vec_params["ngram_range"]),
            **model_params
        })

        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=vec_params["max_features"], ngram_range=vec_params["ngram_range"]
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        print("Training baseline model (Logistic Regression)...")
        model = LogisticRegression(max_iter=model_params["max_iter"])
        model.fit(X_train_vec, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        # Save local artifacts (still useful for CI/DVC)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_FILE)
        with open(METRICS_FILE, "w") as f:
            json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

        # Log model to MLflow registry/artifacts
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:3].tolist(),
            registered_model_name=None  # we'll register in Phase 3
        )
        # Log the vectorizer as a separate artifact
        joblib.dump(vectorizer, "vectorizer.joblib")
        mlflow.log_artifact("vectorizer.joblib")
        Path("vectorizer.joblib").unlink(missing_ok=True)

        # Also log local metrics file as an artifact for convenience
        mlflow.log_artifact(str(METRICS_FILE))

        print(f"Saved model → {MODEL_FILE}")
        print(f"Saved metrics → {METRICS_FILE}")
        print("Run logged to MLflow.")

if __name__ == "__main__":
    main()
