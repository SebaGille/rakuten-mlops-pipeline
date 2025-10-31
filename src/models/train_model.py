import json
import os
from pathlib import Path
import subprocess

import boto3
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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
EXPERIMENT_NAME = "rakuten-baseline-S3-MLflow"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"

git_commit = _git(["git", "rev-parse", "--short", "HEAD"])
git_branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def compare_and_promote_model(client, model_name, current_run_id, current_metrics):
    """
    Compare current model with Production alias model and decide promotion strategy.
    Uses MLflow's new alias system instead of deprecated stages.
    
    Returns:
        tuple: (alias, reason) where alias is "champion" or "challenger"
    """
    try:
        # Get current Production model using alias system
        try:
            prod_version = client.get_model_version_by_alias(model_name, "champion")
        except mlflow.exceptions.MlflowException:
            # No champion model exists, promote directly
            return "champion", "no_existing_champion_model"
        
        prod_run = client.get_run(prod_version.run_id)
        prod_metrics = prod_run.data.metrics
        
        # Extract metrics for comparison
        current_f1 = current_metrics.get("f1_weighted", 0.0)
        current_acc = current_metrics.get("accuracy", 0.0)
        prod_f1 = prod_metrics.get("f1_weighted", 0.0)
        prod_acc = prod_metrics.get("accuracy", 0.0)
        
        print(f"\n=== Model Comparison ===")
        print(f"Champion (v{prod_version.version}): F1={prod_f1:.4f}, Acc={prod_acc:.4f}")
        print(f"Current Run: F1={current_f1:.4f}, Acc={current_acc:.4f}")
        
        # Decision logic: F1 weighted is primary, accuracy is tiebreaker
        if current_f1 > prod_f1:
            return "champion", f"better_f1_weighted ({current_f1:.4f} > {prod_f1:.4f})"
        elif current_f1 == prod_f1:
            if current_acc > prod_acc:
                return "champion", f"equal_f1_but_better_accuracy ({current_acc:.4f} > {prod_acc:.4f})"
            else:
                return "challenger", f"equal_f1_but_lower_or_equal_accuracy ({current_acc:.4f} <= {prod_acc:.4f})"
        else:
            return "challenger", f"lower_f1_weighted ({current_f1:.4f} <= {prod_f1:.4f})"
            
    except Exception as e:
        print(f"Error comparing with Champion model: {e}")
        # Fallback to challenger if comparison fails
        return "challenger", f"comparison_error: {str(e)}"


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
        # Lier le run au contexte de version
        mlflow.set_tags({
            "git_commit": git_commit,
            "git_branch": git_branch,
            "dvc_repo_rev": git_commit,   
            "pipeline_stages": "ingest→features→train"
        })

        # Joindre les manifests DVC/Git de la run
        mlflow.log_artifact(str(PROJECT_ROOT / "dvc.yaml"))
        if (PROJECT_ROOT / "dvc.lock").exists():
            mlflow.log_artifact(str(PROJECT_ROOT / "dvc.lock"))

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
        )
        model.fit(X_train_vec, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        # --- Create Pipeline for MLflow Model Registry ---
        print("Creating pipeline (vectorizer + model)...")
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])

        # --- Log model to MLflow ---
        print("Logging model to MLflow...")
        
        # Prepare input example (pandas Series matching pipeline input format)
        input_example = pd.Series(
            ["Smartphone Samsung Galaxy dernière génération avec écran AMOLED"],
            name="text"
        )
        
        # Log the sklearn pipeline
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example
        )
        
        # --- Register model in MLflow Model Registry (manual registration) ---
        print("Registering model in MLflow Model Registry...")
        try:
            client = mlflow.tracking.MlflowClient()
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            # Register the model
            result = client.create_registered_model("rakuten-baseline")
            print(f"Created registered model: {result.name}")
        except Exception as e:
            # Model already exists, continue
            print(f"Model already registered (this is normal): {e}")
        
        # Create a new model version
        try:
            model_version = client.create_model_version(
                name="rakuten-baseline",
                source=model_uri,
                run_id=mlflow.active_run().info.run_id
            )
            print(f"Created model version {model_version.version}")
            
            # --- Compare and decide promotion ---
            current_metrics = {"f1_weighted": f1, "accuracy": acc}
            target_alias, promotion_reason = compare_and_promote_model(
                client=client,
                model_name="rakuten-baseline",
                current_run_id=mlflow.active_run().info.run_id,
                current_metrics=current_metrics
            )
            
            # Add auto-promotion tags to the run
            mlflow.set_tags({
                "auto_promotion_candidate": target_alias,
                "auto_promotion_reason": promotion_reason
            })
            
            # Set the model alias (replaces stage transition)
            client.set_registered_model_alias(
                name="rakuten-baseline",
                alias=target_alias,
                version=model_version.version
            )
            print(f"✓ Set alias '{target_alias}' to model version {model_version.version}")
            print(f"  Reason: {promotion_reason}")
            
        except Exception as e:
            print(f"Model version creation or promotion failed: {e}")

        # Save local artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "model": model, "vectorizer": vectorizer}, MODEL_FILE)
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
