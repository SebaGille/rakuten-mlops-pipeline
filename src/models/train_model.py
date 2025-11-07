import json
import os
from pathlib import Path
import subprocess
import io

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

# Optional S3 support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Handle imports for both package and script execution
try:
    from src.models.multimodal import MultiModalClassifier
except ModuleNotFoundError:
    from multimodal import MultiModalClassifier


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
IMAGE_FEATURES_FILE = DATA_PROCESSED / "image_features.npy"
MODEL_FILE = MODELS_DIR / "baseline_model.pkl"
EXPERIMENT_NAME = "rakuten-multimodal-text-image"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"

git_commit = _git(["git", "rev-parse", "--short", "HEAD"])
git_branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _load_from_s3(s3_key: str) -> pd.DataFrame:
    """Load a CSV file from S3 if configured"""
    if not S3_AVAILABLE:
        return None
    
    s3_bucket = os.getenv("S3_DATA_BUCKET", "")
    s3_prefix = os.getenv("S3_DATA_PREFIX", "data/")
    
    if not s3_bucket:
        print("S3_DATA_BUCKET not configured, skipping S3 load")
        return None
    
    # Get AWS credentials from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    
    try:
        # Create S3 client with credentials if available
        s3_client_kwargs = {'region_name': aws_region}
        if aws_access_key and aws_secret_key:
            s3_client_kwargs['aws_access_key_id'] = aws_access_key
            s3_client_kwargs['aws_secret_access_key'] = aws_secret_key
        
        s3_client = boto3.client('s3', **s3_client_kwargs)
        full_key = f"{s3_prefix.rstrip('/')}/{s3_key.lstrip('/')}"
        
        print(f"Attempting to load from S3: s3://{s3_bucket}/{full_key}")
        response = s3_client.get_object(Bucket=s3_bucket, Key=full_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        print(f"✓ Successfully loaded {s3_key} from S3: s3://{s3_bucket}/{full_key}")
        return df
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'NoSuchKey':
            print(f"S3 key not found: s3://{s3_bucket}/{full_key}")
            print(f"  Please verify the file exists at this path in S3")
        elif error_code == 'AccessDenied':
            print(f"S3 access denied: s3://{s3_bucket}/{full_key}")
            print(f"  Please check AWS credentials and bucket permissions")
        else:
            print(f"S3 error loading {s3_key}: {e}")
        return None
    except NoCredentialsError as e:
        print(f"AWS credentials not found. Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return None
    except Exception as e:
        print(f"Error loading from S3: {e}")
        print(f"  Attempted path: s3://{s3_bucket}/{full_key}")
        return None


def _load_numpy_from_s3(s3_key: str) -> np.ndarray:
    """Load a numpy array from S3 if configured"""
    if not S3_AVAILABLE:
        return None
    
    s3_bucket = os.getenv("S3_DATA_BUCKET", "")
    s3_prefix = os.getenv("S3_DATA_PREFIX", "data/")
    
    if not s3_bucket:
        print("S3_DATA_BUCKET not configured, skipping S3 load")
        return None
    
    # Get AWS credentials from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    
    try:
        # Create S3 client with credentials if available
        s3_client_kwargs = {'region_name': aws_region}
        if aws_access_key and aws_secret_key:
            s3_client_kwargs['aws_access_key_id'] = aws_access_key
            s3_client_kwargs['aws_secret_access_key'] = aws_secret_key
        
        s3_client = boto3.client('s3', **s3_client_kwargs)
        full_key = f"{s3_prefix.rstrip('/')}/{s3_key.lstrip('/')}"
        
        print(f"Attempting to load from S3: s3://{s3_bucket}/{full_key}")
        response = s3_client.get_object(Bucket=s3_bucket, Key=full_key)
        array = np.load(io.BytesIO(response['Body'].read()))
        print(f"✓ Successfully loaded {s3_key} from S3: s3://{s3_bucket}/{full_key}")
        return array
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'NoSuchKey':
            print(f"S3 key not found: s3://{s3_bucket}/{full_key}")
            print(f"  Please verify the file exists at this path in S3")
        elif error_code == 'AccessDenied':
            print(f"S3 access denied: s3://{s3_bucket}/{full_key}")
            print(f"  Please check AWS credentials and bucket permissions")
        else:
            print(f"S3 error loading {s3_key}: {e}")
        return None
    except NoCredentialsError as e:
        print(f"AWS credentials not found. Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return None
    except Exception as e:
        print(f"Error loading from S3: {e}")
        print(f"  Attempted path: s3://{s3_bucket}/{full_key}")
        return None


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
        
        # Validate that the champion model version has a valid run_id
        if not prod_version.run_id or (isinstance(prod_version.run_id, str) and prod_version.run_id.strip() == ""):
            print(f"Warning: Champion model version {prod_version.version} has no run_id (run_id={prod_version.run_id}).")
            print(f"Promoting current model to champion.")
            return "champion", "champion_model_has_no_run_id"
        
        print(f"Found champion model version {prod_version.version} with run_id: {prod_version.run_id}")
        
        # Get the run for the champion model
        try:
            prod_run = client.get_run(prod_version.run_id)
        except Exception as run_error:
            print(f"Warning: Could not retrieve run {prod_version.run_id} for champion model: {run_error}")
            print(f"Promoting current model to champion.")
            return "champion", f"champion_run_not_found: {str(run_error)}"
        
        # Validate that the run has metrics
        if not hasattr(prod_run, 'data') or not hasattr(prod_run.data, 'metrics'):
            print(f"Warning: Champion run {prod_version.run_id} has no metrics. Promoting current model to champion.")
            return "champion", "champion_run_has_no_metrics"
        
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
    # --- Read configuration from environment variables ---
    sample_size_str = os.getenv("SAMPLE_SIZE", "full")
    use_images_str = os.getenv("USE_IMAGES", "True")
    use_images = use_images_str.lower() in ("true", "1", "yes")
    
    # Read hyperparameters from environment
    max_features = int(os.getenv("HYPERPARAM_MAX_FEATURES", "20000"))
    ngram_range_str = os.getenv("HYPERPARAM_NGRAM_RANGE", "(1, 2)")
    # Parse ngram_range tuple from string
    try:
        ngram_range = eval(ngram_range_str) if isinstance(ngram_range_str, str) else ngram_range_str
    except:
        ngram_range = (1, 2)
    
    max_iter = int(os.getenv("HYPERPARAM_MAX_ITER", "200"))
    solver = os.getenv("HYPERPARAM_SOLVER", "lbfgs")
    random_state = int(os.getenv("HYPERPARAM_RANDOM_STATE", "42"))
    
    print("\n=== Training Configuration ===")
    print(f"Sample size: {sample_size_str}")
    print(f"Use images: {use_images}")
    print(f"TF-IDF max_features: {max_features}")
    print(f"TF-IDF ngram_range: {ngram_range}")
    print(f"Max iterations: {max_iter}")
    print(f"Solver: {solver}")
    print(f"Random state: {random_state}")
    print("=" * 50)
    
    print("\nLoading processed data...")
    # Try local file first, then S3 if available
    if INPUT_FILE.exists():
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded from local file: {INPUT_FILE}")
    else:
        # Try loading from S3
        df = _load_from_s3("processed/train_features.csv")
        if df is None or df.empty:
            raise FileNotFoundError(
                f"Training data not found locally at {INPUT_FILE} and not available in S3. "
                f"Please ensure the data file exists or configure S3_DATA_BUCKET and S3_DATA_PREFIX environment variables."
            )
    
    print(f"Loaded full dataset shape: {df.shape}")

    # --- Apply sampling if specified ---
    if sample_size_str != "full":
        try:
            sample_size = int(sample_size_str)
            if sample_size < len(df):
                print(f"Sampling {sample_size} rows from {len(df)} total rows...")
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
                print(f"Sampled dataset shape: {df.shape}")
        except ValueError:
            print(f"Warning: Invalid SAMPLE_SIZE '{sample_size_str}', using full dataset")

    # Validate required columns
    required = {"designation_clean", "description_clean", "prdtypecode"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Load image features (only if needed)
    image_features = None
    if use_images:
        print("Loading image features...")
        # Try local file first, then S3 if available
        image_features_full = None
        if IMAGE_FEATURES_FILE.exists():
            image_features_full = np.load(IMAGE_FEATURES_FILE)
            print(f"Loaded from local file: {IMAGE_FEATURES_FILE}")
        else:
            # Try loading from S3
            image_features_full = _load_numpy_from_s3("processed/image_features.npy")
            if image_features_full is not None:
                print(f"Loaded from S3: processed/image_features.npy")
        
        # If image features are still not available, fall back to text-only mode
        if image_features_full is None:
            s3_bucket = os.getenv("S3_DATA_BUCKET", "")
            s3_prefix = os.getenv("S3_DATA_PREFIX", "data/")
            s3_path = f"{s3_prefix.rstrip('/')}/processed/image_features.npy" if s3_bucket else "N/A"
            print("\n⚠️  WARNING: Image features not found!")
            print(f"   Local path: {IMAGE_FEATURES_FILE}")
            if s3_bucket:
                print(f"   S3 path: s3://{s3_bucket}/{s3_path}")
            else:
                print(f"   S3: Not configured (S3_DATA_BUCKET not set)")
            print("   Falling back to TEXT-ONLY mode.")
            print("   To use images:")
            print("   1. Run feature extraction: python src/features/build_features.py")
            print("   2. Or ensure file exists in S3 at the path above")
            print("   3. Verify S3_DATA_BUCKET, S3_DATA_PREFIX, and AWS credentials are configured\n")
            use_images = False
        else:
            print(f"Image features shape: {image_features_full.shape}")
            
            # If we sampled the data, we need to get the corresponding image features
            # This assumes df has an index or imageid column to match
            if sample_size_str != "full" and 'imageid' in df.columns:
                # Match image features by index
                # This is a simplification - in production you'd want proper ID matching
                print("Note: Image features might not be properly aligned after sampling!")
                print("      Consider adding proper ID-based matching for production use.")
            
            # For now, just take the first N image features (aligned with CSV order)
            image_features = image_features_full[:len(df)]
            
            if len(df) != len(image_features):
                print(f"\n⚠️  WARNING: Mismatch - {len(df)} rows in CSV but {len(image_features)} image features")
                print("   Falling back to TEXT-ONLY mode.")
                use_images = False
                image_features = None
    
    if not use_images:
        print("Training in TEXT-ONLY mode (images disabled)")

    # Combine text
    df["text"] = df["designation_clean"].fillna("") + " " + df["description_clean"].fillna("")
    X_text = df["text"]
    y = df["prdtypecode"]

    # Split data (same random_state for both text and images to keep alignment)
    if use_images:
        try:
            X_text_train, X_text_test, y_train, y_test, img_train, img_test = train_test_split(
                X_text, y, image_features, test_size=0.2, random_state=random_state, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed ({e}); falling back to non-stratified split.")
            X_text_train, X_text_test, y_train, y_test, img_train, img_test = train_test_split(
                X_text, y, image_features, test_size=0.2, random_state=random_state, stratify=None
            )
    else:
        # Text-only mode: no image features
        try:
            X_text_train, X_text_test, y_train, y_test = train_test_split(
                X_text, y, test_size=0.2, random_state=random_state, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed ({e}); falling back to non-stratified split.")
            X_text_train, X_text_test, y_train, y_test = train_test_split(
                X_text, y, test_size=0.2, random_state=random_state, stratify=None
            )
        img_train = None
        img_test = None
    
    print(f"Train set: {len(X_text_train)} samples")
    print(f"Test set: {len(X_text_test)} samples")

    # --- MLflow setup ---
    print(f"\nConnecting to MLflow at: {MLFLOW_URI}")
    try:
        # Configure host-based routing for AWS ALB (same fix as MLflowManager)
        mlflow_host = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
        if mlflow_host:
            try:
                import requests
                from functools import wraps
                
                # Store original request method
                original_request = requests.Session.request
                tracking_uri_base = MLFLOW_URI.rstrip("/")
                
                @wraps(original_request)
                def request_with_host_header(session_self, method, url, *args, **kwargs):
                    # If the URL matches our tracking URI, add Host header for host-based routing
                    if tracking_uri_base and url.startswith(tracking_uri_base):
                        if 'headers' not in kwargs:
                            kwargs['headers'] = {}
                        kwargs['headers']['Host'] = mlflow_host
                    return original_request(session_self, method, url, *args, **kwargs)
                
                # Patch the Session class to add Host header
                requests.Session.request = request_with_host_header
                print(f"Configured host-based routing with Host header: {mlflow_host}")
            except Exception as e:
                print(f"Warning: Failed to configure Host header for host-based routing: {e}")
        
        mlflow.set_tracking_uri(MLFLOW_URI)
        
        # Try to verify connection by creating/getting experiment
        from mlflow.tracking import MlflowClient
        from mlflow.exceptions import RestException, MlflowException
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        
        # First, try to list all available experiments for diagnostics
        experiment = None
        experiment_id = None
        try:
            all_experiments = client.search_experiments(max_results=100)
            print(f"\n=== Available Experiments in MLflow ===")
            if all_experiments:
                for exp in all_experiments:
                    print(f"  - {exp.name} (ID: {exp.experiment_id})")
                    # Try to find our experiment in the list
                    if exp.name == EXPERIMENT_NAME:
                        experiment = exp
                        experiment_id = exp.experiment_id
            else:
                print("  No experiments found in this MLflow instance.")
            print(f"=== Looking for experiment: '{EXPERIMENT_NAME}' ===\n")
        except Exception as list_error:
            print(f"Warning: Could not list experiments: {list_error}")
        
        # Try to get experiment - use search_experiments result if available, otherwise try get_experiment_by_name
        if experiment is None:
            try:
                experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
                if experiment is not None:
                    experiment_id = experiment.experiment_id
                    print(f"Found experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
            except RestException as e:
                # Check if it's a "Not Found" error (experiment doesn't exist)
                # RestException for non-existent experiment typically has "Not Found" in the message
                error_str = str(e).lower()
                # Check if the error indicates the experiment doesn't exist
                is_not_found = (
                    "not found" in error_str or
                    ("detail" in error_str and "not found" in error_str) or
                    (hasattr(e, 'response') and hasattr(e.response, 'text') and 
                     "not found" in e.response.text.lower())
                )
                
                if is_not_found:
                    print(f"\n⚠️  Experiment '{EXPERIMENT_NAME}' not found via get_experiment_by_name.")
                    print(f"   This could mean:")
                    print(f"   1. The experiment was created in a different MLflow instance")
                    print(f"   2. The MLflow tracking URI points to a different backend store")
                    print(f"   3. The API endpoint is not accessible")
                    print(f"\n   Attempting to create new experiment '{EXPERIMENT_NAME}'...")
                    try:
                        experiment_id = client.create_experiment(EXPERIMENT_NAME)
                        print(f"✓ Created experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
                    except RestException as create_error:
                        # If create also fails with "Not Found", it might be an API endpoint issue
                        create_error_str = str(create_error).lower()
                        if "not found" in create_error_str:
                            print(f"✗ Error: Both get_experiment_by_name and create_experiment failed with 'Not Found'")
                            print(f"   This suggests the MLflow API endpoint is not accessible.")
                            print(f"   Please check:")
                            print(f"   1. MLflow server is running at {MLFLOW_URI}")
                            print(f"   2. Network connectivity to MLflow server")
                            print(f"   3. MLFLOW_TRACKING_URI environment variable is correct")
                            print(f"   4. API endpoint routing is configured correctly")
                            raise RuntimeError(f"MLflow API endpoint not accessible: {create_error}")
                        else:
                            print(f"✗ Error: Failed to create experiment: {create_error}")
                            raise RuntimeError(f"MLflow experiment creation failed: {create_error}")
                    except Exception as create_error:
                        print(f"✗ Error: Failed to create experiment: {create_error}")
                        raise RuntimeError(f"MLflow experiment creation failed: {create_error}")
                else:
                    # Other RestException (connection issue, etc.)
                    print(f"Error: MLflow REST API error: {e}")
                    raise RuntimeError(f"MLflow connection failed: {e}")
        
        # If we still don't have an experiment, try to create it
        if experiment is None and experiment_id is None:
            print(f"Experiment '{EXPERIMENT_NAME}' not found, creating it...")
            try:
                experiment_id = client.create_experiment(EXPERIMENT_NAME)
                print(f"Created experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
            except RestException as create_error:
                # If create also fails with "Not Found", it might be an API endpoint issue
                create_error_str = str(create_error).lower()
                if "not found" in create_error_str:
                    print(f"✗ Error: create_experiment failed with 'Not Found'")
                    print(f"   This suggests the MLflow API endpoint is not accessible.")
                    print(f"   Please check:")
                    print(f"   1. MLflow server is running at {MLFLOW_URI}")
                    print(f"   2. Network connectivity to MLflow server")
                    print(f"   3. MLFLOW_TRACKING_URI environment variable is correct")
                    print(f"   4. API endpoint routing is configured correctly")
                    raise RuntimeError(f"MLflow API endpoint not accessible: {create_error}")
                else:
                    print(f"✗ Error: Failed to create experiment: {create_error}")
                    raise RuntimeError(f"MLflow experiment creation failed: {create_error}")
            except Exception as create_error:
                print(f"Error: Failed to create experiment: {create_error}")
                raise RuntimeError(f"MLflow experiment creation failed: {create_error}")
        
        # Set the experiment (this should work now that it exists)
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception as e:
            print(f"Warning: Failed to set experiment: {e}")
            # Continue anyway - experiment should exist now
    except Exception as e:
        print(f"Error: Failed to connect to MLflow at {MLFLOW_URI}: {e}")
        print("Training will continue but MLflow logging will be disabled.")
        raise RuntimeError(f"MLflow connection failed: {e}")

    # Use parameters from environment variables (configured in UI)
    vec_params = {"max_features": max_features, "ngram_range": ngram_range}
    model_type = "MultiModalLogisticRegression" if use_images else "TextOnlyLogisticRegression"
    model_params = {
        "model": model_type, 
        "max_iter": max_iter,
        "solver": solver,
        "random_state": random_state,
    }
    if use_images and image_features is not None:
        model_params["image_features_dim"] = image_features.shape[1]

    # Generate run name based on configuration
    modality_str = "text+image" if use_images else "text-only"
    run_name = f"{modality_str}-logreg-{sample_size_str}samples"
    
    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_params(
            {
                "vec_max_features": vec_params["max_features"],
                "vec_ngram_range": str(vec_params["ngram_range"]),
                "modality": modality_str,
                "sample_size": sample_size_str,
                **model_params,
            }
        )
        # Lier le run au contexte de version
        tags = {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "dvc_repo_rev": git_commit,   
            "pipeline_stages": f"ingest→features({modality_str})→train",
            "model_type": "multimodal" if use_images else "text_only",
        }
        if use_images:
            tags["image_model"] = "MobileNetV2"
        mlflow.set_tags(tags)

        # Joindre les manifests DVC/Git de la run
        mlflow.log_artifact(str(PROJECT_ROOT / "dvc.yaml"))
        if (PROJECT_ROOT / "dvc.lock").exists():
            mlflow.log_artifact(str(PROJECT_ROOT / "dvc.lock"))

        # Train
        if use_images:
            print(f"\n=== Training Multi-Modal Model (Text + Image) ===")
        else:
            print(f"\n=== Training Text-Only Model ===")
            
        print("Creating vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=vec_params["max_features"],
            ngram_range=vec_params["ngram_range"],
        )

        print("Creating classifier...")
        classifier = LogisticRegression(
            max_iter=model_params["max_iter"],
            random_state=model_params["random_state"],
            solver=solver,
        )

        if use_images:
            print("Creating multi-modal model...")
            model = MultiModalClassifier(
                vectorizer=vectorizer,
                image_features_train=img_train,
                classifier=classifier
            )

            print("Training model on combined features (text + image)...")
            model.fit(X_text_train, y_train)

            print("Evaluating model...")
            y_pred = model.predict(X_text_test, image_features_test=img_test)
        else:
            print("Creating text-only pipeline...")
            model = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            
            print("Training model on text features only...")
            model.fit(X_text_train, y_train)
            
            print("Evaluating model...")
            y_pred = model.predict(X_text_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        # --- Log model to MLflow ---
        print(f"Logging {modality_str} model to MLflow...")
        
        # Note: MLflow sklearn.log_model doesn't support custom objects easily
        # We'll use pyfunc for custom multimodal model or just log as artifact
        # For now, log as joblib artifact and metadata
        
        # Save model locally first
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_data = {
            "model": model,
            "vectorizer": vectorizer,
            "classifier": classifier,
            "use_images": use_images,
            "modality": modality_str
        }
        joblib.dump(model_data, MODEL_FILE)
        print(f"Saved model locally → {MODEL_FILE}")
        
        # Save metrics
        with open(METRICS_FILE, "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_weighted": f1, 
                "model_type": modality_str,
                "sample_size": sample_size_str
            }, f, indent=2)
        print(f"Saved metrics → {METRICS_FILE}")

        # Log artifacts to MLflow
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        if not METRICS_FILE.exists():
            raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(METRICS_FILE))
        
        # Log image features file info (not the actual file, too large)
        if use_images and image_features is not None:
            mlflow.log_param("image_features_file", str(IMAGE_FEATURES_FILE))
            mlflow.log_param("image_features_shape", str(image_features.shape))

        # --- Register model in MLflow Model Registry ---
        print("\nRegistering model in MLflow Model Registry...")
        model_name = "rakuten-multimodal"
        try:
            # Use the same client that was configured earlier with proper tracking URI
            # This ensures host-based routing is configured correctly
            client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
            
            # Try to create registered model (will fail if already exists)
            try:
                result = client.create_registered_model(model_name)
                print(f"Created registered model: {result.name}")
            except Exception as e:
                print(f"Model '{model_name}' already registered (this is normal): {e}")
            
            # Create a new model version using the logged artifact
            current_run_id = mlflow.active_run().info.run_id
            if not current_run_id or (isinstance(current_run_id, str) and current_run_id.strip() == ""):
                raise ValueError(f"Invalid run_id when creating model version: {current_run_id}")
            
            model_uri = f"runs:/{current_run_id}/{MODEL_FILE.name}"
            print(f"Creating model version with run_id: {current_run_id}")
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=current_run_id
            )
            print(f"Created model version {model_version.version}")
            
            # Verify that the model version has the run_id set correctly
            if not model_version.run_id or (isinstance(model_version.run_id, str) and model_version.run_id.strip() == ""):
                print(f"Warning: Created model version {model_version.version} but run_id is empty: {model_version.run_id}")
                print(f"Expected run_id: {current_run_id}")
            else:
                print(f"Model version {model_version.version} has run_id: {model_version.run_id}")
            
            # --- Compare and decide promotion ---
            current_metrics = {"f1_weighted": f1, "accuracy": acc}
            target_alias, promotion_reason = compare_and_promote_model(
                client=client,
                model_name=model_name,
                current_run_id=current_run_id,
                current_metrics=current_metrics
            )
            
            # Add auto-promotion tags to the run
            mlflow.set_tags({
                "auto_promotion_candidate": target_alias,
                "auto_promotion_reason": promotion_reason
            })
            
            # Set the model alias - this is critical for automatic promotion
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias=target_alias,
                    version=model_version.version
                )
                print(f"✓ Set alias '{target_alias}' to model version {model_version.version}")
                print(f"  Reason: {promotion_reason}")
                
                # If promoted to champion, log success
                if target_alias == "champion":
                    print(f"🏆 Model automatically promoted to CHAMPION!")
                    if promotion_reason == "no_existing_champion_model":
                        print(f"   This is the first model run - automatically set as champion.")
                    else:
                        print(f"   New model outperformed previous champion based on metrics.")
            except Exception as alias_error:
                # This is a critical error - raise it so training fails if promotion fails
                error_msg = f"CRITICAL: Failed to set model alias '{target_alias}' for version {model_version.version}: {alias_error}"
                print(f"❌ {error_msg}")
                print(f"   Model was registered but NOT promoted. Manual intervention required.")
                # Re-raise the exception so it's visible in training output
                raise RuntimeError(error_msg) from alias_error
            
        except RuntimeError:
            # Re-raise RuntimeError (promotion failures) to fail training
            raise
        except Exception as e:
            # For other errors, log but don't fail training (model was still trained)
            print(f"⚠️  Model registration/promotion error: {e}")
            print(f"   Model was trained successfully but registration/promotion failed.")
            print(f"   You may need to manually register and promote the model.")
            import traceback
            traceback.print_exc()

        # Print run_id for extraction by TrainingManager
        run_id = mlflow.active_run().info.run_id
        print(f"\n✓ Training completed successfully!")
        print(f"  Model: {MODEL_FILE}")
        print(f"  Metrics: {METRICS_FILE}")
        print(f"  MLflow experiment: {EXPERIMENT_NAME}")
        print(f"  MLflow run_id: {run_id}")
        print(f"  Run logged to MLflow.")


if __name__ == "__main__":
    main()
