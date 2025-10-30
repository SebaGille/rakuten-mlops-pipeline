# Rakuten Product Classification — MLOps Pipeline

> ⚠️ **Note on Project Origin**: This repository was initially forked from an **empty school template**. 100% of the visible code, MLOps architecture, and implementation was developed by me personally. The initial fork only served as an empty folder structure.

A complete MLOps pipeline for Rakuten product classification (text + images) with **traceability**, **versioning**, and **experiment tracking**.

## 🔧 Tech Stack
- **Python 3.11** (venv) — ⚠️ Required for Prefect
- **DVC** + **Dagshub** (data/model versioning)
- **MLflow** + **PostgreSQL** + **FastAPI** (Docker) — experiment tracking & serving
- **MLflow Artifacts**: AWS S3 (via environment variables)
- **Prefect** (orchestration) — installed
- **Prometheus** + **Grafana** + **Evidently** (monitoring & drift detection) — ✅ implemented
- **CI/CD**: GitHub Actions — ✅ implemented

## 📦 Data Structure
```
data/
├─ raw/
│  ├─ X_train.csv
│  ├─ Y_train.csv
│  ├─ X_test.csv
│  └─ images/image_train/   (files: image_<imageid>_product_<productid>.jpg)
├─ interim/
│  └─ merged_train.csv
└─ processed/
    ├─ train_features.csv
    └─ predictions.csv
```

## 🗂️ Repository Structure
```
src/
├─ data/make_dataset.py            # data ingestion + validation
├─ features/build_features.py      # preprocessing & text features
└─ models/
   ├─ train_model.py               # training + MLflow logging
   └─ predict_model.py             # inference on X_test
docker-compose.mlflow.yml          # MLflow + Postgres (Docker)
Dockerfile.mlflow                  # custom MLflow image (psycopg2)
dvc.yaml                           # DVC pipeline (ingest→features→train→predict)
```

## 🚀 Quick Start

### 1) Python Environment
```bash
# Use Python 3.11 (required for Prefect)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Environment Variables (S3 / MLflow)

⚠️ **IMPORTANT**: Create a `.env` file at project root (not committed, already in `.gitignore`):

```bash
# AWS S3 Configuration for MLflow
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=eu-west-1
S3_BUCKET_NAME=your-bucket-name
```

Docker Compose automatically loads this `.env` file for containers.

### 3) Start Docker Services (MLflow + PostgreSQL + API)

```bash
# Start all services (mlflow, postgres, rakuten_api)
docker-compose -f docker-compose.api.yml up -d

# Check that everything is running
docker ps

# MLflow UI: http://localhost:5000
# Rakuten API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Stop all containers**:
```bash
docker-compose -f docker-compose.api.yml down
```

**Restart after modifications**:
```bash
docker-compose -f docker-compose.api.yml down
docker-compose -f docker-compose.api.yml up --build -d
```

### 4) Run the Pipeline (Prefect)

⚠️ **CRITICAL**: For MLflow artifacts to be saved on S3 (not locally), **you MUST load environment variables** before running the pipeline:

```bash
# Activate virtual environment
source .venv/bin/activate

# ⚠️ IMPORTANT: Load AWS variables from .env
export $(cat .env | grep -v '^#' | xargs)

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run complete pipeline (ingest → features → train → predict)
python flows/pipeline_flow.py
```

**One-line command**:
```bash
source .venv/bin/activate && export $(cat .env | grep -v '^#' | xargs) && export MLFLOW_TRACKING_URI=http://localhost:5000 && python flows/pipeline_flow.py
```

### 5) DVC Pipeline (reproducible alternative)

```bash
# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Execute ingest → features → train → predict
dvc repro

# Push artifacts (data/models) to DVC remote (Dagshub)
dvc push
```

### 6) Individual Scripts

```bash
# Load .env first
export $(cat .env | grep -v '^#' | xargs)

# Build features
python src/features/build_features.py

# Training (MLflow logging + S3 artifacts)
python src/models/train_model.py

# Predictions on X_test
python src/models/predict_model.py
```

### 7) Monitoring Stack (Prometheus + Grafana + Evidently)

**Start monitoring services**:
```bash
docker-compose -f docker-compose.monitor.yml up -d

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

**Generate Evidently drift report**:
```bash
# Make some predictions first to populate inference log
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"designation": "Product name", "description": "Product description"}'

# Generate drift report
python src/monitoring/generate_evidently.py

# View report: reports/evidently/evidently_report.html
```

## 🧭 Best Practices & Traceability

* **MLflow**: Each run logs parameters, metrics, and artifacts (model, vectorizer, metrics.json).
  Tags link the run to the **Git commit** (`git_commit`, `git_branch`) and the run includes `dvc.yaml`/`dvc.lock`.
* **DVC**: Manages pipeline outputs (`data/interim`, `data/processed`, `models/*`) and syncs to Dagshub.
* **Branches**: Work on feature branches (e.g., `Dev`) then PR to `main`.

## 🆘 Troubleshooting

* **Error `OSError: [Errno 30] Read-only file system: '/mlflow'`**: 
  - ⚠️ **You forgot to load environment variables!**
  - Solution: `export $(cat .env | grep -v '^#' | xargs)` before running the pipeline
  - MLflow artifacts must go to S3, not local storage

* **S3 auth fail / Access Denied**: 
  - Check that `.env` is loaded in shell: `echo $AWS_ACCESS_KEY_ID`
  - Verify IAM permissions on S3 bucket
  - Verify that `S3_BUCKET_NAME` is defined

* **DVC "tracked by SCM"**: 
  - Remove from Git tracking (`git rm -r --cached <file>`) before declaring as DVC output

* **Missing images**: 
  - Expected path is `data/raw/images/image_train/` with pattern `image_<imageid>_product_<productid>.jpg`

* **Docker containers not starting**:
  - Verify that `.env` file exists at project root
  - Check logs: `docker logs sep25_cmlops_rakuten-mlflow-1`

* **All predictions returning the same class (e.g., class 10)**:
  - ✅ **FIXED**: This was caused by passing a DataFrame to the model instead of raw text
  - The sklearn Pipeline expects a list/array of strings, not a DataFrame
  - Solution: Pass `model.predict([text])` instead of `model.predict(pd.DataFrame({"text": [text]}))`
  - After fix, restart API: `docker-compose -f docker-compose.api.yml up --build -d`

## 👨‍💻 About This Project

**Skills Demonstrated**:
- ✅ Orchestration with Prefect
- ✅ Experiment tracking with MLflow + PostgreSQL
- ✅ Data and model versioning with DVC + Dagshub
- ✅ Model serving API with FastAPI
- ✅ Containerization with Docker & Docker Compose
- ✅ Artifact storage on AWS S3
- ✅ Monitoring with Prometheus + Grafana (metrics, dashboards)
- ✅ Data drift detection with Evidently
- ✅ CI/CD with GitHub Actions (tests, deployment)
- ✅ Git best practices (branches, tags, atomic commits)
- ✅ Debugging & troubleshooting production issues

**Author**: Sébastien

## 📌 License

See `LICENSE`.

