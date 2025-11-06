# Rakuten Product Classification — Local MLOps Pipeline Snapshot (Nov 2025)

> Local-first deployment: everything documented here runs on localhost with Docker, Prefect, and Streamlit. AWS workstreams are tracked separately.

## What This Pipeline Delivers
- Multi-modal classification (text + image metadata) for Rakuten catalog entries
- Prefect orchestration covering ingestion → feature engineering → training → batch predictions
- Data & model versioning with DVC (remote: Dagshub)
- Experiment tracking and model registry via MLflow + PostgreSQL containers
- FastAPI inference service with Prometheus metrics & Evidently drift analysis
- Grafana dashboards and Streamlit “control room” for day-to-day ops
- CI sanity checks through GitHub Actions (lint, unit tests)

## Architecture at a Glance

```
                               ┌─────────────┐
                               │  data/raw   │
                               └─────┬───────┘
                                     │ ingest (Prefect + DVC)
                                     ▼
┌──────────────┐   features   ┌──────────────┐   train   ┌──────────────┐
│ DVC pipeline │────────────▶│ Prefect Flow │──────────▶│   MLflow     │
└──────┬──────┘              └─────┬────────┘          └────┬─────────┘
       │ artifacts                   │ metrics/models        │ artifacts
       ▼                             ▼                       ▼
┌──────────────┐            ┌──────────────┐        ┌──────────────────┐
│ data/interim │            │ data/processed│       │ models/ & metrics │
└──────────────┘            └──────────────┘        └────────┬─────────┘
                                                              │
                                                              ▼
                                                      ┌──────────────┐
                                                      │ FastAPI API  │
                                                      └────┬─────────┘
                                                           │ requests
                                                           ▼
         ┌──────────────────────┐      metrics      ┌──────────────────┐
         │ inference_log.csv    │◀─────────────────│ Prometheus Export │
         └──────────┬───────────┘                  └────────┬──────────┘
                    │                                    scrape │
                    ▼                                         ▼
              ┌─────────────┐                         ┌────────────────┐
              │ Evidently   │                         │ Grafana        │
              └────┬────────┘                         └──────┬─────────┘
                   │ drift insights                         │ dashboards
                   ▼                                         ▼
             Prefect monitor flow                    Streamlit Ops UI
```

The Streamlit app orchestrates Docker services, Prefect flows, MLflow runs, and monitoring dashboards from a single UI.

## Quick Start (Localhost)

1. **Clone & install Python deps** (Python 3.11 is required for Prefect):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Create `.env`** (copy the template and fill values as needed):
   ```bash
   cp .env.example .env
   nano .env
   ```
   Required keys today:
   - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
   - `MLFLOW_TRACKING_URI=http://localhost:5000`
   - `GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD`
   - Optional S3 keys if you replicate artifact sync (not required for local runs)
3. **Start core services** (PostgreSQL + MLflow + FastAPI API):
   ```bash
   docker-compose -f docker-compose.api.yml up -d
   # MLflow UI → http://localhost:5000
   # FastAPI docs → http://localhost:8000/docs
   ```
4. **Run the end-to-end Prefect flow**:
   ```bash
   source .venv/bin/activate
   export $(cat .env | grep -v '^#' | xargs)
   python flows/pipeline_flow.py
   ```
5. **Bring up monitoring** (Prometheus + Grafana):
   ```bash
   docker-compose -f docker-compose.monitor.yml up -d
   # Prometheus → http://localhost:9090
   # Grafana → http://localhost:3000 (use credentials from .env)
   ```
6. **Generate drift report (optional but recommended after predictions)**:
   ```bash
   python src/monitoring/generate_evidently.py
   open reports/evidently/evidently_report.html
   ```
7. **Launch the Streamlit control room**:
   ```bash
   pip install -r requirements-streamlit.txt
   streamlit run streamlit_app/Home.py
   ```

## Operational Cheat-Sheet

**Prefect flows**
- `flows/pipeline_flow.py` — orchestrates ingest → preprocess → train → predict.
- `flows/monitor_and_retrain.py` — checks drift (Evidently) and triggers retrain.

**DVC**
- `dvc repro` reproduces the exact pipeline tracked in `dvc.yaml`.
- `dvc push` uploads artifacts to Dagshub when credentials are configured.

**FastAPI service**
- Exposed at `http://localhost:8000` with `/predict`, `/health`, `/metrics` endpoints.
- Loads latest production model from MLflow registry (tagged via Prefect run).

**Monitoring stack**
- Prometheus scrapes FastAPI metrics and custom exporters.
- Grafana dashboards pre-provisioned under `monitoring/` configs.
- Evidently compares `data/monitoring/inference_log.csv` against reference data.

**Streamlit app**
- Start/stop Docker services, launch Prefect flows, inspect MLflow runs.
- Useful scripts: `run_streamlit.sh` (local helper).

## Testing & CI
- Quick smoke test: `./quick_test.sh`
- Unit/integration tests: `pytest`
- GitHub Actions runs lint + tests on every push to `Dev` and `master`.

## Troubleshooting (Local Only)
- **Missing `.env` variables** → containers fail to start or MLflow writes locally. Re-run `export $(cat .env ...)` in each shell.
- **Model artifacts not found** → run `python flows/pipeline_flow.py` to refresh model + metrics.
- **Grafana empty panels** → ensure `docker-compose.monitor.yml` is running and Prometheus can reach `http://rakuten_api:8000/metrics` inside the network.
- **Streamlit cannot control Docker** → run the app with privileges to access the local Docker socket.

## Future Work (Out of Scope For This Snapshot)
- Dedicated AWS ECS/Fargate deployments (MLflow + API)
- Route53/HTTPS hardening
- Automated model promotions via MLflow REST API
- Cloud-native observability (CloudWatch alarms, managed Prometheus/Grafana)

## License

See `LICENSE` for full terms.

