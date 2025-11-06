"""
Rakuten MLOps Showcase - Main Application

An interactive demonstration of ML Engineering + Product Management skills
through a complete MLOps pipeline for product classification.

Author: Sébastien
"""
import os
import streamlit as st
from pathlib import Path
import sys
import textwrap

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.constants import COLORS, AWS_ALB_URL

# Page configuration
st.set_page_config(
    page_title="Rakuten MLOps Showcase",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # Rakuten MLOps Showcase
        
        An interactive demonstration of:
        - ML Engineering: MLflow, Docker, APIs, Monitoring
        - Product Management: UX design, user workflows, impact orientation
        - System Design: Complete MLOps pipeline from data to production
        
        Built with Streamlit, MLflow, FastAPI, Docker, and more.
        """
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric styling */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Success/Error message styling */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Helpers
def _detect_deployment_mode() -> str:
    """Return 'cloud' when running on Streamlit Cloud / ALB, else 'local'."""

    runtime_env = os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT", "").lower()
    if runtime_env == "cloud":
        return "cloud"

    server_url = os.getenv("STREAMLIT_SERVER_URL", "")
    if server_url and "streamlit.app" in server_url.lower():
        return "cloud"

    server_address = os.getenv("STREAMLIT_SERVER_ADDRESS", "")
    if server_address and "streamlit.app" in server_address.lower():
        return "cloud"

    try:
        browser_host = st.get_option("browser.serverAddress")
    except Exception:  # pragma: no cover - defensive fallback
        browser_host = ""
    if isinstance(browser_host, str) and "streamlit.app" in browser_host.lower():
        return "cloud"

    if AWS_ALB_URL:
        return "cloud"

    return "local"


# Main content
def main():
    # Hero section
    st.markdown('<h1 class="main-title"> Rakuten MLOps Showcase</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Interactive demonstration of ML Engineering + Product Management skills</p>',
        unsafe_allow_html=True
    )
    
    # Introduction
    st.markdown("---")

    st.markdown("""
        ###  What is this?
        
        An **end-to-end MLOps prototype** you can run locally with Docker Compose or deploy on AWS Fargate behind an Application Load Balancer.
        
        With this app you can:
        - Inspect the production-ready cloud stack (ECS Fargate, RDS, S3, Secrets Manager, CloudWatch) or spin up the same services on localhost
        - Orchestrate data ingestion → feature engineering → training → batch predictions via Prefect and DVC
        - Train new models, compare runs in MLflow, and promote champions directly from the UI
        - Serve real-time predictions through the FastAPI endpoint and log telemetry for monitoring
        - Detect data drift with Evidently and trigger retraining flows
        
        The [project README](https://github.com/SebaGille/rakuten-mlops-pipeline) now includes a detailed AWS deployment guide—this page surfaces the same context for interactive exploration.
        """)
    
    st.markdown("---")
    
    # Feature overview
    st.markdown("## Features Overview")

    st.markdown("""
        ### Infrastructure Command Center
        - Inspect AWS ECS services, Application Load Balancer targets, and RDS health when running in the cloud
        - Start/stop the Docker Compose stack and monitor container health when running from a local clone
        - Unified service checks for MLflow, FastAPI, Prefect flows, and supporting storage
        
        **Go to:** Infrastructure (sidebar)
        """)

    st.markdown("""
        ### Dataset Explorer
        - Browse dataset statistics and category distributions
        - Filter, sample, and download subsets for experimentation
        - Visualize product categories before training
        
        **Go to:** Dataset (sidebar)
        """)

    st.markdown("""
        ### Interactive Training
        - Configure experiments with text-only or multimodal pipelines
        - Set hyperparameters and launch Prefect-managed training jobs
        - Track runs directly in MLflow, compare metrics, and promote champions
        
        **Go to:** Training (sidebar)
        """)

    st.markdown("""
        ### Live Predictions
        - Submit product titles/descriptions and optional images for classification
        - Retrieve confidence scores and review inference history
        - Observe latency metrics sourced from the FastAPI service
        
        **Go to:** Predictions (sidebar)
        """)

    st.markdown("""
        ### Monitoring Dashboard
        - Detect data and concept drift with Evidently reports
        - Review system metrics via Prometheus and Grafana (local deployment only)
        - Track post-deployment quality with Prefect monitoring flows
        
        **Go to:** Monitoring (sidebar)
        """)
    
    st.markdown("---")
    
    # Deployment information
    st.markdown("## Deployment Information")
    
    cloud_image_path = PROJECT_ROOT / "streamlit_app" / "assets" / "cloud_architecture.jpg"
    deployment_mode = _detect_deployment_mode()
    is_cloud = deployment_mode == "cloud"

    if is_cloud:
        alb_display = AWS_ALB_URL or "Set AWS_ALB_URL to surface the load balancer URL"
        st.info(f"""
        **AWS Deployment Detected**
        
        This application is running on the managed cloud stack described in the README:
        - **Application Load Balancer:** {alb_display}
        - **Compute:** ECS Fargate services (`rakuten-api`, `rakuten-mlflow`)
        - **Database:** Amazon RDS (PostgreSQL) for the MLflow backend store
        - **Storage:** Amazon S3 for datasets and MLflow artifacts
        - **Secrets:** AWS Secrets Manager provides DB and model credentials to both tasks
        - **Observability:** CloudWatch Logs capture container output; optional Prometheus/Grafana can be run locally against the ALB
        - **Orchestration:** Prefect flows still drive ingestion, training, and retraining schedules
        
        """)

        st.markdown("### AWS Cloud Stack Overview")
        st.markdown(
            textwrap.dedent(
                """
                | Component | Service | Notes |
                |-----------|---------|-------|
                | Networking | Application Load Balancer | Routes `/api` to the FastAPI service and `/mlflow` to the MLflow UI |
                | Compute | ECS Fargate | Stateless tasks packaged from `Dockerfile.api` and `Dockerfile.mlflow` |
                | Model Registry | MLflow Server | Persists metadata to RDS, stores artifacts in S3 |
                | Database | Amazon RDS (PostgreSQL) | Connection string injected via Secrets Manager |
                | Storage | Amazon S3 | Holds datasets (see `upload_to_s3.sh`) and MLflow artifacts |
                | Secrets | AWS Secrets Manager | Provides DB creds, MLflow config, model name/stage |
                | Observability | CloudWatch Logs | Streams ECS task logs; extend with CloudWatch alarms |
                """
            )
        )

    else:
        st.info("""
        **Local Deployment**
        
        This application is running **locally** with Docker Compose. The stack mirrors production but runs on `localhost`:
        - **PostgreSQL:** MLflow backend database (`docker-compose.mlflow.yml`)
        - **MLflow:** Tracking & model registry → http://localhost:5000
        - **FastAPI:** Prediction API → http://localhost:8000 (`/docs`, `/metrics`, `/predict`)
        - **Prometheus & Grafana:** Optional monitoring dashboards → http://localhost:9090 / http://localhost:3000
        - **Prefect:** Launch flows via CLI (`python flows/pipeline_flow.py`) or integrate with Prefect Cloud
        
        """)

    if is_cloud and cloud_image_path.exists():
        st.markdown("### AWS Cloud Architecture Diagram")
        st.image(str(cloud_image_path), use_column_width=True, caption="Cloud-native deployment at a glance")

    st.markdown("### Architecture Overview")

    architecture_diagram = textwrap.dedent("""
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
    """)

    st.code(architecture_diagram, language="text")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Built  with ❤️ by <strong>Sébastien</strong></p>
        <p>ML Engineer + Product Manager | Seeking combined ML/Product roles</p>
        <p>
            <a href='https://github.com/sebagille' target='_blank'>GitHub</a> | 
            <a href='https://linkedin.com/in/seba-gille' target='_blank'>LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

