"""Infrastructure Management Page - control AWS ECS services or local Docker services."""

import sys
import time
from pathlib import Path

import streamlit as st

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.ecs_manager import ECSManager
from streamlit_app.utils.docker_manager import LocalDockerManager
from streamlit_app.utils.constants import (
    MLFLOW_URL,
    API_URL,
    AWS_REGION,
    AWS_ECS_CLUSTER,
    AWS_ALB_URL,
    AWS_RDS_INSTANCE_ID,
    CONTAINERS,
)

st.set_page_config(
    page_title="Infrastructure - Rakuten MLOps",
    page_icon="",
    layout="wide"
)


def _is_cloud_deployment() -> bool:
    """Detect if running in cloud (AWS) or localhost mode."""
    # Check if AWS_ALB_URL is set (indicates cloud deployment)
    if AWS_ALB_URL:
        return True
    # Check if secrets are available (indicates cloud deployment)
    from streamlit_app.utils.constants import _has_secrets_file, _safe_get_secret
    if _has_secrets_file():
        # If secrets file exists, check if AWS_ALB_URL is in it
        alb_url = _safe_get_secret("AWS_ALB_URL", "")
        if alb_url:
            return True
    return False


@st.cache_resource
def get_aws_manager() -> ECSManager:
    """Cache the ECS manager across reruns (only used in cloud mode)."""

    return ECSManager(
        region=AWS_REGION,
        cluster=AWS_ECS_CLUSTER,
        rds_instance_id=AWS_RDS_INSTANCE_ID,
    )


@st.cache_resource
def get_docker_manager() -> LocalDockerManager:
    """Cache the Docker manager across reruns (only used in localhost mode)."""
    return LocalDockerManager(str(PROJECT_ROOT))


# Detect deployment mode
is_cloud = _is_cloud_deployment()

st.title("Infrastructure Management")

if is_cloud:
    st.markdown(
        """
        Keep an eye on the managed AWS deployment powering the Rakuten MLOps pipeline: an
        Application Load Balancer fronts two ECS Fargate services (`rakuten-api`, `rakuten-mlflow`),
        backed by Amazon RDS, S3 artifacts, Secrets Manager, and CloudWatch logging.
        """
    )
    if AWS_ALB_URL:
        st.info(f"🔗 Application Load Balancer endpoint: {AWS_ALB_URL}")
    
    ecs_manager = get_aws_manager()
    services_status = ecs_manager.get_all_services_status()
    rds_status = ecs_manager.get_rds_status()
else:
    st.markdown(
        """
        Monitor your local Docker Compose stack: PostgreSQL, MLflow, FastAPI, Prometheus, and Grafana
        containers running on `localhost`. This mirrors the production AWS stack for local development.
        """
    )
    docker_manager = get_docker_manager()
    # Get Docker services status
    docker_services_status = docker_manager.get_all_services_status(CONTAINERS)
    services_status = {}
    rds_status = None  # No RDS in localhost mode


def render_status_indicator(health: str) -> str:
    """Return an emoji+label based on health string."""

    if health == "healthy":
        return "🟢 Healthy"
    if health == "deploying":
        return "🟡 Deploying"
    if health == "stopped":
        return "🔴 Stopped"
    if health:
        return f"⚪ {health.capitalize()}"
    return "⚪ Unknown"


st.markdown("### Service Status Overview")

service_cards = []

if is_cloud:
    # AWS ECS services
    for key, info in (services_status or {}).items():
        running = info.get("runningCount", 0)
        desired = info.get("desired", 0)
        lines = [
            f"Desired: {desired} | Running: {running}",
        ]
        if info.get("serviceArn"):
            lines.append(f"Service ARN: {info['serviceArn'].split('/')[-1]}")

        links = []
        if info.get("url"):
            links.append(("Open service", info["url"]))
        if key == "api" and API_URL:
            links.append(("API docs", f"{API_URL}/docs"))
        if key == "mlflow" and MLFLOW_URL:
            links.append(("MLflow UI", MLFLOW_URL))

        service_cards.append(
            {
                "title": info.get("displayName", key.title()),
                "status": render_status_indicator(info.get("health")),
                "lines": lines,
                "links": links,
                "footnote": info.get("lastEvent"),
            }
        )
else:
    # Docker services
    docker_service_names = {
        "postgres": "PostgreSQL",
        "mlflow": "MLflow Server",
        "api": "Rakuten API",
        "prometheus": "Prometheus",
        "grafana": "Grafana"
    }
    
    docker_urls = {
        "postgres": None,
        "mlflow": "http://localhost:5000",
        "api": "http://localhost:8000",
        "prometheus": "http://localhost:9090",
        "grafana": "http://localhost:3000"
    }
    
    for key, info in (docker_services_status or {}).items():
        running = 1 if info.get("running", False) else 0
        status = info.get("status", "unknown")
        health = info.get("health", "unknown")
        
        lines = [
            f"Status: {status}",
        ]
        if health != "unknown":
            lines.append(f"Health: {health}")
        
        links = []
        if docker_urls.get(key):
            links.append(("Open service", docker_urls[key]))
        if key == "api" and docker_urls.get("api"):
            links.append(("API docs", f"{docker_urls['api']}/docs"))
        if key == "mlflow" and docker_urls.get("mlflow"):
            links.append(("MLflow UI", docker_urls["mlflow"]))
        
        service_cards.append(
            {
                "title": docker_service_names.get(key, key.title()),
                "status": render_status_indicator(health if health != "unknown" else ("healthy" if running else "stopped")),
                "lines": lines,
                "links": links,
                "footnote": None,
            }
        )

if is_cloud and rds_status:
    rds_raw_status = (rds_status.get("status") or "").lower()
    if rds_raw_status in {"available", "storage-optimization"}:
        rds_health = "healthy"
    elif rds_raw_status in {"creating", "modifying", "backing-up"}:
        rds_health = "deploying"
    elif rds_raw_status in {"failed", "storage-full"}:
        rds_health = "stopped"
    else:
        rds_health = "unknown"

    rds_lines = [
        f"Status: {rds_status.get('status', 'unknown')}",
        f"Endpoint: {rds_status.get('endpoint', 'n/a')}",
        f"Engine: {rds_status.get('engine', 'postgresql')} | Multi-AZ: {rds_status.get('multiAZ', False)}",
    ]

    service_cards.append(
        {
            "title": "Amazon RDS (PostgreSQL)",
            "status": render_status_indicator(rds_health),
            "lines": rds_lines,
            "links": [],
            "footnote": None,
        }
    )
else:
    service_cards.append(
        {
            "title": "Amazon RDS (PostgreSQL)",
            "status": render_status_indicator("stopped"),
            "lines": ["Not configured for this environment."],
            "links": [],
            "footnote": None,
        }
    )

if AWS_ALB_URL:
    service_cards.append(
        {
            "title": "Application Load Balancer",
            "status": render_status_indicator("healthy"),
            "lines": [
                f"URL: {AWS_ALB_URL}",
                "Routes /api → rakuten-api",
                "Routes /mlflow → rakuten-mlflow",
            ],
            "links": [("Open ALB endpoint", AWS_ALB_URL)],
            "footnote": None,
        }
    )

if not service_cards:
    st.warning("No infrastructure resources are configured for this dashboard.")
else:
    for start_idx in range(0, len(service_cards), 3):
        row_cards = service_cards[start_idx : start_idx + 3]
        cols = st.columns(len(row_cards))
        for col, card in zip(cols, row_cards):
            with col:
                card_html = """
                <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
                """
                card_html += f"<h4>{card['title']}</h4>"
                card_html += f"<p><strong>Status:</strong> {card['status']}</p>"
                for line in card["lines"]:
                    card_html += f"<p>{line}</p>"
                for link_label, link_url in card["links"]:
                    card_html += (
                        f"<p><a href='{link_url}' target='_blank'>{link_label}</a></p>"
                    )
                if card.get("footnote"):
                    card_html += (
                        f"<p style='font-size:0.85em; color:#555'><em>{card['footnote']}</em></p>"
                    )
                card_html += "</div>"
                st.markdown(card_html, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Control Panel")

if is_cloud:
    # AWS ECS control panel
    control_all_start, control_all_stop = st.columns(2)
    
    with control_all_start:
        if st.button("Start all services", type="primary"):
            with st.spinner("Scaling services to 1 instance..."):
                ecs_manager.scale_all(1)
                time.sleep(1)
            st.success("Scale request submitted.")
            st.rerun()
    
    with control_all_stop:
        if st.button("Stop all services"):
            with st.spinner("Stopping all services..."):
                ecs_manager.scale_all(0)
                time.sleep(1)
            st.success("Scale request submitted.")
            st.rerun()
else:
    # Docker control panel
    control_all_start, control_all_stop = st.columns(2)
    
    with control_all_start:
        if st.button("Start all services", type="primary"):
            with st.spinner("Starting Docker services..."):
                from streamlit_app.utils.constants import COMPOSE_FILES
                compose_files = list(COMPOSE_FILES.values())
                success, message = docker_manager.start_services(compose_files[0])  # Start MLflow first
                if success and len(compose_files) > 1:
                    for compose_file in compose_files[1:]:
                        docker_manager.start_services(compose_file)
                time.sleep(1)
            if success:
                st.success("✅ Docker services started successfully.")
            else:
                st.error(f"❌ {message}")
            st.rerun()
    
    with control_all_stop:
        if st.button("Stop all services"):
            with st.spinner("Stopping Docker services..."):
                from streamlit_app.utils.constants import COMPOSE_FILES
                compose_files = list(COMPOSE_FILES.values())
                success, message = docker_manager.stop_all_services(compose_files)
                time.sleep(1)
            if success:
                st.success("✅ Docker services stopped successfully.")
            else:
                st.error(f"❌ {message}")
            st.rerun()

st.markdown("---")

st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ by <strong>Sébastien</strong></p>
    <p>ML Engineer + Product Manager | Seeking combined ML/Product roles</p>
    <p>
        <a href='https://github.com/sebagille' target='_blank'>GitHub</a> |
        <a href='https://linkedin.com/in/seba-gille' target='_blank'>LinkedIn</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
