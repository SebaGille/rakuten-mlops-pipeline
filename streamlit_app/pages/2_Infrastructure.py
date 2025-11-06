"""Infrastructure Management Page - control AWS ECS services."""

import sys
import time
from pathlib import Path

import streamlit as st

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.ecs_manager import ECSManager
from streamlit_app.utils.constants import (
    MLFLOW_URL,
    API_URL,
    AWS_REGION,
    AWS_ECS_CLUSTER,
    AWS_ALB_URL,
    AWS_RDS_INSTANCE_ID,
)

st.set_page_config(
    page_title="Infrastructure - Rakuten MLOps",
    page_icon="",
    layout="wide"
)


@st.cache_resource
def get_aws_manager() -> ECSManager:
    """Cache the ECS manager across reruns."""

    return ECSManager(
        region=AWS_REGION,
        cluster=AWS_ECS_CLUSTER,
        rds_instance_id=AWS_RDS_INSTANCE_ID,
    )


ecs_manager = get_aws_manager()

st.title("Infrastructure Management")
st.markdown(
    """
    Keep an eye on the managed AWS deployment powering the Rakuten MLOps pipeline: an
    Application Load Balancer fronts two ECS Fargate services (`rakuten-api`, `rakuten-mlflow`),
    backed by Amazon RDS, S3 artifacts, Secrets Manager, and CloudWatch logging.

    Prefer to explore locally? Clone the repository and start the five-container stack with
    `docker-compose` (PostgreSQL, MLflow, FastAPI, Prometheus, Grafana) to mirror this control
    plane on your laptop.
    """
)

if AWS_ALB_URL:
    st.info(f"🔗 Application Load Balancer endpoint: {AWS_ALB_URL}")

services_status = ecs_manager.get_all_services_status()
rds_status = ecs_manager.get_rds_status()


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

if rds_status:
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

control_all_start, control_all_stop = st.columns(2)

with control_all_start:
    if st.button("Start all services", type="primary", use_container_width=True):
        with st.spinner("Scaling services to 1 instance..."):
            ecs_manager.scale_all(1)
            time.sleep(1)
        st.success("Scale request submitted.")
        st.rerun()

with control_all_stop:
    if st.button("Stop all services", use_container_width=True):
        with st.spinner("Stopping all services..."):
            ecs_manager.scale_all(0)
            time.sleep(1)
        st.success("Scale request submitted.")
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
