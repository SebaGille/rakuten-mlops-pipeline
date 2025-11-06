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
    "Monitor and control the AWS ECS services that power the Rakuten MLOps pipeline."
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

if not services_status:
    st.warning("No ECS services are configured for this dashboard.")
else:
    cols = st.columns(len(services_status))
    for idx, (key, info) in enumerate(services_status.items()):
        with cols[idx]:
            running = info.get("runningCount", 0)
            desired = info.get("desired", 0)
            card_html = f"""
            <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
                <h4>{info.get('displayName', key.title())}</h4>
                <p><strong>Status:</strong> {render_status_indicator(info.get('health'))}</p>
                <p><strong>Desired:</strong> {desired} &nbsp;|&nbsp; <strong>Running:</strong> {running}</p>
                <p><strong>AWS Service:</strong> {info.get('serviceArn', '-').split('/')[-1] if info.get('serviceArn') else '-'}</p>
            """
            if info.get("url"):
                card_html += f"<p><a href='{info['url']}' target='_blank'>Open service</a></p>"
            if key == "api" and API_URL:
                card_html += f"<p><a href='{API_URL}/docs' target='_blank'>Open API docs</a></p>"
            if key == "mlflow" and MLFLOW_URL:
                card_html += f"<p><a href='{MLFLOW_URL}' target='_blank'>Open MLflow UI</a></p>"
            if info.get("lastEvent"):
                card_html += f"<p style='font-size:0.85em; color:#555'><em>{info['lastEvent']}</em></p>"
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

st.markdown("#### Individual Services")

control_cols = st.columns(len(services_status) or 1)
for idx, (key, info) in enumerate(services_status.items()):
    with control_cols[idx]:
        st.write(info.get("displayName", key.title()))
        desired = info.get("desired", 0)
        running = info.get("runningCount", 0)

        start_disabled = desired >= 1
        stop_disabled = desired == 0 and running == 0

        if st.button(
            "Start",
            key=f"start_{key}",
            use_container_width=True,
            disabled=start_disabled,
        ):
            with st.spinner(f"Starting {info.get('displayName', key)}..."):
                ecs_manager.scale_service(key, 1)
                time.sleep(1)
            st.success("Scale request submitted.")
            st.rerun()

        if st.button(
            "Stop",
            key=f"stop_{key}",
            use_container_width=True,
            disabled=stop_disabled,
        ):
            with st.spinner(f"Stopping {info.get('displayName', key)}..."):
                ecs_manager.scale_service(key, 0)
                time.sleep(1)
            st.success("Scale request submitted.")
            st.rerun()

st.markdown("---")

st.markdown("### Database")
if rds_status:
    st.write(
        f"**Status:** {rds_status.get('status', 'unknown')}  |  "
        f"**Endpoint:** {rds_status.get('endpoint', 'n/a')}"
    )
else:
    st.info("No RDS instance configured for monitoring.")

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
