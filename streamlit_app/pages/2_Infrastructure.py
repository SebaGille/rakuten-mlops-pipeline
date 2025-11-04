"""Infrastructure Management Page - Docker Services Control"""
import streamlit as st
from pathlib import Path
import sys
import time

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.docker_manager import DockerManager
from streamlit_app.utils.constants import (
    CONTAINERS, COMPOSE_FILES, MLFLOW_URL, API_URL, 
    PROMETHEUS_URL, GRAFANA_URL, COLORS
)

st.set_page_config(
    page_title="Infrastructure - Rakuten MLOps",
    page_icon="",
    layout="wide"
)

# Initialize Docker Manager
@st.cache_resource
def get_docker_manager():
    return DockerManager(str(PROJECT_ROOT))

docker_manager = get_docker_manager()

# Header
st.title("Infrastructure Management")
st.markdown("Monitor and control Docker services for the MLOps pipeline.")
st.markdown("All machines need to be up and healthy if you want to use Training, Predicting and Monitoring features")
st.markdown("---")

# Status indicator function
def status_indicator(is_healthy: bool, is_running: bool) -> str:
    """Return colored status indicator"""
    if is_healthy and is_running:
        return "🟢 Healthy"
    elif is_running:
        return "🟡 Running (no health check)"
    else:
        return "🔴 Stopped"

# Service Status Overview
st.markdown("### Service Status Overview")

# Status legend
st.markdown("""
<div style='margin-bottom: 1rem; padding: 0.5rem; font-size: 0.9em; color: #666;'>
    <strong>Status Legend:</strong> 
    🟢 Healthy | 
    🟡 Running (no health check) | 
    🔴 Stopped
</div>
""", unsafe_allow_html=True)

# Get status of all services
services_status = docker_manager.get_all_services_status(CONTAINERS)

# Display service cards
st.markdown("#### Core Services")

col1, col2, col3 = st.columns(3)

with col1:
    # PostgreSQL Status
    postgres_status = services_status.get('postgres', {})
    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
        <h4>PostgreSQL</h4>
        <p><strong>Status:</strong> {status_indicator(postgres_status.get('health') == 'healthy', postgres_status.get('running'))}</p>
        <p><strong>Port:</strong> 5432</p>
        <p><strong>Purpose:</strong> MLflow backend store</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # MLflow Status
    mlflow_status = services_status.get('mlflow', {})
    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
        <h4>MLflow Server</h4>
        <p><strong>Status:</strong> {status_indicator(mlflow_status.get('health') == 'healthy', mlflow_status.get('running'))}</p>
        <p><strong>Port:</strong> 5000</p>
        <p><strong>Purpose:</strong> Experiment tracking</p>
        {'<p><a href="' + MLFLOW_URL + '" target="_blank">Open MLflow UI</a></p>' if mlflow_status.get('running') else ''}
    </div>
    """, unsafe_allow_html=True)

with col3:
    # API Status
    api_status = services_status.get('api', {})
    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
        <h4>Rakuten API</h4>
        <p><strong>Status:</strong> {status_indicator(api_status.get('health') == 'healthy', api_status.get('running'))}</p>
        <p><strong>Port:</strong> 8000</p>
        <p><strong>Purpose:</strong> Model serving</p>
        {'<p><a href="' + API_URL + '/docs" target="_blank">Open API Docs</a></p>' if api_status.get('running') else ''}
    </div>
    """, unsafe_allow_html=True)

st.markdown("#### Monitoring Services")

col1, col2, col3 = st.columns(3)

with col1:
    # Prometheus Status
    prometheus_status = services_status.get('prometheus', {})
    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
        <h4>Prometheus</h4>
        <p><strong>Status:</strong> {status_indicator(prometheus_status.get('health') == 'healthy', prometheus_status.get('running'))}</p>
        <p><strong>Port:</strong> 9090</p>
        <p><strong>Purpose:</strong> Metrics collection</p>
        {'<p><a href="' + PROMETHEUS_URL + '" target="_blank">Open Prometheus</a></p>' if prometheus_status.get('running') else ''}
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Grafana Status
    grafana_status = services_status.get('grafana', {})
    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;'>
        <h4>Grafana</h4>
        <p><strong>Status:</strong> {status_indicator(grafana_status.get('health') == 'healthy', grafana_status.get('running'))}</p>
        <p><strong>Port:</strong> 3000</p>
        <p><strong>Purpose:</strong> Dashboards</p>
        {'<p><a href="' + GRAFANA_URL + '" target="_blank">Open Grafana</a></p>' if grafana_status.get('running') else ''}
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Summary
    running_count = sum(1 for s in services_status.values() if s.get('running'))
    total_count = len(CONTAINERS)
    st.metric(
        label="Services Running",
        value=f"{running_count}/{total_count}",
        delta=f"{running_count - total_count} to start" if running_count < total_count else "All running!"
    )

st.markdown("---")

# Control Panel
st.markdown("### Control Panel")

col1, col2 = st.columns(2)

with col1:
    st.markdown("If they are not all healthy, click below to restart them")
    
    if st.button("Restart all machines", type="secondary", use_container_width=True):
        with st.spinner("🔄 Restarting all services... This may take 1-2 minutes. Please wait..."):
            progress = st.progress(0)
            
            # Restart MLflow stack
            success1, msg1 = docker_manager.restart_services(COMPOSE_FILES['mlflow'])
            progress.progress(33)
            time.sleep(3)
            
            # Restart API stack
            success2, msg2 = docker_manager.restart_services(COMPOSE_FILES['api'])
            progress.progress(66)
            time.sleep(3)
            
            # Restart monitoring stack
            success3, msg3 = docker_manager.restart_services(COMPOSE_FILES['monitor'])
            progress.progress(100)
            
            if success1 and success2 and success3:
                st.success("✅ All 5 services restarted successfully!")
                time.sleep(2)
                st.rerun()
            else:
                error_messages = []
                if not success1:
                    error_messages.append(f"MLflow stack: {msg1}")
                if not success2:
                    error_messages.append(f"API stack: {msg2}")
                if not success3:
                    error_messages.append(f"Monitoring stack: {msg3}")
                st.error("❌ Some services failed to restart:\n" + "\n".join(error_messages))
    
    st.markdown("---")
    

# Resource Usage
st.markdown("### Resource Usage")

if running_count > 0:
    cols = st.columns(len([k for k, v in services_status.items() if v.get('running')]))
    col_idx = 0
    
    for service_name, status in services_status.items():
        if status.get('running'):
            with cols[col_idx]:
                stats = docker_manager.get_container_stats(CONTAINERS[service_name])
                if stats:
                    st.metric(
                        label=f" {service_name}",
                        value=f"{stats['memory_usage_mb']:.0f} MB",
                        delta=f"{stats['cpu_percent']:.1f}% CPU"
                    )
                else:
                    st.metric(
                        label=f" {service_name}",
                        value="Stats unavailable"
                    )
                col_idx += 1
else:
    st.info("No services running. Start services to see resource usage.")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ by <strong>Sébastien</strong></p>
    <p>ML Engineer + Product Manager | Seeking combined ML/Product roles</p>
    <p>
        <a href='https://github.com/sebagille' target='_blank'>GitHub</a> | 
        <a href='https://linkedin.com/in/seba-gille' target='_blank'>LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)

