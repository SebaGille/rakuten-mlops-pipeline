"""Monitoring and Drift Detection Page"""
import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import json
import os
import io
from typing import Optional

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.constants import (
    PROMETHEUS_URL, GRAFANA_URL, API_URL
)
from streamlit_app.utils.prediction_manager import PredictionManager

# Optional S3 support for drift reports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


def _get_s3_client():
    """Get S3 client if configured"""
    if not S3_AVAILABLE:
        return None
    
    try:
        import streamlit as st
        s3_bucket = st.secrets.get("S3_DATA_BUCKET", os.getenv("S3_DATA_BUCKET", ""))
        aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
        aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
        aws_region = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
    except (ImportError, AttributeError, KeyError):
        s3_bucket = os.getenv("S3_DATA_BUCKET", "")
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    
    if not s3_bucket:
        return None
    
    try:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        ), s3_bucket, os.getenv("S3_DATA_PREFIX", "data/")
    except Exception:
        return None


def _load_json_from_s3(s3_client, bucket, prefix, s3_key: str) -> Optional[dict]:
    """Load JSON from S3"""
    try:
        full_key = f"{prefix.rstrip('/')}/{s3_key.lstrip('/')}"
        response = s3_client.get_object(Bucket=bucket, Key=full_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
            return None
        return None
    except Exception:
        return None

st.set_page_config(
    page_title="Monitoring - Rakuten MLOps",
    page_icon="",
    layout="wide"
)

# Header
st.title("📊 Monitoring & Observability")
st.markdown("Monitor model performance, system metrics, and detect data drift")
st.markdown("---")

# ==================== SECTION 1: System Metrics ====================
st.markdown("### 🎯 System Metrics Overview")

st.info("""
📊 **Real-time metrics** are collected by Prometheus from the API service.

Metrics include:
- Request count and latency
- Prediction distribution
- Input text length statistics
- Error rates
""")

# Load inference log using PredictionManager (supports S3)
prediction_manager = PredictionManager(API_URL, PROJECT_ROOT)
df_log = None

try:
    df_log = prediction_manager.get_prediction_history(limit=10000)
except Exception as e:
    st.error(f"Error loading inference log: {str(e)}")
    st.info("The log file may be corrupted or not accessible.")
    df_log = None

if df_log is not None and not df_log.empty:
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df_log))
    
    with col2:
        if 'predicted_class' in df_log.columns:
            unique_classes = df_log['predicted_class'].nunique()
            st.metric("Unique Classes Predicted", unique_classes)
        else:
            st.metric("Unique Classes Predicted", "N/A")
    
    with col3:
        if 'designation_length' in df_log.columns:
            avg_des_len = df_log['designation_length'].mean()
            st.metric("Avg Designation Length", f"{avg_des_len:.0f} chars")
        else:
            st.metric("Avg Designation Length", "N/A")
    
    with col4:
        if 'description_length' in df_log.columns:
            avg_desc_len = df_log['description_length'].mean()
            st.metric("Avg Description Length", f"{avg_desc_len:.0f} chars")
        else:
            st.metric("Avg Description Length", "N/A")
    
    st.markdown("---")
    
    # Predictions over time
    st.markdown("#### 📈 Predictions Over Time")
    
    if 'timestamp' in df_log.columns:
        df_time = df_log.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
        df_time = df_time.dropna(subset=['timestamp'])
        df_time['date'] = df_time['timestamp'].dt.date
        predictions_by_date = df_time.groupby('date').size().reset_index(name='count')
        
        fig_time = px.line(
            predictions_by_date,
            x='date',
            y='count',
            title='Daily Prediction Volume',
            labels={'date': 'Date', 'count': 'Number of Predictions'},
            markers=True
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Timestamp data not available")
    
    st.markdown("---")
    
    # Class distribution
    st.markdown("#### 🏷️ Prediction Class Distribution")
    
    if 'predicted_class' in df_log.columns:
        class_dist = df_log['predicted_class'].value_counts().head(15)
        df_class = pd.DataFrame({
            'Category': class_dist.index,
            'Count': class_dist.values
        })
        
        fig_class = px.bar(
            df_class,
            x='Category',
            y='Count',
            title='Top 15 Predicted Categories',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_class.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_class, use_container_width=True)
    else:
        st.info("Prediction class data not available")
    
    st.markdown("---")
    
    # Text length distribution
    st.markdown("#### 📏 Input Text Length Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'designation_length' in df_log.columns:
            fig_des = px.histogram(
                df_log,
                x='designation_length',
                title='Designation Length Distribution',
                nbins=50,
                color_discrete_sequence=['#3498db']
            )
            fig_des.update_layout(height=300)
            st.plotly_chart(fig_des, use_container_width=True)
        else:
            st.info("Designation length data not available")
    
    with col2:
        if 'description_length' in df_log.columns:
            fig_desc = px.histogram(
                df_log,
                x='description_length',
                title='Description Length Distribution',
                nbins=50,
                color_discrete_sequence=['#e74c3c']
            )
            fig_desc.update_layout(height=300)
            st.plotly_chart(fig_desc, use_container_width=True)
        else:
            st.info("Description length data not available")
else:
    st.warning("No prediction logs found yet. Make some predictions first!")
    st.info("Logs are stored in S3 (if configured) or locally at: `data/monitoring/inference_log.csv`")

st.markdown("---")

# ==================== SECTION 2: Data Drift ====================
st.markdown("### 📉 Data Drift Detection")

st.info("""
**Data drift** occurs when the distribution of input data changes over time, 
potentially degrading model performance.

We use **Evidently** to detect:
- Feature drift (text characteristics)
- Prediction drift (category distribution changes)
- Target drift (if labels are available)
""")

# Check for Evidently reports (local or S3)
drift_status_path = PROJECT_ROOT / "reports" / "evidently" / "drift_status.json"
drift_status = None

# Try S3 first if configured
s3_result = _get_s3_client()
if s3_result:
    s3_client, s3_bucket, s3_prefix = s3_result
    drift_status = _load_json_from_s3(s3_client, s3_bucket, s3_prefix, "reports/evidently/drift_status.json")

# Fallback to local file
if drift_status is None and drift_status_path.exists():
    try:
        with open(drift_status_path, 'r') as f:
            drift_status = json.load(f)
    except Exception as e:
        st.warning(f"Error loading drift status: {e}")
        drift_status = None

if drift_status:
    st.markdown("#### 🚨 Current Drift Status")
    
    drift_detected = drift_status.get('drift_detected', False)
    
    if drift_detected:
        st.error("⚠️ **DRIFT DETECTED** - Model retraining recommended!")
    else:
        st.success("✅ **No Significant Drift Detected** - Model is stable")
    
    # Display drift metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Drift Score",
            f"{drift_status.get('drift_score', 0):.3f}",
            delta="Threshold: 0.5" if drift_status.get('drift_score', 0) < 0.5 else None
        )
    
    with col2:
        st.metric(
            "Drifted Features",
            drift_status.get('n_drifted_features', 0)
        )
    
    with col3:
        st.metric(
            "Last Check",
            drift_status.get('timestamp', 'Unknown')
        )
    
    st.markdown("---")
    
    # Detailed drift information
    if 'drift_by_columns' in drift_status:
        st.markdown("#### 📊 Drift by Feature")
        
        drift_details = []
        for col_name, col_drift in drift_status['drift_by_columns'].items():
            drift_details.append({
                'Feature': col_name,
                'Drift Detected': '⚠️ Yes' if col_drift.get('drift_detected', False) else '✅ No',
                'Drift Score': f"{col_drift.get('drift_score', 0):.3f}",
                'Method': col_drift.get('stattest_name', 'N/A')
            })
        
        df_drift = pd.DataFrame(drift_details)
        st.dataframe(df_drift, use_container_width=True, height=300)
    
    # Link to full HTML report
    evidently_html_path = PROJECT_ROOT / "reports" / "evidently" / "evidently_report.html"
    html_report_available = False
    
    # Check if HTML report exists (local or S3)
    if s3_result:
        s3_client, s3_bucket, s3_prefix = s3_result
        try:
            full_key = f"{s3_prefix.rstrip('/')}/reports/evidently/evidently_report.html"
            s3_client.head_object(Bucket=s3_bucket, Key=full_key)
            html_report_available = True
            html_report_path = f"s3://{s3_bucket}/{full_key}"
        except ClientError:
            pass
    
    if not html_report_available and evidently_html_path.exists():
        html_report_available = True
        html_report_path = str(evidently_html_path.relative_to(PROJECT_ROOT))
    
    if html_report_available:
        st.markdown("---")
        st.markdown("#### 📄 Full Evidently Report")
        
        st.info(f"""
        A detailed HTML report is available at:
        `{html_report_path}`
        
        The report includes:
        - Feature distributions comparison
        - Statistical tests details
        - Interactive visualizations
        - Recommendations
        """)
        
        # Note: Cannot open file:// URLs on Streamlit Cloud, so just show the path
        if html_report_path.startswith("s3://"):
            st.info("💡 Download the report from S3 to view it in your browser.")
        elif html_report_path.startswith("/") or "\\" in html_report_path:
            st.info("💡 Open the report file from your local filesystem to view it in your browser.")

else:
    st.warning("No drift reports found yet.")
    st.info("""
    **To generate a drift report:**
    
    1. Make sure you have some predictions logged
    2. Run: `python src/monitoring/generate_evidently.py`
    3. The report will appear here
    
    **Or use the automated schedule:**
    - Prefect runs drift checks daily at 9:00 UTC
    - Retraining is triggered automatically if drift is detected
    """)
    
    # Show how to generate report
    with st.expander("📋 Manual Drift Report Generation"):
        st.code("""
# Generate Evidently drift report
python src/monitoring/generate_evidently.py

# Or trigger via Prefect
prefect deployment run "monitor-and-retrain/monitor-and-retrain-daily"
        """, language="bash")

st.markdown("---")

# ==================== SECTION 3: Model Performance ====================
st.markdown("### 📈 Model Performance Tracking")

st.info("""
Track model performance over time to identify degradation and opportunities for improvement.
""")

# Load metrics from MLflow if available
from streamlit_app.utils.mlflow_manager import MLflowManager
from streamlit_app.utils.constants import MLFLOW_URL, MLFLOW_TRACKING_URI

mlflow_manager = MLflowManager(MLFLOW_TRACKING_URI)

if mlflow_manager.check_connection():
    st.markdown("#### 🏆 Production Model Metrics")
    
    # Get production model details
    models = mlflow_manager.get_registered_models()
    
    if models:
        for model in models:
            if model['champion_version']:
                champion_details = mlflow_manager.get_model_version_details(
                    model['name'],
                    alias='champion'
                )
                
                if champion_details:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{champion_details.get('accuracy', 0):.4f}")
                    
                    with col2:
                        st.metric("F1-Weighted", f"{champion_details.get('f1_weighted', 0):.4f}")
                    
                    with col3:
                        st.metric("Model Version", f"v{champion_details['version']}")
                    
                    with col4:
                        days_old = (pd.Timestamp.now() - champion_details['creation_timestamp']).days
                        st.metric("Model Age", f"{days_old} days")
                    
                    st.markdown("---")
    
    # Get experiment history
    experiments, error_msg = mlflow_manager.get_experiments()
    if error_msg:
        st.warning(f"⚠️ Failed to retrieve experiments: {error_msg}")
    elif experiments:
        exp_names = [e['name'] for e in experiments if not e['name'].startswith('Default')]
        if exp_names:
            selected_exp = st.selectbox("Select experiment:", exp_names, index=0)
            exp_id = next(e['experiment_id'] for e in experiments if e['name'] == selected_exp)
            
            df_runs = mlflow_manager.get_runs(exp_id, max_results=50)
            
            if not df_runs.empty:
                st.markdown("#### 📊 Model Performance Over Time")
                
                # Metrics over time
                fig = px.scatter(
                    df_runs,
                    x='start_time',
                    y=['accuracy', 'f1_weighted'],
                    title='Model Metrics Evolution',
                    labels={'value': 'Score', 'variable': 'Metric', 'start_time': 'Training Date'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance statistics
                st.markdown("#### 📊 Performance Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Accuracy Statistics:**")
                    acc_stats = df_runs['accuracy'].describe()
                    st.write(f"- Mean: {acc_stats['mean']:.4f}")
                    st.write(f"- Std: {acc_stats['std']:.4f}")
                    st.write(f"- Min: {acc_stats['min']:.4f}")
                    st.write(f"- Max: {acc_stats['max']:.4f}")
                
                with col2:
                    st.markdown("**F1-Weighted Statistics:**")
                    f1_stats = df_runs['f1_weighted'].describe()
                    st.write(f"- Mean: {f1_stats['mean']:.4f}")
                    st.write(f"- Std: {f1_stats['std']:.4f}")
                    st.write(f"- Min: {f1_stats['min']:.4f}")
                    st.write(f"- Max: {f1_stats['max']:.4f}")
else:
    st.warning("MLflow server not accessible. Start infrastructure first.")

st.markdown("---")

# ==================== SECTION 4: External Dashboards ====================
st.markdown("### 🔗 External Monitoring Dashboards")

st.info("""
The MLOps pipeline includes several external monitoring tools.
Click the links below to access them (make sure services are running).
""")

# Dashboard cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 2rem; border-radius: 10px; color: white;'>
        <h3 style='color: white;'>🔍 Prometheus</h3>
        <p>Metrics collection and querying</p>
        <p><strong>Port:</strong> 9090</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Time-series metrics</li>
            <li>PromQL queries</li>
            <li>Target health monitoring</li>
            <li>Alert rules</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"[🔗 Open Prometheus]({PROMETHEUS_URL})")

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 2rem; border-radius: 10px; color: white;'>
        <h3 style='color: white;'>📊 Grafana</h3>
        <p>Visualization and dashboards</p>
        <p><strong>Port:</strong> 3000</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Real-time dashboards</li>
            <li>Custom panels</li>
            <li>Alert notifications</li>
            <li>Historical trends</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"[🔗 Open Grafana]({GRAFANA_URL})")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 2rem; border-radius: 10px; color: #333;'>
        <h3>📊 MLflow UI</h3>
        <p>Experiment tracking and model registry</p>
        <p><strong>Port:</strong> 5000</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Experiment comparison</li>
            <li>Model registry</li>
            <li>Artifact storage</li>
            <li>Run parameters & metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"[🔗 Open MLflow]({MLFLOW_URL})")

with col4:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                padding: 2rem; border-radius: 10px; color: #333;'>
        <h3>🚀 API Documentation</h3>
        <p>Interactive API documentation</p>
        <p><strong>Port:</strong> 8000</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>OpenAPI/Swagger UI</li>
            <li>Try API endpoints</li>
            <li>Request/response schemas</li>
            <li>Authentication docs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"[🔗 Open API Docs]({API_URL}/docs)")

st.markdown("---")

# Monitoring best practices
with st.expander("📚 Monitoring Best Practices"):
    st.markdown("""
    ### Key Monitoring Principles
    
    **1. Four Golden Signals:**
    - **Latency**: How long does a prediction take?
    - **Traffic**: How many predictions per second?
    - **Errors**: What's the error rate?
    - **Saturation**: Are resources (CPU, memory) maxed out?
    
    **2. ML-Specific Monitoring:**
    - **Data Drift**: Distribution changes in input features
    - **Concept Drift**: Relationship between features and target changes
    - **Model Performance**: Accuracy/F1 degradation over time
    - **Prediction Distribution**: Unusual patterns in outputs
    
    **3. Alerting Strategy:**
    - Set thresholds for drift scores
    - Alert on error rate spikes
    - Monitor prediction latency
    - Track resource usage trends
    
    **4. Proactive Maintenance:**
    - Schedule regular model retraining
    - Review drift reports weekly
    - Update model when drift detected
    - A/B test new models before full deployment
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> Enable automated drift monitoring with Prefect for continuous oversight</p>
    <p>📊 Check dashboards regularly to catch issues early</p>
</div>
""", unsafe_allow_html=True)

