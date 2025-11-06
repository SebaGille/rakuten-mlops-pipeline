"""
Rakuten MLOps Showcase - Main Application

An interactive demonstration of ML Engineering + Product Management skills
through a complete MLOps pipeline for product classification.

Author: Sébastien
"""
import streamlit as st
from pathlib import Path
import sys

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
        
        An **interactive prototype** showcasing a complete MLOps pipeline for product classification.
        
        This demo lets you:
        - Verify Docker infrastructure required for the prototype
        - Select a sample of data from the Rakuten dataset and train a model
        - Make real-time predictions using your model and your own product
        - Monitor model performance
        
        Behind the scene, a complete MLOps pipeline is at work, tracking training dataset, model weights and artifacts, promoting best models...
        
        https://github.com/SebaGille/rakuten-mlops-pipeline
        """)
    
    st.markdown("---")
    
    # Feature overview
    st.markdown("## Features Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Dataset Explorer
        - Browse dataset statistics
        - Category distribution analysis
        - Filter and sample data
        - Visualize product categories
        
        **Go to: Dataset** (sidebar)
        """)
        
        st.markdown("""
        ### Infrastructure Management
        - Real-time service monitoring
        - One-click start/stop controls
        - Container/ECS health checks
        - Resource usage tracking
        
        **Go to: Infrastructure** (sidebar)
        """)
    
    with col2:
        st.markdown("""
        ### Interactive Training
        - Browse and filter datasets
        - Configure hyperparameters
        - Choose text-only or multimodal
        - Track experiments in MLflow
        - Compare models side-by-side
        - Auto-promote champion models
        
        **Go to: Training** (sidebar)
        """)
        
        st.markdown("""
        ### Live Predictions
        - Enter product text + upload images
        - Real-time classification
        - Confidence scores display
        - Prediction history tracking
        - Inference latency metrics
        
        **Go to: Predictions** (sidebar)
        """)
    
    with col3:
        st.markdown("""
        ### Monitoring Dashboard
        - Drift detection (Evidently)
        - System metrics tracking
        - Model performance analysis
        - Data quality checks
        - Prometheus metrics (local only)
        - Grafana dashboards (local only)
        
        **Go to: Monitoring** (sidebar)
        """)
    
    st.markdown("---")
    
    # Deployment information
    st.markdown("## Deployment Information")
    
    if AWS_ALB_URL:
        st.info(f"""
        **AWS Deployment Detected**
        
        This application is running on **AWS ECS** with the following configuration:
        - **Load Balancer:** {AWS_ALB_URL}
        - **Services:** MLflow and FastAPI API running on ECS
        - **Database:** AWS RDS (PostgreSQL)
        - **Storage:** S3 for model artifacts
        
        **Note:** Grafana and Prometheus monitoring are not available in AWS deployment.
        For full monitoring capabilities, use the local deployment with Docker Compose.
        """)
    else:
        st.info("""
        **Local Deployment**
        
        This application is running **locally** with Docker Compose. The following services are available:
        - **PostgreSQL:** Database for MLflow backend
        - **MLflow:** Model tracking and registry (http://localhost:5000)
        - **FastAPI:** Prediction API (http://localhost:8000)
        - **Prometheus:** Metrics collection (http://localhost:9090)
        - **Grafana:** Monitoring dashboards (http://localhost:3000)
        
        To start services, use the **Infrastructure** page or run:
        ```bash
        docker-compose -f docker-compose.api.yml up -d
        docker-compose -f docker-compose.monitor.yml up -d
        ```
        """)
    
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

