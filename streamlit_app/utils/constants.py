"""Constants and configuration for Streamlit application."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Product categories mapping
PRODUCT_CATEGORIES = {
    10: "📚 Books",
    40: "🎮 Games & Toys",
    50: "🏡 Home & Garden",
    60: "🎯 Video Games",
    1140: "🗿 Figurines",
    1160: "🃏 Trading Cards",
    1180: "👶 Children's Books",
    1280: "📰 Magazines",
    1281: "💬 Comics",
    1300: "📎 Office Supplies",
    1301: "👜 Bags & Luggage",
    1302: "🔧 DIY Tools",
    1320: "💄 Health & Beauty",
    1560: "🪑 Furniture",
    1920: "🐾 Pet Supplies",
    1940: "🍕 Food & Beverages",
    2060: "🎨 Decoration",
    2220: "👶 Baby Products",
    2280: "🏊 Pool & Accessories",
    2403: "📖 Books (Non-fiction)",
    2462: "🎲 Board Games",
    2522: "🎉 Party Supplies",
    2582: "⚽ Sports Equipment",
    2583: "🛏️ Home Textiles",
    2585: "✏️ School Supplies",
    2705: "🍳 Kitchen Appliances",
    2905: "🧹 Cleaning Products"
}

# Service URLs (override via environment variables or Streamlit secrets for cloud mode)
# MLFLOW_TRACKING_URI: Tracking URI for MLflow client (uses host-based routing)
# MLFLOW_URL: Full URL for UI links (same as tracking URI)
# Note: ALB is configured for host-based routing using mlflow.rakuten.dev or mlflow.rakuten.local
# When AWS_ALB_URL is set, we use the ALB URL with Host header for host-based routing

def _get_config_value(key: str, default: str = "") -> str:
    """Get configuration value from Streamlit secrets or environment variables"""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except (ImportError, AttributeError, KeyError, FileNotFoundError):
        # FileNotFoundError occurs when secrets.toml doesn't exist
        return os.getenv(key, default)

# AWS / ECS defaults - use _get_config_value to read from Streamlit secrets or environment variables
AWS_REGION = _get_config_value("AWS_REGION", "eu-west-1")
AWS_ECS_CLUSTER = _get_config_value("AWS_ECS_CLUSTER", "rakuten-mlops-cluster")
AWS_ALB_URL = _get_config_value("AWS_ALB_URL", "")
AWS_RDS_INSTANCE_ID = _get_config_value("RDS_INSTANCE_ID", "rakuten-mlflow-db")


def _alb_prefixed(path: str) -> str:
    if AWS_ALB_URL:
        return f"{AWS_ALB_URL.rstrip('/')}/{path.lstrip('/')}"
    return ""

MLFLOW_HOST = _get_config_value("MLFLOW_HOST", "mlflow.rakuten.dev")
# Use ALB URL directly since hostname might not resolve, but set Host header for routing
MLFLOW_TRACKING_URI = _get_config_value(
    "MLFLOW_TRACKING_URI",
    AWS_ALB_URL.rstrip("/") if AWS_ALB_URL else "http://localhost:5000",
)
MLFLOW_URL = _get_config_value(
    "MLFLOW_URL",
    AWS_ALB_URL.rstrip("/") if AWS_ALB_URL else "http://localhost:5000",
)
API_HOST = _get_config_value("API_HOST", "api.rakuten.dev")
# Use ALB URL directly since hostname might not resolve, but set Host header for routing
# The ALB routes based on Host header (api.rakuten.dev), not path prefix
API_URL = _get_config_value(
    "API_URL",
    AWS_ALB_URL.rstrip("/") if AWS_ALB_URL else "http://localhost:8000",
)
PROMETHEUS_URL = _get_config_value("PROMETHEUS_URL", "")
GRAFANA_URL = _get_config_value("GRAFANA_URL", "")

# Docker compose files
COMPOSE_FILES = {
    "mlflow": "docker-compose.mlflow.yml",
    "api": "docker-compose.api.yml",
    "monitor": "docker-compose.monitor.yml"
}

# Container names
CONTAINERS = {
    "postgres": "sep25_cmlops_rakuten-postgres-1",
    "mlflow": "sep25_cmlops_rakuten-mlflow-1",
    "api": "sep25_cmlops_rakuten-rakuten_api-1",
    "prometheus": "sep25_cmlops_rakuten-prometheus-1",
    "grafana": "sep25_cmlops_rakuten-grafana-1"
}

# Color scheme
COLORS = {
    "primary": "#1f77b4",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "dark": "#2c3e50",
    "light": "#ecf0f1"
}

# Training configurations
SAMPLE_SIZES = [100, 1000, 5000, 10000, "Full Dataset"]

# Model hyperparameters defaults
DEFAULT_HYPERPARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "max_iter": 200,
    "random_state": 42,
    "solver": "lbfgs"
}

