"""Constants and configuration for Streamlit application"""
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

# Service URLs
MLFLOW_URL = "http://localhost:5000"
API_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"

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

