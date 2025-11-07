"""Configuration constants for the Streamlit application"""
import os
import logging

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Timeout configurations (in seconds)
MLFLOW_CLIENT_INIT_TIMEOUT = int(os.getenv("MLFLOW_CLIENT_INIT_TIMEOUT", "3"))
MLFLOW_CONNECTION_TIMEOUT = int(os.getenv("MLFLOW_CONNECTION_TIMEOUT", "5"))
MLFLOW_REQUEST_TIMEOUT = int(os.getenv("MLFLOW_REQUEST_TIMEOUT", "30"))
API_REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "30"))
API_HEALTH_CHECK_TIMEOUT = int(os.getenv("API_HEALTH_CHECK_TIMEOUT", "5"))
MANAGER_INIT_TIMEOUT = int(os.getenv("MANAGER_INIT_TIMEOUT", "5"))

# Cache TTL (in seconds)
MLFLOW_CACHE_TTL = int(os.getenv("MLFLOW_CACHE_TTL", "60"))  # 1 minute
MODEL_REGISTRY_CACHE_TTL = int(os.getenv("MODEL_REGISTRY_CACHE_TTL", "120"))  # 2 minutes

# API call limits
MLFLOW_MAX_RESULTS = int(os.getenv("MLFLOW_MAX_RESULTS", "1000"))
MLFLOW_DEFAULT_MAX_RESULTS = int(os.getenv("MLFLOW_DEFAULT_MAX_RESULTS", "10"))

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "1.0"))

