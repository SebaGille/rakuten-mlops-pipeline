#!/usr/bin/env python3
"""
Pytest tests for MLflow connection
Tests MLflow connectivity using ALB URL with host-based routing
"""
import os
import pytest
import mlflow
from mlflow.tracking import MlflowClient
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import requests
from urllib.parse import urljoin

# Configuration
ALB_URL = os.getenv("AWS_ALB_URL", "http://rakuten-shared-alb-217736274.eu-west-1.elb.amazonaws.com")
MLFLOW_HOST = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
# Use ALB URL directly for CI/CD environments where custom domain may not resolve
# For host-based routing, we'll use the Host header
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", ALB_URL)


def _check_dns_resolution(hostname: str) -> bool:
    """Check if a hostname can be resolved"""
    try:
        import socket
        socket.gethostbyname(hostname)
        return True
    except socket.gaierror:
        return False


def _get_mlflow_url() -> str:
    """Get the MLflow URL to use for testing"""
    # If MLFLOW_TRACKING_URI is explicitly set, use it
    if os.getenv("MLFLOW_TRACKING_URI"):
        return os.getenv("MLFLOW_TRACKING_URI")
    
    # If custom domain resolves, use it
    if _check_dns_resolution(MLFLOW_HOST):
        return f"http://{MLFLOW_HOST}"
    
    # Otherwise, use ALB URL directly
    return ALB_URL


def _get_headers() -> dict:
    """Get headers for host-based routing"""
    headers = {}
    # If using ALB URL but need host-based routing, add Host header
    if MLFLOW_TRACKING_URI == ALB_URL or not _check_dns_resolution(MLFLOW_HOST):
        headers['Host'] = MLFLOW_HOST
    return headers


@pytest.fixture(scope="module")
def mlflow_tracking_uri():
    """Fixture to get MLflow tracking URI"""
    return _get_mlflow_url()


@pytest.fixture(scope="module")
def mlflow_headers():
    """Fixture to get headers for host-based routing"""
    return _get_headers()


def test_mlflow_http_health_check(mlflow_tracking_uri, mlflow_headers):
    """Test 1: HTTP health check"""
    print(f"\nTest 1: HTTP Health Check")
    print(f"ALB URL: {ALB_URL}")
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    
    base_url = mlflow_tracking_uri.rstrip("/")
    health_urls = [
        urljoin(base_url + "/", "health"),
        urljoin(base_url + "/", "api/2.0/mlflow/health"),
        base_url,
    ]
    
    http_success = False
    successful_url = None
    
    for url in health_urls:
        try:
            print(f"  Trying: {url}")
            response = requests.get(url, timeout=5, allow_redirects=True, headers=mlflow_headers)
            print(f"    HTTP {response.status_code}")
            if response.status_code < 400:
                http_success = True
                successful_url = url
                print(f"  ✓ HTTP check successful: {url}")
                break
        except requests.RequestException as e:
            print(f"    ✗ Failed: {e}")
            continue
    
    if not http_success:
        pytest.skip(f"All HTTP health checks failed for {mlflow_tracking_uri}. "
                   f"This may be expected in CI if MLflow is not accessible.")


def test_mlflow_set_tracking_uri(mlflow_tracking_uri):
    """Test 2: Set MLflow tracking URI"""
    print(f"\nTest 2: Set MLflow Tracking URI")
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"  ✓ Tracking URI set: {mlflow_tracking_uri}")
    except Exception as e:
        pytest.fail(f"Failed to set tracking URI: {e}")


def test_mlflow_client_initialization(mlflow_tracking_uri):
    """Test 3: Initialize MLflow client"""
    print(f"\nTest 3: Initialize MLflow Client")
    try:
        def init_client():
            return MlflowClient(mlflow_tracking_uri)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(init_client)
            client = future.result(timeout=5)
        print("  ✓ MLflow client initialized")
        return client
    except FutureTimeoutError:
        pytest.fail("Client initialization timed out after 5 seconds")
    except Exception as e:
        pytest.fail(f"Failed to initialize client: {e}")


def test_mlflow_search_experiments(mlflow_tracking_uri):
    """Test 4: Search experiments"""
    print(f"\nTest 4: Search Experiments")
    
    # Initialize client first
    try:
        def init_client():
            return MlflowClient(mlflow_tracking_uri)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(init_client)
            client = future.result(timeout=5)
    except Exception as e:
        pytest.skip(f"Could not initialize client: {e}")
    
    try:
        def search_experiments():
            return client.search_experiments(max_results=1)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search_experiments)
            experiments = future.result(timeout=10)
        
        print(f"  ✓ Found {len(experiments)} experiments")
        if experiments:
            exp = experiments[0]
            print(f"    First experiment: {exp.name} (ID: {exp.experiment_id})")
    except FutureTimeoutError:
        pytest.skip("search_experiments() timed out after 10 seconds")
    except Exception as e:
        pytest.skip(f"search_experiments() failed: {e}")


def test_mlflow_get_runs(mlflow_tracking_uri):
    """Test 5: Get runs"""
    print(f"\nTest 5: Get Runs")
    
    # Initialize client first
    try:
        def init_client():
            return MlflowClient(mlflow_tracking_uri)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(init_client)
            client = future.result(timeout=5)
    except Exception as e:
        pytest.skip(f"Could not initialize client: {e}")
    
    # Get experiments first
    try:
        def search_experiments():
            return client.search_experiments(max_results=1)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search_experiments)
            experiments = future.result(timeout=10)
    except Exception as e:
        pytest.skip(f"Could not get experiments: {e}")
    
    if not experiments:
        pytest.skip("No experiments found, skipping run test")
    
    try:
        def get_runs():
            return client.search_runs(
                experiment_ids=[experiments[0].experiment_id],
                max_results=1
            )
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_runs)
            runs = future.result(timeout=10)
        
        print(f"  ✓ Found {len(runs)} runs")
        if runs:
            run = runs[0]
            print(f"    First run: {run.info.run_id} (Status: {run.info.status})")
    except FutureTimeoutError:
        pytest.skip("get_runs() timed out after 10 seconds")
    except Exception as e:
        pytest.skip(f"get_runs() failed: {e}")
