#!/usr/bin/env python3
"""
Python script to test MLflow connection
Run this to verify MLflow connectivity before debugging Streamlit
"""
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import requests
from urllib.parse import urljoin

# Configuration
ALB_URL = os.getenv("AWS_ALB_URL", "http://rakuten-shared-alb-217736274.eu-west-1.elb.amazonaws.com")
MLFLOW_HOST = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
# Use host-based routing: ALB routes based on Host header
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://{MLFLOW_HOST}")

print("=" * 60)
print("MLflow Connection Test (Python)")
print("=" * 60)
print(f"ALB URL: {ALB_URL}")
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print()

# Test 1: HTTP health check
print("Test 1: HTTP Health Check")
base_url = MLFLOW_TRACKING_URI.rstrip("/")
health_urls = [
    urljoin(base_url + "/", "health"),
    urljoin(base_url + "/", "api/2.0/mlflow/health"),
    base_url,
]

http_success = False
for url in health_urls:
    try:
        print(f"  Trying: {url}")
        response = requests.get(url, timeout=5, allow_redirects=True)
        print(f"    HTTP {response.status_code}")
        if response.status_code < 400:
            http_success = True
            print(f"  ✓ HTTP check successful: {url}")
            break
    except requests.RequestException as e:
        print(f"    ✗ Failed: {e}")
        continue

if not http_success:
    print("  ✗ All HTTP health checks failed")
    sys.exit(1)
print()

# Test 2: Set tracking URI
print("Test 2: Set MLflow Tracking URI")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"  ✓ Tracking URI set: {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"  ✗ Failed to set tracking URI: {e}")
    sys.exit(1)
print()

# Test 3: Initialize MLflow client
print("Test 3: Initialize MLflow Client")
try:
    def init_client():
        return MlflowClient(MLFLOW_TRACKING_URI)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(init_client)
        client = future.result(timeout=5)
    print("  ✓ MLflow client initialized")
except FutureTimeoutError:
    print("  ✗ Client initialization timed out after 5 seconds")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed to initialize client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 4: Search experiments
print("Test 4: Search Experiments")
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
    print("  ✗ search_experiments() timed out after 10 seconds")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ search_experiments() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 5: Get runs (if experiments exist)
if experiments:
    print("Test 5: Get Runs")
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
        print("  ✗ get_runs() timed out after 10 seconds")
    except Exception as e:
        print(f"  ✗ get_runs() failed: {e}")
    print()

print("=" * 60)
print("✓ All tests passed! MLflow connection is working.")
print("=" * 60)

