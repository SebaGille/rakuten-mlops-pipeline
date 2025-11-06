#!/bin/bash
# Test script to verify MLflow connection from terminal
# This helps debug connection issues before fixing Streamlit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "MLflow Connection Test Script"
echo "=========================================="
echo ""

# Get ALB URL from environment or use default
ALB_URL="${AWS_ALB_URL:-https://rakuten-mlops-alb-123456789.eu-west-1.elb.amazonaws.com}"
MLFLOW_BASE_URL="${ALB_URL}/mlflow"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-${MLFLOW_BASE_URL}}"

echo "Configuration:"
echo "  ALB URL: ${ALB_URL}"
echo "  MLflow Base URL: ${MLFLOW_BASE_URL}"
echo "  MLflow Tracking URI: ${MLFLOW_TRACKING_URI}"
echo ""

# Extract hostname for DNS test
HOSTNAME=$(echo "${ALB_URL}" | sed -e 's|^[^/]*//||' -e 's|/.*$||')
echo "Testing DNS resolution for: ${HOSTNAME}"
if nslookup "${HOSTNAME}" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ DNS resolution successful${NC}"
    IP=$(nslookup "${HOSTNAME}" | grep -A 1 "Name:" | tail -1 | awk '{print $2}')
    echo "  Resolved to: ${IP}"
else
    echo -e "${RED}✗ DNS resolution failed${NC}"
    exit 1
fi
echo ""

# Test 1: Basic connectivity to ALB
echo "Test 1: Basic HTTP connectivity to ALB"
echo "  Testing: ${ALB_URL}"
if curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${ALB_URL}" | grep -q "200\|301\|302\|404"; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${ALB_URL}")
    echo -e "${GREEN}✓ ALB is reachable (HTTP ${HTTP_CODE})${NC}"
else
    echo -e "${RED}✗ ALB is not reachable${NC}"
    echo "  Trying with verbose output..."
    curl -v --max-time 10 "${ALB_URL}" || true
    exit 1
fi
echo ""

# Test 2: MLflow base URL
echo "Test 2: MLflow base URL connectivity"
echo "  Testing: ${MLFLOW_BASE_URL}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${MLFLOW_BASE_URL}" || echo "000")
if [ "${HTTP_CODE}" = "200" ] || [ "${HTTP_CODE}" = "301" ] || [ "${HTTP_CODE}" = "302" ] || [ "${HTTP_CODE}" = "404" ]; then
    echo -e "${GREEN}✓ MLflow base URL is reachable (HTTP ${HTTP_CODE})${NC}"
else
    echo -e "${YELLOW}⚠ MLflow base URL returned HTTP ${HTTP_CODE}${NC}"
    echo "  This might be OK if ALB routing is configured"
fi
echo ""

# Test 3: MLflow health endpoint
echo "Test 3: MLflow health endpoint"
HEALTH_URL="${MLFLOW_BASE_URL}/health"
echo "  Testing: ${HEALTH_URL}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${HEALTH_URL}" || echo "000")
if [ "${HTTP_CODE}" = "200" ]; then
    echo -e "${GREEN}✓ Health endpoint is accessible (HTTP ${HTTP_CODE})${NC}"
    RESPONSE=$(curl -s --max-time 10 "${HEALTH_URL}")
    echo "  Response: ${RESPONSE}"
else
    echo -e "${YELLOW}⚠ Health endpoint returned HTTP ${HTTP_CODE}${NC}"
fi
echo ""

# Test 4: MLflow API health endpoint
echo "Test 4: MLflow API health endpoint"
API_HEALTH_URL="${MLFLOW_BASE_URL}/api/2.0/mlflow/health"
echo "  Testing: ${API_HEALTH_URL}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${API_HEALTH_URL}" || echo "000")
if [ "${HTTP_CODE}" = "200" ]; then
    echo -e "${GREEN}✓ API health endpoint is accessible (HTTP ${HTTP_CODE})${NC}"
    RESPONSE=$(curl -s --max-time 10 "${API_HEALTH_URL}")
    echo "  Response: ${RESPONSE}"
else
    echo -e "${YELLOW}⚠ API health endpoint returned HTTP ${HTTP_CODE}${NC}"
fi
echo ""

# Test 5: MLflow experiments API
echo "Test 5: MLflow experiments API"
EXPERIMENTS_URL="${MLFLOW_BASE_URL}/api/2.0/mlflow/experiments/search"
echo "  Testing: ${EXPERIMENTS_URL}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X POST "${EXPERIMENTS_URL}" \
    -H "Content-Type: application/json" \
    -d '{"max_results": 1}' || echo "000")
if [ "${HTTP_CODE}" = "200" ]; then
    echo -e "${GREEN}✓ Experiments API is accessible (HTTP ${HTTP_CODE})${NC}"
    RESPONSE=$(curl -s --max-time 10 -X POST "${EXPERIMENTS_URL}" \
        -H "Content-Type: application/json" \
        -d '{"max_results": 1}')
    echo "  Response preview: $(echo "${RESPONSE}" | head -c 200)..."
else
    echo -e "${RED}✗ Experiments API returned HTTP ${HTTP_CODE}${NC}"
    echo "  Full response:"
    curl -s --max-time 10 -X POST "${EXPERIMENTS_URL}" \
        -H "Content-Type: application/json" \
        -d '{"max_results": 1}' || true
fi
echo ""

# Test 6: Python MLflow client test
echo "Test 6: Python MLflow client test"
echo "  Testing MLflow client with tracking URI: ${MLFLOW_TRACKING_URI}"
python3 << EOF
import sys
import mlflow
from mlflow.tracking import MlflowClient
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time

tracking_uri = "${MLFLOW_TRACKING_URI}"
print(f"  Setting tracking URI: {tracking_uri}")

try:
    mlflow.set_tracking_uri(tracking_uri)
    print("  ✓ Tracking URI set successfully")
except Exception as e:
    print(f"  ✗ Failed to set tracking URI: {e}")
    sys.exit(1)

print("  Initializing MLflow client...")
try:
    def init_client():
        return MlflowClient(tracking_uri)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(init_client)
        client = future.result(timeout=5)
    print("  ✓ MLflow client initialized successfully")
except FutureTimeoutError:
    print("  ✗ MLflow client initialization timed out after 5 seconds")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed to initialize MLflow client: {e}")
    sys.exit(1)

print("  Testing search_experiments()...")
try:
    def search_experiments():
        return client.search_experiments(max_results=1)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(search_experiments)
        experiments = future.result(timeout=10)
    print(f"  ✓ search_experiments() successful (found {len(experiments)} experiments)")
    if experiments:
        print(f"    First experiment: {experiments[0].name}")
except FutureTimeoutError:
    print("  ✗ search_experiments() timed out after 10 seconds")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ search_experiments() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  ✓ All Python MLflow client tests passed!")
EOF

PYTHON_EXIT=$?
if [ $PYTHON_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Python MLflow client test passed${NC}"
else
    echo -e "${RED}✗ Python MLflow client test failed${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "If all tests passed, the MLflow connection should work from Streamlit."
echo "If any test failed, check:"
echo "  1. ALB URL is correct"
echo "  2. ALB is running and healthy"
echo "  3. MLflow service is running in ECS"
echo "  4. ALB path-based routing is configured correctly"
echo "  5. Security groups allow traffic"
echo "  6. DNS resolution is working"
echo ""

