#!/bin/bash
# quick_test.sh - Rapid health check for all components
# Usage: ./quick_test.sh

set -e

echo "=== Quick Test Script for Rakuten MLOps Pipeline ==="
echo ""

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs) 2>/dev/null || true
fi
export MLFLOW_TRACKING_URI=http://localhost:5000

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Helper function to test and report
test_component() {
    local name=$1
    local command=$2
    local expected=$3
    
    echo -n "Testing $name... "
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
}

# 1. Git status
echo "━━━ Code Versioning ━━━"
test_component "Git repository" "git status"

# 2. DVC status
test_component "DVC configuration" "dvc status"
echo ""

# 3. Docker services
echo "━━━ Docker Services ━━━"
RUNNING_CONTAINERS=$(docker ps --format "{{.Names}}" | wc -l)
echo "Running containers: $RUNNING_CONTAINERS"

test_component "PostgreSQL" "docker ps | grep postgres"
test_component "MLflow container" "docker ps | grep mlflow"
test_component "Rakuten API container" "docker ps | grep rakuten_api"
test_component "Prometheus" "docker ps | grep prometheus"
test_component "Grafana" "docker ps | grep grafana"
echo ""

# 4. MLflow
echo "━━━ MLflow ━━━"
test_component "MLflow HTTP endpoint" "curl -sf http://localhost:5000/health"

# Check experiments
EXP_COUNT=$(curl -sf http://localhost:5000/api/2.0/mlflow/experiments/list 2>/dev/null | jq '.experiments | length' 2>/dev/null || echo "0")
echo "MLflow experiments count: $EXP_COUNT"
echo ""

# 5. API
echo "━━━ Rakuten API ━━━"
test_component "API health endpoint" "curl -sf http://localhost:8000/health"

# Check API status
API_HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null | jq -r '.status' 2>/dev/null || echo "unknown")
echo "API status: $API_HEALTH"

MODEL_LOADED=$(curl -sf http://localhost:8000/health 2>/dev/null | jq -r '.model_loaded' 2>/dev/null || echo "false")
echo "Model loaded: $MODEL_LOADED"

# 6. Prediction test
echo ""
echo "━━━ Prediction Test ━━━"
PRED_RESULT=$(curl -sf -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"designation": "Ordinateur portable", "description": "HP 15 pouces"}' 2>/dev/null | jq -r '.predicted_prdtypecode' 2>/dev/null || echo "ERROR")

if [ "$PRED_RESULT" != "ERROR" ] && [ "$PRED_RESULT" != "null" ]; then
    echo -e "${GREEN}✓ Prediction successful${NC}"
    echo "  Predicted class: $PRED_RESULT"
    ((PASSED++))
else
    echo -e "${RED}✗ Prediction failed${NC}"
    ((FAILED++))
fi

# Check inference log
if [ -f data/monitoring/inference_log.csv ]; then
    LOG_COUNT=$(wc -l < data/monitoring/inference_log.csv)
    echo "Inference log entries: $((LOG_COUNT - 1))"
else
    echo "Inference log: NOT FOUND"
fi
echo ""

# 7. Monitoring Stack
echo "━━━ Monitoring ━━━"
test_component "Prometheus health" "curl -sf http://localhost:9090/-/healthy"
test_component "Grafana health" "curl -sf http://localhost:3000/api/health"

# Check metrics endpoint
METRICS_COUNT=$(curl -sf http://localhost:8000/metrics 2>/dev/null | grep -c "rakuten_" || echo "0")
echo "API metrics exposed: $METRICS_COUNT metrics"
echo ""

# 8. Data and Models
echo "━━━ Data & Models ━━━"
test_component "Training data (X_train.csv)" "[ -f data/raw/X_train.csv ]"
test_component "Test data (X_test.csv)" "[ -f data/raw/X_test.csv ]"
test_component "Processed features" "[ -f data/processed/train_features.csv ]"
test_component "Trained model" "[ -f models/baseline_model.pkl ]"
echo ""

# 9. Drift Detection
echo "━━━ Drift Detection ━━━"
test_component "Evidently report" "[ -f reports/evidently/evidently_report.html ]"
test_component "Drift status JSON" "[ -f reports/evidently/drift_status.json ]"

if [ -f reports/evidently/drift_status.json ]; then
    DRIFT_DETECTED=$(cat reports/evidently/drift_status.json 2>/dev/null | jq -r '.dataset_drift' 2>/dev/null || echo "unknown")
    echo "Dataset drift detected: $DRIFT_DETECTED"
fi
echo ""

# 10. Prefect
echo "━━━ Orchestration (Prefect) ━━━"
test_component "Prefect CLI" "prefect version"

# Check deployments
DEPLOYMENT_COUNT=$(prefect deployment ls 2>/dev/null | grep -c "monitor-and-retrain" || echo "0")
echo "Prefect deployments: $DEPLOYMENT_COUNT"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Tests passed: ${GREEN}$PASSED${NC}"
echo -e "Tests failed: ${RED}$FAILED${NC}"
TOTAL=$((PASSED + FAILED))
echo "Total tests: $TOTAL"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All systems operational!${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Some components failed. Check the log above.${NC}"
    exit 1
fi

