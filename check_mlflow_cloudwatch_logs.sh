#!/bin/bash
# Script to check CloudWatch logs for MLflow ECS service
# This helps debug what requests are being received by the MLflow server

set -e

REGION="${AWS_REGION:-eu-west-1}"
LOG_GROUP="/ecs/rakuten-mlflow"
SERVICE_NAME="${ECS_MLFLOW_SERVICE_NAME:-rakuten-mlflow-service}"
CLUSTER_NAME="${AWS_ECS_CLUSTER:-rakuten-mlops-cluster}"

echo "=========================================="
echo "MLflow CloudWatch Logs Checker"
echo "=========================================="
echo "Region: $REGION"
echo "Log Group: $LOG_GROUP"
echo "Service: $SERVICE_NAME"
echo "Cluster: $CLUSTER_NAME"
echo ""

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed or not in PATH"
    echo "Install it with: pip install awscli"
    exit 1
fi

# Check if log group exists
if ! aws logs describe-log-groups --region "$REGION" --log-group-name-prefix "$LOG_GROUP" --query "logGroups[?logGroupName=='$LOG_GROUP']" --output text | grep -q "$LOG_GROUP"; then
    echo "Warning: Log group $LOG_GROUP not found"
    echo "Available log groups:"
    aws logs describe-log-groups --region "$REGION" --query "logGroups[*].logGroupName" --output table
    exit 1
fi

echo "Fetching recent logs from $LOG_GROUP..."
echo ""

# Get the most recent log stream
LOG_STREAM=$(aws logs describe-log-streams \
    --region "$REGION" \
    --log-group-name "$LOG_GROUP" \
    --order-by LastEventTime \
    --descending \
    --max-items 1 \
    --query "logStreams[0].logStreamName" \
    --output text)

if [ "$LOG_STREAM" == "None" ] || [ -z "$LOG_STREAM" ]; then
    echo "No log streams found in $LOG_GROUP"
    exit 1
fi

echo "Most recent log stream: $LOG_STREAM"
echo ""

# Get logs from the last 10 minutes
# Try macOS date syntax first, fallback to Linux/GNU date
if date -v-10M +%s &>/dev/null; then
    # macOS date
    START_TIME=$(($(date -u -v-10M +%s) * 1000))
    END_TIME=$(($(date -u +%s) * 1000))
else
    # Linux/GNU date
    START_TIME=$(($(date -u -d '10 minutes ago' +%s) * 1000))
    END_TIME=$(($(date -u +%s) * 1000))
fi

echo "Fetching logs from last 10 minutes..."
echo ""

# Fetch and display logs
aws logs get-log-events \
    --region "$REGION" \
    --log-group-name "$LOG_GROUP" \
    --log-stream-name "$LOG_STREAM" \
    --start-time "$START_TIME" \
    --query "events[*].[timestamp,message]" \
    --output table | head -100

echo ""
echo "=========================================="
echo "To see more logs, run:"
echo "aws logs tail $LOG_GROUP --follow --region $REGION"
echo "=========================================="

