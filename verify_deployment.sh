#!/bin/bash
# ============================================
# Streamlit Cloud Deployment Verification
# ============================================

echo "✓ Checking deployment requirements..."
echo ""

# 1. Check entry point file
echo "1. Entry point file:"
if [ -f "streamlit_app/Home.py" ]; then
    echo "   ✓ streamlit_app/Home.py exists"
else
    echo "   ✗ streamlit_app/Home.py NOT FOUND"
fi
echo ""

# 2. Check requirements file
echo "2. Requirements file:"
if [ -f "requirements-deploy.txt" ]; then
    echo "   ✓ requirements-deploy.txt exists"
    echo "   Checking for key dependencies:"
    grep -q "streamlit" requirements-deploy.txt && echo "     ✓ streamlit" || echo "     ✗ streamlit MISSING"
    grep -q "boto3" requirements-deploy.txt && echo "     ✓ boto3" || echo "     ✗ boto3 MISSING"
    grep -q "pandas" requirements-deploy.txt && echo "     ✓ pandas" || echo "     ✗ pandas MISSING"
    grep -q "Pillow" requirements-deploy.txt && echo "     ✓ Pillow" || echo "     ✗ Pillow MISSING"
    grep -q "plotly" requirements-deploy.txt && echo "     ✓ plotly" || echo "     ✗ plotly MISSING"
else
    echo "   ✗ requirements-deploy.txt NOT FOUND"
fi
echo ""

# 3. Check Streamlit config
echo "3. Streamlit configuration:"
if [ -f "streamlit_app/.streamlit/config.toml" ]; then
    echo "   ✓ streamlit_app/.streamlit/config.toml exists"
else
    echo "   ⚠ streamlit_app/.streamlit/config.toml not found (optional)"
fi
echo ""

# 4. Check Dataset page
echo "4. Dataset page:"
if [ -f "streamlit_app/pages/3_Dataset.py" ]; then
    echo "   ✓ streamlit_app/pages/3_Dataset.py exists"
    grep -q "load_image" streamlit_app/pages/3_Dataset.py && echo "     ✓ Uses S3 image loading" || echo "     ⚠ May not use S3 image loading"
else
    echo "   ✗ streamlit_app/pages/3_Dataset.py NOT FOUND"
fi
echo ""

# 5. Check TrainingManager
echo "5. TrainingManager (S3 support):"
if [ -f "streamlit_app/utils/training_manager.py" ]; then
    echo "   ✓ streamlit_app/utils/training_manager.py exists"
    grep -q "boto3" streamlit_app/utils/training_manager.py && echo "     ✓ Has boto3 import" || echo "     ✗ Missing boto3 import"
    grep -q "_load_from_s3" streamlit_app/utils/training_manager.py && echo "     ✓ Has S3 loading methods" || echo "     ✗ Missing S3 loading methods"
else
    echo "   ✗ streamlit_app/utils/training_manager.py NOT FOUND"
fi
echo ""

# 6. Verify S3 data is uploaded
echo "6. S3 Data Verification:"
BUCKET_NAME="mlops-rakuten-seba"
echo "   Checking S3 bucket: $BUCKET_NAME"
if aws s3 ls s3://$BUCKET_NAME/data/interim/merged_train.csv > /dev/null 2>&1; then
    echo "     ✓ merged_train.csv exists in S3"
else
    echo "     ✗ merged_train.csv NOT FOUND in S3"
fi
if aws s3 ls s3://$BUCKET_NAME/data/processed/train_features.csv > /dev/null 2>&1; then
    echo "     ✓ train_features.csv exists in S3"
else
    echo "     ✗ train_features.csv NOT FOUND in S3"
fi
IMAGE_COUNT=$(aws s3 ls s3://$BUCKET_NAME/data/raw/images/image_train/ --recursive 2>/dev/null | wc -l | tr -d ' ')
if [ "$IMAGE_COUNT" -gt 0 ]; then
    echo "     ✓ $IMAGE_COUNT images found in S3"
else
    echo "     ✗ No images found in S3"
fi
echo ""

echo "============================================"
echo "Verification complete!"
echo "============================================"
echo ""
echo "Next steps for Streamlit Cloud:"
echo "1. Ensure secrets are configured in Streamlit Cloud:"
echo "   - S3_DATA_BUCKET=mlops-rakuten-seba"
echo "   - AWS_ACCESS_KEY_ID"
echo "   - AWS_SECRET_ACCESS_KEY"
echo "   - AWS_DEFAULT_REGION=eu-west-1"
echo ""
echo "2. Set main file path: streamlit_app/Home.py"
echo ""
echo "3. Set requirements file: requirements-deploy.txt"
echo ""

