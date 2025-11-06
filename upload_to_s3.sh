#!/bin/bash
# ============================================
# S3 Dataset Upload Script
# ============================================
# This script uploads the dataset files to S3 for Streamlit Cloud deployment

BUCKET_NAME="mlops-rakuten-seba"
REGION="eu-west-1"

echo "============================================"
echo "Uploading dataset to S3: $BUCKET_NAME"
echo "============================================"

# Step 1: Upload merged_train.csv (52MB)
echo ""
echo "Step 2: Uploading merged_train.csv (52MB)..."
aws s3 cp data/interim/merged_train.csv s3://$BUCKET_NAME/data/interim/merged_train.csv
if [ $? -eq 0 ]; then
    echo "✓ merged_train.csv uploaded successfully"
else
    echo "✗ Failed to upload merged_train.csv"
    exit 1
fi

# Step 3: Upload train_features.csv (49MB) - optional fallback
echo ""
echo "Step 3: Uploading train_features.csv (49MB)..."
aws s3 cp data/processed/train_features.csv s3://$BUCKET_NAME/data/processed/train_features.csv
if [ $? -eq 0 ]; then
    echo "✓ train_features.csv uploaded successfully"
else
    echo "✗ Failed to upload train_features.csv"
    exit 1
fi

# Step 4: Upload images (2.3GB, ~85k images) - this will take a while
echo ""
echo "Step 4: Uploading images (2.3GB, ~85k images)..."
echo "This may take 10-30 minutes depending on your connection..."
aws s3 sync data/raw/images/image_train/ s3://$BUCKET_NAME/data/raw/images/image_train/ --no-progress
if [ $? -eq 0 ]; then
    echo "✓ Images uploaded successfully"
else
    echo "✗ Failed to upload images"
    exit 1
fi

# Step 4: Verify uploads
echo ""
echo "Step 4: Verifying uploads..."
echo ""
echo "Checking merged_train.csv:"
aws s3 ls s3://$BUCKET_NAME/data/interim/merged_train.csv
echo ""
echo "Checking train_features.csv:"
aws s3 ls s3://$BUCKET_NAME/data/processed/train_features.csv
echo ""
echo "Counting uploaded images:"
IMAGE_COUNT=$(aws s3 ls s3://$BUCKET_NAME/data/raw/images/image_train/ --recursive | wc -l)
echo "Uploaded $IMAGE_COUNT images"

echo ""
echo "============================================"
echo "✓ Upload complete!"
echo "============================================"
echo "Bucket name: $BUCKET_NAME"
echo "Region: $REGION"
echo ""
echo "Next steps:"
echo "1. Configure Streamlit Cloud secrets with:"
echo "   - S3_DATA_BUCKET=$BUCKET_NAME"
echo "   - AWS_ACCESS_KEY_ID=your-key"
echo "   - AWS_SECRET_ACCESS_KEY=your-secret"
echo "   - AWS_DEFAULT_REGION=$REGION"
echo ""

