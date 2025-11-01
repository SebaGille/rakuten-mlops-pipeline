import pandas as pd
import joblib
import numpy as np
import re
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Handle imports for both package and script execution
try:
    from src.models.multimodal import MultiModalClassifier
except ModuleNotFoundError:
    from multimodal import MultiModalClassifier

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_DIR = DATA_RAW / "images" / "image_test"  # Test images directory

MODEL_FILE = MODELS_DIR / "baseline_model.pkl"
INPUT_FILE = DATA_RAW / "X_test.csv"
OUTPUT_FILE = DATA_PROCESSED / "predictions.csv"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_image_feature_extractor():
    """Load MobileNetV2 pre-trained model for feature extraction"""
    print("Loading MobileNetV2 model (pre-trained on ImageNet)...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    print("Model loaded successfully.")
    return base_model

def extract_image_features(df, model, image_dir):
    """Extract features from test images using pre-trained CNN"""
    print(f"Extracting image features from {len(df)} test images...")
    features_list = []
    missing_count = 0
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing image {idx}/{len(df)}...")
        
        # Construct image filename
        img_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        img_path = image_dir / img_name
        
        if not img_path.exists():
            # If image is missing, use zero vector
            features_list.append(np.zeros(1280))
            missing_count += 1
            continue
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = model.predict(img_array, verbose=0)
            features_list.append(features.flatten())
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            features_list.append(np.zeros(1280))
            missing_count += 1
    
    print(f"Image features extracted. Missing/error images: {missing_count}/{len(df)}")
    return np.array(features_list)

def main():
    print("Loading multimodal model...")
    saved = joblib.load(MODEL_FILE)
    multimodal_model = saved["multimodal_model"]

    print("Loading test data...")
    X_test = pd.read_csv(INPUT_FILE)
    print(f"Loaded shape: {X_test.shape}")

    # Preprocess text
    print("Preprocessing text...")
    X_test["designation_clean"] = X_test["designation"].apply(clean_text)
    X_test["description_clean"] = X_test["description"].apply(clean_text)
    X_test["text"] = X_test["designation_clean"].fillna("") + " " + X_test["description_clean"].fillna("")

    # Extract image features for test set
    print("\n=== Extracting Image Features from Test Set ===")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Test image directory not found: {IMAGE_DIR}")
    
    cnn_model = load_image_feature_extractor()
    test_image_features = extract_image_features(X_test, cnn_model, IMAGE_DIR)
    print(f"Test image features shape: {test_image_features.shape}")

    # Make predictions with multimodal model
    print("\nMaking predictions with multimodal model (text + image)...")
    preds = multimodal_model.predict(X_test["text"], image_features_test=test_image_features)

    X_test["predicted_prdtypecode"] = preds

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_test[["productid", "predicted_prdtypecode"]].to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions → {OUTPUT_FILE}")
    print(f"\n✓ Predictions completed successfully!")

if __name__ == "__main__":
    main()
