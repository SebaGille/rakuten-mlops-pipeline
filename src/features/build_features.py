import pandas as pd
import re
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import joblib

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
IMAGE_DIR = DATA_RAW / "images" / "image_train"

INPUT_FILE = DATA_INTERIM / "merged_train.csv"
OUTPUT_FILE = DATA_PROCESSED / "train_features.csv"
IMAGE_FEATURES_FILE = DATA_PROCESSED / "image_features.npy"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)      # remove HTML tags
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_image_feature_extractor():
    """Load MobileNetV2 pre-trained model for feature extraction"""
    print("Loading MobileNetV2 model (pre-trained on ImageNet)...")
    # Load model without top classification layer (include_top=False)
    # This gives us a 1280-dimensional feature vector per image
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',  # Global average pooling
        input_shape=(224, 224, 3)
    )
    print("Model loaded successfully.")
    return base_model

def extract_image_features(df, model, image_dir):
    """Extract features from images using pre-trained CNN"""
    print(f"Extracting image features from {len(df)} images...")
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

def build_features(df):
    df["designation_clean"] = df["designation"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)

    # simple numerical features
    df["designation_len"] = df["designation_clean"].apply(len)
    df["description_len"] = df["description_clean"].apply(len)
    df["has_description"] = df["description"].notna().astype(int)

    # minimal selection for modeling
    features = df[[
        "productid", "imageid", "designation_clean", "description_clean",
        "designation_len", "description_len", "has_description", "prdtypecode"
    ]]
    return features

def main():
    print("Loading merged dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded shape: {df.shape}")

    print("Building text features...")
    processed = build_features(df)
    print(f"Processed shape: {processed.shape}")

    # Extract image features
    print("\n=== Extracting Image Features ===")
    model = load_image_feature_extractor()
    image_features = extract_image_features(df, model, IMAGE_DIR)
    print(f"Image features shape: {image_features.shape}")

    # Save outputs
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved text features → {OUTPUT_FILE}")
    
    # Save image features as numpy array
    np.save(IMAGE_FEATURES_FILE, image_features)
    print(f"Saved image features → {IMAGE_FEATURES_FILE}")
    print(f"\n✓ Feature extraction completed successfully!")

if __name__ == "__main__":
    main()
