import os
import pandas as pd
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

# --- Input files ---
X_TRAIN_PATH = DATA_RAW / "X_train.csv"
Y_TRAIN_PATH = DATA_RAW / "Y_train.csv"
X_TEST_PATH = DATA_RAW / "X_test.csv"
IMAGE_DIR = DATA_RAW / "images" / "image_train"

# --- Output file ---
MERGED_TRAIN_PATH = DATA_INTERIM / "merged_train.csv"

def load_data():
    print("Loading CSV files...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    Y_train = pd.read_csv(Y_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    # Keep Y_train as-is (its 'Unnamed: 0' is only an index)
    # Just ensure correct column names for visibility
    if "prdtypecode" not in Y_train.columns:
        raise ValueError("Expected column 'prdtypecode' not found in Y_train.csv")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}")
    print("Columns loaded successfully.")
    return X_train, Y_train, X_test

def merge_train_data(X_train, Y_train):
    print("Merging training data by row order...")
    if len(X_train) != len(Y_train):
        raise ValueError("Row counts differ between X_train and Y_train!")
    # Reset indices to align properly
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    # Concatenate column-wise (axis=1)
    merged = pd.concat([X_train, Y_train["prdtypecode"]], axis=1)
    print(f"Merged shape: {merged.shape}")
    return merged

def verify_images_exist(df, image_dir):
    print("Verifying image existence...")
    missing = []
    for _, row in df.iterrows():
        img_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        if not (image_dir / img_name).exists():
            missing.append(img_name)
    if missing:
        print(f"Missing {len(missing)} images!")
    else:
        print("All referenced images are present.")
    return missing

def save_merged_data(df):
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_TRAIN_PATH, index=False)
    print(f"Saved merged training data → {MERGED_TRAIN_PATH}")

def main():
    X_train, Y_train, X_test = load_data()
    merged = merge_train_data(X_train, Y_train)
    verify_images_exist(merged, IMAGE_DIR)
    save_merged_data(merged)
    print("Dataset ingestion completed successfully.")

if __name__ == "__main__":
    main()

