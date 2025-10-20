import pandas as pd
import re
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_INTERIM / "merged_train.csv"
OUTPUT_FILE = DATA_PROCESSED / "train_features.csv"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)      # remove HTML tags
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

    print("Building features...")
    processed = build_features(df)
    print(f"Processed shape: {processed.shape}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved processed features → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
