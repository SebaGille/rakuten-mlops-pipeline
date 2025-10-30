import os
from pathlib import Path
import json
import pandas as pd

from evidently import Report
from evidently import metrics

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
INFERENCE_LOG = PROJECT_ROOT / "data" / "monitoring" / "inference_log.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "evidently"
REPORT_HTML = REPORT_DIR / "evidently_report.html"
REPORT_JSON = REPORT_DIR / "evidently_report.json"

def load_reference_df() -> pd.DataFrame:
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(f"Reference file not found: {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)

    required_cols = {"designation_clean", "description_clean", "prdtypecode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in reference: {sorted(missing)}")

    # Construire le signal minimal pour le drift
    text = (df["designation_clean"].fillna("") + " " + df["description_clean"].fillna("")).str.strip()
    
    # Handle NaN values before casting to int
    prdtypecode = pd.to_numeric(df["prdtypecode"], errors='coerce')
    if prdtypecode.isna().any():
        raise ValueError(f"Reference data contains invalid prdtypecode values: {df[prdtypecode.isna()].index.tolist()[:5]}")
    
    ref = pd.DataFrame({
        "text_len": text.str.len(),
        "prdtypecode": prdtypecode.astype(int),
    })
    
    if ref.empty:
        raise ValueError("Reference data is empty after processing")
    
    return ref

def load_current_df() -> pd.DataFrame:
    if not INFERENCE_LOG.exists():
        raise FileNotFoundError(f"Inference log not found: {INFERENCE_LOG}\n"
                                f"Call the API to populate it before running this script.")
    cur = pd.read_csv(INFERENCE_LOG)
    
    # Validate and compute text_len if needed
    if "text_len" not in cur.columns:
        # fallback si ancien log sans text_len
        if "designation" not in cur.columns or "description" not in cur.columns:
            raise KeyError("Missing columns: need either 'text_len' or both 'designation' and 'description'")
        cur["text_len"] = (cur["designation"].fillna("") + " " + cur["description"].fillna("")).str.len()

    # Aligner le nom de la colonne « target » pour TargetDriftPreset
    if "predicted_prdtypecode" in cur.columns:
        cur = cur.rename(columns={"predicted_prdtypecode": "prdtypecode"})
    elif "prdtypecode" not in cur.columns:
        raise KeyError("Missing target column: need either 'predicted_prdtypecode' or 'prdtypecode'")
    
    # Handle NaN values before casting to int
    prdtypecode = pd.to_numeric(cur["prdtypecode"], errors='coerce')
    if prdtypecode.isna().any():
        print(f"⚠️  Warning: {prdtypecode.isna().sum()} invalid prdtypecode values found, filling with -1")
    cur["prdtypecode"] = prdtypecode.fillna(-1).astype(int)
    
    cur = cur[["text_len", "prdtypecode"]].copy()
    
    if cur.empty:
        raise ValueError("Inference log is empty after processing")
    
    return cur

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_reference_df()
    cur = load_current_df()

    report = Report(metrics=[
        metrics.DriftedColumnsCount(),  # Count of drifted columns
        metrics.ValueDrift(column="text_len"),  # Drift detection for text length
        metrics.ValueDrift(column="prdtypecode"),  # Drift detection for target variable
        metrics.RowCount(),  # Basic dataset statistics
        metrics.ColumnCount(),  # Column count comparison
    ])
    snapshot = report.run(reference_data=ref, current_data=cur)

    # Sauvegardes
    snapshot.save_html(str(REPORT_HTML))
    snapshot.save_json(str(REPORT_JSON))

    print(f"✅ Evidently report saved:\n - HTML: {REPORT_HTML}\n - JSON: {REPORT_JSON}")

if __name__ == "__main__":
    main()
