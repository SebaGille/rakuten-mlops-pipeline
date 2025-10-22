from prefect import flow, task
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Importer les mains de tes scripts
# (les scripts ont un bloc if __name__ == "__main__": donc l'import est sûr)
from src.data.make_dataset import main as make_dataset_main
from src.features.build_features import main as build_features_main
from src.models.train_model import main as train_model_main
from src.models.predict_model import main as predict_model_main

@task(retries=1)
def ingest():
    make_dataset_main()

@task(retries=1)
def build_features():
    build_features_main()

@task(retries=1)
def train():
    train_model_main()

@task(retries=1)
def predict():
    predict_model_main()

@flow(name="rakuten_pipeline_local")
def rakuten_pipeline(run_predict: bool = True):
    ingest()
    build_features()
    train()
    if run_predict:
        predict()

if __name__ == "__main__":
    rakuten_pipeline()