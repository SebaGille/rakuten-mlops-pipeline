# Rakuten Product Classification — MLOps (DataScientest)

Projet de pipeline MLOps complet pour la classification des produits Rakuten (texte + image) avec **traçabilité**, **versioning** et **suivi d'expériences**.

## 🔧 Stack (progressive)
- **Python 3.11** (venv) — ⚠️ Requis pour Prefect
- **DVC** + **Dagshub** (versioning data/modèles)
- **MLflow** + **PostgreSQL** (Docker) — suivi d'expériences
- **Artifacts MLflow**: S3 (via variables d'environnement)
- **Prefect** (orchestration) — installé
- **À venir** : FastAPI (serving), CI/CD GitHub Actions, Prometheus/Grafana, Evidently

## 📦 Données
```
data/
├─ raw/
│  ├─ X_train.csv
│  ├─ Y_train.csv
│  ├─ X_test.csv
│  └─ images/image_train/   (fichiers: image_<imageid>*product*<productid>.jpg)
├─ interim/
│  └─ merged_train.csv
└─ processed/
├─ train_features.csv
└─ predictions.csv

```

## 🗂️ Arborescence du repo (principal)
```
src/
├─ data/make_dataset.py            # ingestion + contrôles
├─ features/build_features.py      # prétraitement & features texte
└─ models/
├─ train_model.py               # entraînement + logs MLflow
└─ predict_model.py             # inférence sur X_test
docker-compose.mlflow.yml          # MLflow + Postgres (Docker)
Dockerfile.mlflow                  # image MLflow custom (psycopg2)
dvc.yaml                           # pipeline DVC (ingest→features→train→predict)

````

## 🚀 Démarrage rapide

### 1) Environnement Python
```bash
# Utiliser Python 3.11 (requis pour Prefect)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Variables d’environnement (S3 / MLflow)

Créer un fichier `.env` (non commité) :

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-1
# Optionnel si S3 compatible (MinIO, etc.)
# MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Pour pointer localement (sinon défini dans le script)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=rakuten-baseline
```

Charger dans la session :

```bash
set -a; source .env; set +a
```

### 3) Lancer MLflow (Docker)

```bash
docker compose -f docker-compose.mlflow.yml up -d
# UI: http://127.0.0.1:5000
```

### 4) Pipeline DVC (reproductible)

```bash
# exécute ingest → features → train → predict
dvc repro
# pousse les artefacts (data/modèles) vers le remote DVC (Dagshub)
dvc push
```

### 5) Scripts unitaires

```bash
# build features
python src/features/build_features.py
# entraînement (log MLflow + artefacts S3)
python src/models/train_model.py
# prédictions sur X_test
python src/models/predict_model.py
```

## 🧭 Bonnes pratiques & traçabilité

* **MLflow** : chaque run logue paramètres, métriques et artefacts (modèle, vectorizer, metrics.json).
  Des tags lient le run au **commit Git** (`git_commit`, `git_branch`) et la run joint `dvc.yaml`/`dvc.lock`.
* **DVC** : gère les outputs de pipeline (`data/interim`, `data/processed`, `models/*`) et synchronise vers Dagshub.
* **Branches** : travail sur `feat/seba/bootstrap` puis PR vers `main`.

## 🆘 Dépannage rapide

* **S3 auth fail** : vérifier `.env` chargé dans le shell (`set -a; source .env; set +a`) et droits IAM.
* **MLflow artifacts en erreur** : vérifier que le compose utilise `/home/mlflow/artifacts` et que le dossier local `./mlruns` existe.
* **DVC “tracked by SCM”** : retirer du suivi Git (`git rm -r --cached <fichier>`) avant de déclarer en output DVC.
* **Images manquantes** : le chemin attendu est `data/raw/images/image_train/` avec le motif `image_<imageid>_product_<productid>.jpg`.

## 📌 Licence

Voir `LICENSE`.
MD

```

