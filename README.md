# Rakuten Product Classification — MLOps (DataScientest)

Projet de pipeline MLOps complet pour la classification des produits Rakuten (texte + image) avec **traçabilité**, **versioning** et **suivi d'expériences**.

## 🔧 Stack (progressive)
- **Python 3.11** (venv) — ⚠️ Requis pour Prefect
- **DVC** + **Dagshub** (versioning data/modèles)
- **MLflow** + **PostgreSQL** + **FastAPI** (Docker) — suivi d'expériences & serving
- **Artifacts MLflow**: S3 (via variables d'environnement)
- **Prefect** (orchestration) — installé
- **À venir** : CI/CD GitHub Actions, Prometheus/Grafana, Evidently

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

### 2) Variables d'environnement (S3 / MLflow)

⚠️ **IMPORTANT** : Créer un fichier `.env` à la racine du projet (non commité, déjà dans `.gitignore`) :

```bash
# AWS S3 Configuration for MLflow
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=eu-west-1
S3_BUCKET_NAME=your-bucket-name
```

Docker Compose charge automatiquement ce `.env` pour les containers.

### 3) Lancer les services Docker (MLflow + PostgreSQL + API)

```bash
# Démarre tous les services (mlflow, postgres, rakuten_api)
docker-compose -f docker-compose.api.yml up -d

# Vérifier que tout tourne
docker ps

# UI MLflow: http://localhost:5000
# API Rakuten: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Arrêter tous les containers** :
```bash
docker-compose -f docker-compose.api.yml down
```

**Redémarrer après modification** :
```bash
docker-compose -f docker-compose.api.yml down
docker-compose -f docker-compose.api.yml up --build -d
```

### 4) Exécuter le Pipeline (Prefect)

⚠️ **CRITIQUE** : Pour que les artifacts MLflow soient sauvegardés sur S3 (et non localement), **vous DEVEZ charger les variables d'environnement** avant d'exécuter le pipeline :

```bash
# Activer l'environnement virtuel
source .venv/bin/activate

# ⚠️ IMPORTANT: Charger les variables AWS depuis .env
export $(cat .env | grep -v '^#' | xargs)

# Définir l'URI de MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000

# Lancer le pipeline complet (ingest → features → train → predict)
python flows/pipeline_flow.py
```

**Commande complète en une ligne** :
```bash
source .venv/bin/activate && export $(cat .env | grep -v '^#' | xargs) && export MLFLOW_TRACKING_URI=http://localhost:5000 && python flows/pipeline_flow.py
```

### 5) Pipeline DVC (alternative reproductible)

```bash
# Charger les variables d'environnement
export $(cat .env | grep -v '^#' | xargs)

# Exécute ingest → features → train → predict
dvc repro

# Pousse les artefacts (data/modèles) vers le remote DVC (Dagshub)
dvc push
```

### 6) Scripts unitaires

```bash
# Charger .env d'abord
export $(cat .env | grep -v '^#' | xargs)

# Build features
python src/features/build_features.py

# Entraînement (log MLflow + artefacts S3)
python src/models/train_model.py

# Prédictions sur X_test
python src/models/predict_model.py
```

## 🧭 Bonnes pratiques & traçabilité

* **MLflow** : chaque run logue paramètres, métriques et artefacts (modèle, vectorizer, metrics.json).
  Des tags lient le run au **commit Git** (`git_commit`, `git_branch`) et la run joint `dvc.yaml`/`dvc.lock`.
* **DVC** : gère les outputs de pipeline (`data/interim`, `data/processed`, `models/*`) et synchronise vers Dagshub.
* **Branches** : travail sur `feat/seba/bootstrap` puis PR vers `main`.

## 🆘 Dépannage rapide

* **Erreur `OSError: [Errno 30] Read-only file system: '/mlflow'`** : 
  - ⚠️ **Vous avez oublié de charger les variables d'environnement !**
  - Solution : `export $(cat .env | grep -v '^#' | xargs)` avant d'exécuter le pipeline
  - Les artifacts MLflow doivent aller sur S3, pas en local

* **S3 auth fail / Access Denied** : 
  - Vérifier que `.env` est bien chargé dans le shell : `echo $AWS_ACCESS_KEY_ID`
  - Vérifier les droits IAM sur le bucket S3
  - Vérifier que `S3_BUCKET_NAME` est bien défini

* **MLflow experiment with local artifact path** :
  - L'expérience a été créée avant que S3 soit configuré
  - Solution : recréer l'expérience ou redémarrer les containers Docker

* **DVC "tracked by SCM"** : 
  - Retirer du suivi Git (`git rm -r --cached <fichier>`) avant de déclarer en output DVC

* **Images manquantes** : 
  - Le chemin attendu est `data/raw/images/image_train/` avec le motif `image_<imageid>_product_<productid>.jpg`

* **Docker containers not starting** :
  - Vérifier que le fichier `.env` existe à la racine
  - Vérifier les logs : `docker logs sep25_cmlops_rakuten-mlflow-1`

## 📌 Licence

Voir `LICENSE`.
MD

```

