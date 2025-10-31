# Sécurité et Configuration des Variables d'Environnement

## 📋 Vue d'ensemble

Les identifiants sensibles ont été externalisés dans le fichier `.env` pour améliorer la sécurité du projet.

## 🔐 Fichiers de Configuration

### `.env` (NON versionné)
- Contient les **vraies valeurs** des identifiants
- Ignoré par Git (voir `.gitignore`)
- À créer localement sur chaque environnement
- **Ne JAMAIS commiter ce fichier**

### `.env.example` (Versionné)
- Modèle documentant les variables requises
- Contient des valeurs d'exemple/placeholders
- Peut être versionné dans Git
- Sert de documentation pour les nouveaux développeurs

## ⚙️ Variables Externalisées

Les variables suivantes ont été déplacées vers `.env` :

### Grafana
```bash
GF_SECURITY_ADMIN_USER=seba
GF_SECURITY_ADMIN_PASSWORD=sebamlops
```

### PostgreSQL
```bash
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow
```

### AWS S3 (déjà externalisées)
```bash
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***
AWS_DEFAULT_REGION=eu-west-1
S3_BUCKET_NAME=mlops-rakuten-seba
```

## 📊 Impacts de la Migration

### ✅ Avantages

1. **Sécurité Renforcée**
   - Les identifiants ne sont plus visibles dans le code
   - Pas de risque de commit accidentel des credentials
   - Conformité aux bonnes pratiques de sécurité

2. **Gestion Multi-Environnements**
   - Différents `.env` pour dev/staging/production
   - Pas besoin de modifier les docker-compose
   - Facilite les déploiements

3. **Flexibilité**
   - Changement de mot de passe sans toucher au code
   - Configuration différente par développeur
   - Rotation des secrets simplifiée

4. **Auditabilité**
   - Les changements de config ne polluent pas l'historique Git
   - Traçabilité des modifications de structure seulement

### ⚠️ Points d'Attention

1. **Configuration Initiale**
   - **IMPACT** : Chaque développeur doit créer son `.env`
   - **SOLUTION** : Copier `.env.example` vers `.env` et remplir les valeurs
   ```bash
   cp .env.example .env
   # Puis éditer .env avec les vraies valeurs
   ```

2. **Docker Compose**
   - **IMPACT** : Docker Compose doit charger le `.env`
   - **SOLUTION** : Docker Compose charge automatiquement `.env` du répertoire courant
   - Pas de changement nécessaire dans les commandes

3. **CI/CD**
   - **IMPACT** : Les pipelines doivent définir les variables d'environnement
   - **SOLUTION** : Utiliser les secrets du CI/CD (GitHub Secrets, GitLab CI/CD Variables, etc.)

4. **Documentation Équipe**
   - **IMPACT** : L'équipe doit connaître les nouvelles variables
   - **SOLUTION** : Ce fichier + `.env.example` documentent tout

## 🚀 Migration - Comment Utiliser

### Pour les Nouveaux Développeurs

1. Cloner le repository
2. Copier le fichier template :
   ```bash
   cp .env.example .env
   ```
3. Éditer `.env` avec les vraies valeurs (demander à l'équipe)
4. Lancer les services normalement :
   ```bash
   docker-compose -f docker-compose.monitor.yml up
   docker-compose -f docker-compose.mlflow.yml up
   docker-compose -f docker-compose.api.yml up
   ```

### Pour les Développeurs Existants

Si vous aviez déjà lancé les services :
1. Le fichier `.env` a été créé/mis à jour automatiquement
2. Arrêter les conteneurs existants :
   ```bash
   docker-compose -f docker-compose.monitor.yml down
   docker-compose -f docker-compose.mlflow.yml down
   docker-compose -f docker-compose.api.yml down
   ```
3. Recréer les conteneurs :
   ```bash
   docker-compose -f docker-compose.monitor.yml up --build
   docker-compose -f docker-compose.mlflow.yml up --build
   docker-compose -f docker-compose.api.yml up --build
   ```

## 🔍 Vérification

Pour vérifier que les variables sont bien chargées :

```bash
# Vérifier que le .env existe
ls -la .env

# Lancer avec verbose pour voir les variables
docker-compose -f docker-compose.monitor.yml config
```

## 🛡️ Bonnes Pratiques

1. **Ne jamais commiter `.env`** ✅ (déjà dans `.gitignore`)
2. **Toujours commiter `.env.example`** ✅
3. **Utiliser des mots de passe forts** en production
4. **Rotation régulière des secrets** (AWS, PostgreSQL, Grafana)
5. **Utiliser des gestionnaires de secrets** pour la production (AWS Secrets Manager, Vault, etc.)

## 📝 Checklist de Sécurité

- [x] `.env` est dans `.gitignore`
- [x] `.env.example` est versionné
- [x] Pas d'identifiants en dur dans `docker-compose*.yml`
- [x] Documentation des variables créée
- [ ] Rotation des mots de passe pour la production
- [ ] Configuration des secrets CI/CD (si applicable)

## 🆘 Dépannage

### Erreur "variable not set"
```bash
# Vérifier que le .env existe dans le bon répertoire
pwd
ls .env
```

### Les variables ne sont pas chargées
```bash
# S'assurer d'être dans le bon répertoire
cd /Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten

# Recréer les conteneurs
docker-compose -f docker-compose.monitor.yml down
docker-compose -f docker-compose.monitor.yml up
```

### PostgreSQL refuse la connexion
Vérifier que `POSTGRES_USER`, `POSTGRES_PASSWORD` et `POSTGRES_DB` sont cohérents dans `.env` et que les anciennes données ne créent pas de conflit :
```bash
docker-compose -f docker-compose.mlflow.yml down -v  # -v pour supprimer les volumes
docker-compose -f docker-compose.mlflow.yml up
```

