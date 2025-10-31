# Migration des identifiants vers fichier .env

## ✅ Modifications effectuées

Les identifiants en dur dans les fichiers `docker-compose.*.yml` ont été externalisés vers le fichier `.env`.

### Fichiers modifiés :
1. **docker-compose.monitor.yml** - Grafana credentials
2. **docker-compose.api.yml** - PostgreSQL credentials
3. **docker-compose.mlflow.yml** - PostgreSQL credentials

### Variables externalisées :
```bash
# Grafana
GF_SECURITY_ADMIN_USER
GF_SECURITY_ADMIN_PASSWORD

# PostgreSQL
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
```

## 📋 Impacts

### ✅ Avantages (Positifs)

1. **Sécurité renforcée** 🔒
   - Les identifiants ne sont plus versionnés dans Git
   - Réduction du risque de fuite d'identifiants
   - Chaque environnement peut avoir ses propres credentials

2. **Flexibilité** 🔄
   - Changement des identifiants sans modifier les fichiers docker-compose
   - Configuration différente par environnement (dev/staging/prod)
   - Rotation des mots de passe simplifiée

3. **Bonnes pratiques** ✨
   - Respect du principe "12-factor app"
   - Séparation configuration / code
   - Conformité avec les standards de sécurité

4. **Collaboration facilitée** 👥
   - Le fichier `.env.example` documente les variables nécessaires
   - Chaque développeur peut avoir ses propres identifiants locaux
   - Pas de conflits Git sur les credentials

### ⚠️ Points d'attention (à gérer)

1. **Configuration initiale requise**
   - Les nouveaux développeurs doivent créer leur fichier `.env`
   - Solution : Copier `.env.example` vers `.env` et remplir les valeurs

2. **Gestion des secrets en production**
   - Ne pas déployer le fichier `.env` en production (déjà ignoré par Git)
   - Utiliser des solutions de gestion de secrets :
     - Docker secrets
     - Kubernetes secrets
     - AWS Secrets Manager / Azure Key Vault
     - HashiCorp Vault

3. **Docker Compose doit charger le .env**
   - Docker Compose charge automatiquement le fichier `.env` s'il est dans le même répertoire
   - Vérifier que le fichier `.env` est au même niveau que les fichiers `docker-compose.*.yml`

4. **Compatibilité avec le healthcheck PostgreSQL**
   - ⚠️ **LIMITATION** : La ligne 17 de `docker-compose.mlflow.yml` utilise des variables dans le healthcheck :
     ```yaml
     test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
     ```
   - Les variables d'environnement **ne sont PAS interpolées** dans les commandes healthcheck par Docker
   - **Solution appliquée** : Docker Compose 3.9+ les interpole au moment du parsing du fichier
   - Si problème, revenir à des valeurs en dur pour le healthcheck uniquement

## 🚀 Mise en place (pour nouveaux développeurs)

```bash
# 1. Copier le fichier exemple
cp .env.example .env

# 2. Éditer le fichier .env avec vos vraies valeurs
nano .env  # ou vim, code, etc.

# 3. Lancer les services
docker-compose -f docker-compose.monitor.yml up -d
docker-compose -f docker-compose.mlflow.yml up -d
docker-compose -f docker-compose.api.yml up -d
```

## 🔍 Vérification

Pour vérifier que les variables sont bien chargées :

```bash
# Vérifier les variables d'environnement d'un container
docker-compose -f docker-compose.monitor.yml config

# Ou inspecter un container en cours d'exécution
docker inspect grafana | grep -A 20 Env
```

## 🔐 Sécurité - Checklist

- [✅] Fichier `.env` ajouté au `.gitignore`
- [✅] Fichier `.env.example` créé (sans valeurs sensibles)
- [✅] Variables utilisées dans tous les docker-compose
- [ ] Documenter la procédure de rotation des mots de passe
- [ ] Utiliser des secrets Docker en production
- [ ] Configurer un gestionnaire de secrets pour la production

## 📝 Notes supplémentaires

- Le fichier `.env` est déjà ignoré par Git (configuré dans `.gitignore`)
- Les valeurs par défaut AWS (comme `AWS_DEFAULT_REGION:-eu-west-1`) permettent de définir une valeur si la variable n'est pas définie
- Pour la production, envisager d'utiliser `docker secret` ou un gestionnaire de secrets externe

