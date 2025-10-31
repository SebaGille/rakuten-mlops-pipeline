# Security & Environment Configuration

## 📋 Overview

This project uses environment variables stored in a `.env` file to manage sensitive credentials securely.

## 🔐 Configuration Files

### `.env` (NOT versioned)
- Contains **actual credentials**
- Ignored by Git (see `.gitignore`)
- Must be created locally on each environment
- **NEVER commit this file**

### `.env.example` (Versioned)
- Template documenting required variables
- Contains example/placeholder values
- Safe to version in Git
- Serves as documentation

## ⚙️ Required Environment Variables

### Grafana (Monitoring Dashboard)
```bash
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=your_secure_password
```

### PostgreSQL (MLflow Backend)
```bash
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=mlflow
```

### AWS S3 (MLflow Artifacts Storage)
```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=eu-west-1
S3_BUCKET_NAME=your-bucket-name
```

## 🚀 Quick Setup

### Initial Configuration

1. Clone the repository
2. Copy the template file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` with your actual credentials
4. Start services:
   ```bash
   docker-compose -f docker-compose.api.yml up -d
   docker-compose -f docker-compose.monitor.yml up -d
   ```

### Running the Pipeline

Load environment variables before running scripts:
```bash
# Activate virtual environment
source .venv/bin/activate

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run pipeline
python flows/pipeline_flow.py
```

## 🔍 Verification

Check that environment variables are loaded:

```bash
# Verify .env exists
ls -la .env

# Check variables in Docker Compose
docker-compose -f docker-compose.api.yml config

# Check variables in shell
echo $AWS_ACCESS_KEY_ID
echo $POSTGRES_USER
```

## 🛡️ Security Best Practices

1. ✅ **Never commit `.env`** (already in `.gitignore`)
2. ✅ **Always commit `.env.example`**
3. ⚠️ **Use strong passwords** in production
4. 🔄 **Rotate secrets regularly** (AWS, PostgreSQL, Grafana)
5. 🔐 **Use secret managers in production** (AWS Secrets Manager, HashiCorp Vault, etc.)

## 📝 Security Checklist

- [x] `.env` is in `.gitignore`
- [x] `.env.example` is versioned
- [x] No hardcoded credentials in `docker-compose*.yml`
- [x] Variables documentation created
- [ ] Secrets rotation procedure documented
- [ ] CI/CD secrets configured (if applicable)
- [ ] Production uses a secret manager

## 🆘 Troubleshooting

### Error: "variable not set"
```bash
# Verify .env exists in project root
pwd
ls .env

# Check file content (redact sensitive values)
cat .env | grep -v PASSWORD | grep -v KEY
```

### Variables not loaded
```bash
# Ensure you're in the correct directory
cd /path/to/sep25_cmlops_rakuten

# Restart containers
docker-compose -f docker-compose.api.yml down
docker-compose -f docker-compose.api.yml up -d
```

### MLflow: "OSError: Read-only file system"
This means environment variables are not loaded. Solution:
```bash
export $(cat .env | grep -v '^#' | xargs)
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### PostgreSQL connection refused
Reset database volumes if credentials changed:
```bash
docker-compose -f docker-compose.mlflow.yml down -v  # -v removes volumes
docker-compose -f docker-compose.mlflow.yml up -d
```

## 🔒 Production Deployment

For production environments, **do not use `.env` files**. Instead:

1. **Use secret management services:**
   - AWS Secrets Manager / Parameter Store
   - Azure Key Vault
   - Google Cloud Secret Manager
   - HashiCorp Vault

2. **For CI/CD:**
   - GitHub Secrets
   - GitLab CI/CD Variables
   - Jenkins Credentials

3. **For Kubernetes:**
   - Kubernetes Secrets
   - External Secrets Operator

