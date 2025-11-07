"""Training orchestration and dataset management utilities"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json
import os
import sys
import io
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class TrainingManager:
    """Manage dataset loading, training configuration, and execution"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_processed = project_root / "data" / "processed"
        self.data_interim = project_root / "data" / "interim"
        self.data_raw = project_root / "data" / "raw"
        
        # S3 Configuration (for Streamlit Cloud / AWS deployment)
        # Try Streamlit secrets first (for Streamlit Cloud), then environment variables
        try:
            import streamlit as st
            # Streamlit secrets are available when running in Streamlit
            self.s3_bucket = st.secrets.get("S3_DATA_BUCKET", os.getenv("S3_DATA_BUCKET", ""))
            self.s3_prefix = st.secrets.get("S3_DATA_PREFIX", os.getenv("S3_DATA_PREFIX", "data/"))
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
            aws_region = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
        except (ImportError, AttributeError, KeyError, FileNotFoundError):
            # FileNotFoundError occurs when secrets.toml doesn't exist (localhost)
            # Fallback to environment variables if Streamlit is not available
            self.s3_bucket = os.getenv("S3_DATA_BUCKET", "")
            self.s3_prefix = os.getenv("S3_DATA_PREFIX", "data/")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
        
        self.use_s3 = bool(self.s3_bucket)
        
        # Initialize S3 client if bucket is configured
        self.s3_client = None
        if self.use_s3:
            try:
                # Try to get credentials from environment or Streamlit secrets
                # Streamlit Cloud uses secrets.toml for credentials
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                # Test S3 connection
                try:
                    self.s3_client.head_bucket(Bucket=self.s3_bucket)
                except (ClientError, NoCredentialsError) as e:
                    print(f"S3 bucket access failed: {e}. Falling back to local files.")
                    self.use_s3 = False
                    self.s3_client = None
            except Exception as e:
                print(f"S3 initialization failed: {e}. Falling back to local files.")
                self.use_s3 = False
                self.s3_client = None
    
    def _load_from_s3(self, s3_key: str) -> Optional[pd.DataFrame]:
        """Load a CSV file from S3"""
        if not self.use_s3 or not self.s3_client:
            return None
        
        try:
            full_key = f"{self.s3_prefix.rstrip('/')}/{s3_key.lstrip('/')}"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=full_key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            return df
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchKey':
                print(f"S3 key not found: {full_key}")
            else:
                print(f"S3 error loading {full_key}: {e}")
            return None
        except Exception as e:
            print(f"Error loading from S3: {e}")
            return None
    
    def _load_image_from_s3(self, image_filename: str) -> Optional[bytes]:
        """Load an image file from S3"""
        if not self.use_s3 or not self.s3_client:
            return None
        
        try:
            # Images are typically in data/raw/images/image_train/
            s3_key = f"{self.s3_prefix.rstrip('/')}/raw/images/image_train/{image_filename}"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code != 'NoSuchKey':  # Don't log missing images as errors
                print(f"S3 error loading image {image_filename}: {e}")
            return None
        except Exception as e:
            print(f"Error loading image from S3: {e}")
            return None
    
    def load_dataset_sample(self, sample_size: Optional[int] = None, 
                           categories: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load a sample of the training dataset
        
        Args:
            sample_size: Number of rows to load (None = all)
            categories: List of category codes to filter (None = all)
        
        Returns:
            pd.DataFrame: Sampled dataset
        """
        try:
            df = None
            
            # Try S3 first if configured
            if self.use_s3:
                # Try loading from S3: interim/merged_train.csv
                df = self._load_from_s3("interim/merged_train.csv")
                if df is None or df.empty:
                    # Fallback to processed features from S3
                    df = self._load_from_s3("processed/train_features.csv")
            
            # Fallback to local files if S3 failed or not configured
            if df is None or df.empty:
                merged_train_path = self.data_interim / "merged_train.csv"
                if merged_train_path.exists():
                    df = pd.read_csv(merged_train_path)
                else:
                    # Fallback to processed features
                    train_features_path = self.data_processed / "train_features.csv"
                    if train_features_path.exists():
                        df = pd.read_csv(train_features_path)
                    else:
                        return pd.DataFrame()
            
            # Filter by categories if specified
            if categories:
                df = df[df['prdtypecode'].isin(categories)]
            
            # Sample if specified
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
            
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the full dataset"""
        try:
            df = None
            
            # Try S3 first if configured
            if self.use_s3:
                # Try loading from S3: interim/merged_train.csv
                df = self._load_from_s3("interim/merged_train.csv")
                if df is None or df.empty:
                    # Fallback to processed features from S3
                    df = self._load_from_s3("processed/train_features.csv")
            
            # Fallback to local files if S3 failed or not configured
            if df is None or df.empty:
                merged_train_path = self.data_interim / "merged_train.csv"
                if merged_train_path.exists():
                    df = pd.read_csv(merged_train_path)
                else:
                    # Fallback to processed features
                    train_features_path = self.data_processed / "train_features.csv"
                    if train_features_path.exists():
                        df = pd.read_csv(train_features_path)
                    else:
                        return {}
            
            # Calculate average text length from both designation and description
            avg_text_length = 0
            if 'designation' in df.columns and 'description' in df.columns:
                # Combine both fields and calculate average length
                combined_text = (df['designation'].fillna('') + ' ' + df['description'].fillna(''))
                avg_text_length = combined_text.str.len().mean()
            elif 'designation' in df.columns:
                avg_text_length = df['designation'].fillna('').str.len().mean()
            elif 'description' in df.columns:
                avg_text_length = df['description'].fillna('').str.len().mean()
            
            return {
                'total_samples': len(df),
                'num_categories': df['prdtypecode'].nunique(),
                'category_distribution': df['prdtypecode'].value_counts().to_dict(),
                'avg_text_length': avg_text_length,
                'missing_designation': df['designation'].isna().sum() if 'designation' in df.columns else 0,
                'missing_description': df['description'].isna().sum() if 'description' in df.columns else 0
            }
        except Exception as e:
            print(f"Error getting dataset statistics: {e}")
            return {}
    
    def load_image(self, image_filename: str) -> Optional[bytes]:
        """
        Load an image file from S3 or local filesystem
        
        Args:
            image_filename: Name of the image file (e.g., "image_123_product_456.jpg")
        
        Returns:
            bytes: Image data, or None if not found
        """
        # Try S3 first if configured
        if self.use_s3:
            image_data = self._load_image_from_s3(image_filename)
            if image_data:
                return image_data
        
        # Fallback to local filesystem
        images_path = self.data_raw / "images" / "image_train"
        img_path = images_path / image_filename
        if img_path.exists():
            try:
                return img_path.read_bytes()
            except Exception as e:
                print(f"Error reading local image {image_filename}: {e}")
                return None
        
        return None
    
    def create_training_config(self, 
                              sample_size: Optional[int] = None,
                              use_images: bool = True,
                              hyperparams: Optional[Dict] = None) -> Dict:
        """Create a training configuration dictionary"""
        config = {
            'sample_size': sample_size or 'full',
            'use_images': use_images,
            'modality': 'text+image' if use_images else 'text',
            'hyperparams': hyperparams or {}
        }
        return config
    
    def trigger_training(self, config: Dict) -> tuple[bool, str, Optional[str]]:
        """
        Trigger a training job with the given configuration
        
        Returns:
            tuple: (success: bool, message: str, run_id: Optional[str])
        """
        try:
            # Prepare environment variables
            env_vars = {
                'SAMPLE_SIZE': str(config.get('sample_size', 'full')),
                'USE_IMAGES': str(config.get('use_images', True)),
            }
            
            # Add MLflow tracking URI and host if available
            try:
                import streamlit as st
                try:
                    from streamlit_app.utils.constants import MLFLOW_TRACKING_URI, MLFLOW_HOST
                    env_vars['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
                    env_vars['MLFLOW_HOST'] = MLFLOW_HOST
                except (ImportError, AttributeError):
                    # Fallback to environment variable
                    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
                    mlflow_host = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
                    if mlflow_uri:
                        env_vars['MLFLOW_TRACKING_URI'] = mlflow_uri
                    env_vars['MLFLOW_HOST'] = mlflow_host
            except ImportError:
                # Not in Streamlit context, use environment variable
                mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
                mlflow_host = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
                if mlflow_uri:
                    env_vars['MLFLOW_TRACKING_URI'] = mlflow_uri
                env_vars['MLFLOW_HOST'] = mlflow_host
            
            # Add S3 configuration if available
            try:
                import streamlit as st
                s3_bucket = st.secrets.get("S3_DATA_BUCKET", os.getenv("S3_DATA_BUCKET", ""))
                s3_prefix = st.secrets.get("S3_DATA_PREFIX", os.getenv("S3_DATA_PREFIX", "data/"))
                aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
                aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
                aws_region = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
                if s3_bucket:
                    env_vars['S3_DATA_BUCKET'] = s3_bucket
                    env_vars['S3_DATA_PREFIX'] = s3_prefix
                if aws_access_key:
                    env_vars['AWS_ACCESS_KEY_ID'] = aws_access_key
                if aws_secret_key:
                    env_vars['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
                if aws_region:
                    env_vars['AWS_DEFAULT_REGION'] = aws_region
            except (ImportError, AttributeError, FileNotFoundError):
                # FileNotFoundError occurs when secrets.toml doesn't exist (localhost)
                s3_bucket = os.getenv("S3_DATA_BUCKET")
                s3_prefix = os.getenv("S3_DATA_PREFIX")
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
                if s3_bucket:
                    env_vars['S3_DATA_BUCKET'] = s3_bucket
                if s3_prefix:
                    env_vars['S3_DATA_PREFIX'] = s3_prefix
                if aws_access_key:
                    env_vars['AWS_ACCESS_KEY_ID'] = aws_access_key
                if aws_secret_key:
                    env_vars['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
                if aws_region:
                    env_vars['AWS_DEFAULT_REGION'] = aws_region
            
            # Add hyperparameters
            if 'hyperparams' in config:
                for key, value in config['hyperparams'].items():
                    env_vars[f'HYPERPARAM_{key.upper()}'] = str(value)
            
            # Build command - use sys.executable to ensure same Python interpreter
            python_executable = sys.executable
            cmd = f"cd {self.project_root} && "
            
            # Export environment variables (properly quote values to handle special characters)
            for key, value in env_vars.items():
                # Escape single quotes in value and wrap in single quotes
                escaped_value = str(value).replace("'", "'\"'\"'")
                cmd += f"export {key}='{escaped_value}' && "
            
            # Run training script with the same Python interpreter
            cmd += f"{python_executable} src/models/train_model.py"
            
            print(f"Executing: {cmd}")
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                # Try to extract run_id from output
                run_id = None
                for line in result.stdout.split('\n'):
                    if 'run_id' in line.lower():
                        # Try to extract run_id (this is a simple heuristic)
                        parts = line.split()
                        for part in parts:
                            if len(part) == 32 and all(c in '0123456789abcdef' for c in part):
                                run_id = part
                                break
                
                # Check for warnings in output (e.g., fallback to text-only mode)
                output_lines = result.stdout.split('\n')
                warnings = [line for line in output_lines if 'WARNING' in line or '⚠️' in line or 'falling back' in line.lower()]
                
                success_message = "✅ Training completed successfully!"
                if warnings:
                    # Include warnings in the message
                    warning_text = '\n'.join(warnings[:3])  # Show first 3 warnings
                    success_message += f"\n\n⚠️ Note: {warning_text}"
                
                return True, success_message, run_id
            else:
                return False, f"❌ Training failed:\n{result.stderr}", None
                
        except subprocess.TimeoutExpired:
            return False, "❌ Training timeout (>10 minutes)", None
        except Exception as e:
            return False, f"❌ Error: {str(e)}", None
    
    def get_training_progress(self, run_id: str) -> Optional[Dict]:
        """
        Get training progress for a specific run
        Note: This is a placeholder - real implementation would need 
        MLflow tracking or a progress file
        """
        # This would query MLflow or read from a progress file
        # For now, return None (not implemented)
        return None
    
    def export_sample_dataset(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Export a sample dataset for inspection"""
        try:
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            return False

