"""Prediction and inference utilities"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import os
import logging
from datetime import datetime
from PIL import Image
import io

# Optional S3 support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Import configuration
from streamlit_app.utils.config import (
    API_REQUEST_TIMEOUT,
    API_HEALTH_CHECK_TIMEOUT
)

# Import API_HOST from constants to ensure it reads from Streamlit secrets
try:
    from streamlit_app.utils.constants import API_HOST
except ImportError:
    # Fallback if constants module is not available
    API_HOST = os.getenv("API_HOST", "api.rakuten.dev")

# Set up logger
logger = logging.getLogger(__name__)


class PredictionManager:
    """Manage model predictions and inference logging"""
    
    def __init__(self, api_url: str, project_root: Path):
        self.api_url = api_url.rstrip("/")
        self.project_root = project_root
        self.monitoring_dir = project_root / "data" / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.inference_log_path = self.monitoring_dir / "inference_log.csv"
        
        # Get API host for host-based routing (if using ALB)
        # Use API_HOST from constants which reads from Streamlit secrets
        self.api_host = API_HOST
        
        # Configure host-based routing for AWS ALB (similar to MLflowManager)
        self._configure_api_host_header()
        
        # S3 Configuration (for Streamlit Cloud / AWS deployment)
        try:
            import streamlit as st
            self.s3_bucket = st.secrets.get("S3_DATA_BUCKET", os.getenv("S3_DATA_BUCKET", ""))
            self.s3_prefix = st.secrets.get("S3_DATA_PREFIX", os.getenv("S3_DATA_PREFIX", "data/"))
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
            aws_region = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
        except (ImportError, AttributeError, KeyError):
            self.s3_bucket = os.getenv("S3_DATA_BUCKET", "")
            self.s3_prefix = os.getenv("S3_DATA_PREFIX", "data/")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
        
        self.use_s3 = bool(self.s3_bucket) and S3_AVAILABLE
        
        # Initialize S3 client if bucket is configured
        self.s3_client = None
        if self.use_s3:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
            except Exception as e:
                logger.warning(f"S3 initialization failed: {e}. Falling back to local files.")
                self.use_s3 = False
                self.s3_client = None
    
    def _configure_api_host_header(self):
        """Configure API requests to use custom Host header for host-based routing"""
        try:
            from functools import wraps
            
            api_url = self.api_url.rstrip("/")
            api_host = self.api_host
            
            # Store original request method if not already stored
            if not hasattr(requests.Session, '_original_request'):
                requests.Session._original_request = requests.Session.request
            
            # Get the original request method (or the already-patched one)
            original_request = requests.Session._original_request
            
            # Check if MLflowManager has already patched (it would have stored MLflow URL info)
            # We need to work together with MLflowManager's patching
            try:
                from streamlit_app.utils.constants import MLFLOW_TRACKING_URI, MLFLOW_HOST
                mlflow_url = MLFLOW_TRACKING_URI.rstrip("/") if MLFLOW_TRACKING_URI else None
                mlflow_host = MLFLOW_HOST
            except:
                mlflow_url = None
                mlflow_host = None
            
            @wraps(original_request)
            def request_with_host_header(session_self, method, url, *args, **kwargs):
                # Ensure headers dict exists
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                
                url_normalized = url.rstrip("/")
                
                # Check for API URL first
                if api_url and (url_normalized.startswith(api_url) or url.startswith(api_url)):
                    kwargs['headers']['Host'] = api_host
                # Check for MLflow URL (if MLflowManager hasn't already handled it)
                elif mlflow_url and (url_normalized.startswith(mlflow_url) or url.startswith(mlflow_url)):
                    if 'Host' not in kwargs['headers']:  # Only set if not already set
                        kwargs['headers']['Host'] = mlflow_host
                
                return original_request(session_self, method, url, *args, **kwargs)
            
            # Only patch if not already patched (or re-patch to include API support)
            requests.Session.request = request_with_host_header
            logger.info(f"Configured host-based routing: {api_url} -> Host: {api_host}")
        except Exception as e:
            logger.warning(f"Failed to configure Host header: {e}", exc_info=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests with host-based routing"""
        headers = {}
        if self.api_host and (self.api_url.startswith("http://") or self.api_url.startswith("https://")):
            headers['Host'] = self.api_host
        return headers
    
    def check_api_health(self) -> Tuple[bool, Optional[str]]:
        """Check if the API is accessible with host-based routing support
        
        Returns:
            tuple: (is_healthy, error_message)
        """
        try:
            # Ensure host header is configured
            self._configure_api_host_header()
            
            headers = self._get_headers()
            
            # Try multiple health check endpoints
            health_urls = [
                f"{self.api_url}/health",
                f"{self.api_url.rstrip('/')}/health",
                f"{self.api_url}/",
            ]
            
            last_error_msg = None
            
            for url in health_urls:
                try:
                    logger.debug(f"Trying health check: {url} with headers: {headers}")
                    response = requests.get(
                        url, 
                        timeout=API_HEALTH_CHECK_TIMEOUT, 
                        headers=headers,
                        allow_redirects=True
                    )
                    if response.status_code == 200:
                        logger.info(f"API health check successful: {url}")
                        return (True, None)
                    else:
                        logger.debug(f"API returned status {response.status_code} for {url}")
                        last_error_msg = f"API returned status {response.status_code}"
                except requests.exceptions.Timeout:
                    logger.warning(f"API health check timed out for {url}")
                    last_error_msg = f"Request timed out after {API_HEALTH_CHECK_TIMEOUT} seconds"
                    continue
                except requests.exceptions.ConnectionError as e:
                    logger.warning(f"API connection error for {url}: {e}")
                    last_error_msg = f"Connection failed: Could not reach API at {self.api_url}. The API may not be deployed or running."
                    continue
                except requests.exceptions.RequestException as e:
                    logger.warning(f"API request error for {url}: {e}")
                    last_error_msg = f"Request failed: {str(e)}"
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error during API health check for {url}: {e}")
                    last_error_msg = f"Unexpected error: {str(e)}"
                    continue
            
            # All health checks failed
            error_msg = (
                f"API is not accessible at {self.api_url}. "
                f"Please ensure:\n"
                f"1. The API is deployed and running\n"
                f"2. The ALB is configured correctly\n"
                f"3. The API_HOST ({self.api_host}) is set correctly for host-based routing\n"
                f"4. Network connectivity is available"
            )
            if last_error_msg:
                error_msg = f"{error_msg}\n\nLast error: {last_error_msg}"
            return (False, error_msg)
        except Exception as e:
            # Catch any unexpected errors and ensure we always return a tuple
            logger.error(f"Unexpected error in check_api_health: {e}", exc_info=True)
            error_msg = f"Unexpected error during health check: {str(e)}"
            return (False, error_msg)
    
    def predict(self, designation: str, description: str, 
                image: Optional[bytes] = None) -> Optional[Dict]:
        """
        Make a prediction using the API with host-based routing support
        
        Args:
            designation: Product designation/title
            description: Product description
            image: Optional image bytes
        
        Returns:
            dict: Prediction results or None if failed
        """
        try:
            # Ensure host header is configured
            self._configure_api_host_header()
            
            # For now, only text-based prediction (API doesn't support images yet)
            payload = {
                "designation": designation,
                "description": description
            }
            
            headers = self._get_headers()
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=API_REQUEST_TIMEOUT,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Log the prediction
                self.log_prediction(designation, description, result)
                
                return result
            else:
                return {
                    'error': True,
                    'message': f"API returned status {response.status_code}",
                    'details': response.text
                }
        except Exception as e:
            return {
                'error': True,
                'message': f"Prediction failed: {str(e)}"
            }
    
    def predict_local(self, designation: str, description: str, 
                     image_path: Optional[Path] = None) -> Optional[Dict]:
        """
        Make a prediction using local model (fallback if API unavailable)
        This is a placeholder - would load model from MLflow in production
        """
        # TODO: Implement local prediction using joblib model
        return {
            'error': True,
            'message': 'Local prediction not implemented yet. Please ensure API is running.'
        }
    
    def get_prediction_confidence(self, prediction: Dict) -> List[Tuple[int, str, float]]:
        """
        Extract top predictions with confidence scores
        
        Returns:
            list: [(category_code, category_name, confidence), ...]
        """
        # This would extract confidence scores if available from API
        # For now, return placeholder
        if 'predicted_class' in prediction:
            return [(prediction['predicted_class'], 'Unknown', 1.0)]
        return []
    
    def _save_to_s3(self, s3_key: str, content: bytes) -> bool:
        """Save content to S3"""
        if not self.use_s3 or not self.s3_client:
            return False
        
        try:
            full_key = f"{self.s3_prefix.rstrip('/')}/{s3_key.lstrip('/')}"
            self.s3_client.put_object(Bucket=self.s3_bucket, Key=full_key, Body=content)
            return True
        except Exception as e:
            logger.warning(f"S3 error saving {s3_key}: {e}")
            return False
    
    def _load_from_s3(self, s3_key: str) -> Optional[pd.DataFrame]:
        """Load CSV from S3"""
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
                return None  # File doesn't exist yet
            logger.warning(f"S3 error loading {s3_key}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error loading from S3: {e}")
            return None
    
    def _append_to_s3_csv(self, s3_key: str, new_row: Dict) -> bool:
        """Append a row to CSV in S3"""
        if not self.use_s3 or not self.s3_client:
            return False
        
        try:
            # Load existing data
            df = self._load_from_s3(s3_key)
            if df is None:
                # Create new DataFrame
                df = pd.DataFrame([new_row])
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save back to S3
            csv_content = df.to_csv(index=False).encode('utf-8')
            return self._save_to_s3(s3_key, csv_content)
        except Exception as e:
            logger.warning(f"Error appending to S3 CSV: {e}")
            return False
    
    def log_prediction(self, designation: str, description: str, 
                      prediction: Dict) -> None:
        """Log prediction to CSV for monitoring (local or S3)"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'designation': designation,
                'description': description,
                'predicted_class': prediction.get('predicted_class', None),
                'designation_length': len(designation),
                'description_length': len(description) if description else 0
            }
            
            # Try S3 first if configured
            if self.use_s3:
                success = self._append_to_s3_csv("monitoring/inference_log.csv", log_entry)
                if success:
                    return
            
            # Fallback to local file
            df = pd.DataFrame([log_entry])
            if self.inference_log_path.exists():
                df.to_csv(self.inference_log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.inference_log_path, mode='w', header=True, index=False)
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}", exc_info=True)
    
    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent prediction history (from S3 or local)"""
        try:
            df = None
            
            # Try S3 first if configured
            if self.use_s3:
                df = self._load_from_s3("monitoring/inference_log.csv")
            
            # Fallback to local file
            if df is None or df.empty:
                if self.inference_log_path.exists():
                    df = pd.read_csv(self.inference_log_path, on_bad_lines='skip', encoding='utf-8')
                else:
                    return pd.DataFrame()
            
            # Handle old column name for backward compatibility
            if 'predicted_prdtypecode' in df.columns and 'predicted_class' not in df.columns:
                df.rename(columns={'predicted_prdtypecode': 'predicted_class'}, inplace=True)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'designation', 'predicted_class']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column '{col}' in inference log")
            
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Error loading prediction history: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_prediction_statistics(self) -> Dict:
        """Get statistics about predictions (from S3 or local)"""
        try:
            df = None
            
            # Try S3 first if configured
            if self.use_s3:
                df = self._load_from_s3("monitoring/inference_log.csv")
            
            # Fallback to local file
            if df is None or df.empty:
                if not self.inference_log_path.exists():
                    return {}
                df = pd.read_csv(self.inference_log_path, on_bad_lines='skip', encoding='utf-8')
            
            # Handle old column name for backward compatibility
            if 'predicted_prdtypecode' in df.columns and 'predicted_class' not in df.columns:
                df.rename(columns={'predicted_prdtypecode': 'predicted_class'}, inplace=True)
            
            stats = {
                'total_predictions': len(df),
                'unique_classes': 0,
                'class_distribution': {},
                'avg_designation_length': 0,
                'avg_description_length': 0,
                'predictions_today': 0
            }
            
            # Only compute stats if columns exist
            if 'predicted_class' in df.columns:
                stats['unique_classes'] = df['predicted_class'].nunique()
                stats['class_distribution'] = df['predicted_class'].value_counts().to_dict()
            
            if 'designation_length' in df.columns:
                stats['avg_designation_length'] = df['designation_length'].mean()
            elif 'designation' in df.columns:
                stats['avg_designation_length'] = df['designation'].str.len().mean()
            
            if 'description_length' in df.columns:
                stats['avg_description_length'] = df['description_length'].mean()
            elif 'description' in df.columns:
                stats['avg_description_length'] = df['description'].fillna('').str.len().mean()
            
            if 'timestamp' in df.columns:
                df_temp = df.copy()
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], errors='coerce')
                stats['predictions_today'] = len(df_temp[df_temp['timestamp'].dt.date == datetime.now().date()])
            
            return stats
        except Exception as e:
            logger.error(f"Error getting prediction statistics: {e}", exc_info=True)
            return {
                'total_predictions': 0,
                'unique_classes': 0,
                'class_distribution': {},
                'avg_designation_length': 0,
                'avg_description_length': 0,
                'predictions_today': 0
            }
    
    def validate_image(self, image_bytes: bytes) -> Tuple[bool, str]:
        """Validate uploaded image"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            
            # Check format
            if img.format not in ['JPEG', 'JPG', 'PNG']:
                return False, "Image must be JPEG or PNG format"
            
            # Check size
            width, height = img.size
            if width < 50 or height < 50:
                return False, "Image too small (minimum 50x50 pixels)"
            if width > 5000 or height > 5000:
                return False, "Image too large (maximum 5000x5000 pixels)"
            
            # Check file size
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
                return False, "File size too large (maximum 10MB)"
            
            return True, "Image valid"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def clear_prediction_history(self) -> bool:
        """Clear prediction history (for testing/demo purposes)"""
        try:
            # Try S3 first if configured
            if self.use_s3 and self.s3_client:
                try:
                    full_key = f"{self.s3_prefix.rstrip('/')}/monitoring/inference_log.csv"
                    self.s3_client.delete_object(Bucket=self.s3_bucket, Key=full_key)
                    return True
                except Exception as e:
                    logger.warning(f"Error clearing S3 prediction history: {e}")
            
            # Fallback to local file
            if self.inference_log_path.exists():
                self.inference_log_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error clearing prediction history: {e}", exc_info=True)
            return False

