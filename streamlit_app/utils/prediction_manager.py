"""Prediction and inference utilities"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import io


class PredictionManager:
    """Manage model predictions and inference logging"""
    
    def __init__(self, api_url: str, project_root: Path):
        self.api_url = api_url
        self.project_root = project_root
        self.monitoring_dir = project_root / "data" / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.inference_log_path = self.monitoring_dir / "inference_log.csv"
    
    def check_api_health(self) -> bool:
        """Check if the API is accessible"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def predict(self, designation: str, description: str, 
                image: Optional[bytes] = None) -> Optional[Dict]:
        """
        Make a prediction using the API
        
        Args:
            designation: Product designation/title
            description: Product description
            image: Optional image bytes
        
        Returns:
            dict: Prediction results or None if failed
        """
        try:
            # For now, only text-based prediction (API doesn't support images yet)
            payload = {
                "designation": designation,
                "description": description
            }
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=30
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
    
    def log_prediction(self, designation: str, description: str, 
                      prediction: Dict) -> None:
        """Log prediction to CSV for monitoring"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'designation': designation,
                'description': description,
                'predicted_class': prediction.get('predicted_class', None),
                'designation_length': len(designation),
                'description_length': len(description) if description else 0
            }
            
            # Append to CSV
            df = pd.DataFrame([log_entry])
            if self.inference_log_path.exists():
                df.to_csv(self.inference_log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.inference_log_path, mode='w', header=True, index=False)
                
        except Exception as e:
            print(f"Error logging prediction: {e}")
    
    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent prediction history"""
        try:
            if self.inference_log_path.exists():
                df = pd.read_csv(self.inference_log_path, on_bad_lines='skip', encoding='utf-8')
                
                # Handle old column name for backward compatibility
                if 'predicted_prdtypecode' in df.columns and 'predicted_class' not in df.columns:
                    df.rename(columns={'predicted_prdtypecode': 'predicted_class'}, inplace=True)
                
                # Ensure required columns exist
                required_cols = ['timestamp', 'designation', 'predicted_class']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"Warning: Missing column '{col}' in inference log")
                
                return df.tail(limit)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading prediction history: {e}")
            return pd.DataFrame()
    
    def get_prediction_statistics(self) -> Dict:
        """Get statistics about predictions"""
        try:
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
            print(f"Error getting prediction statistics: {e}")
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
            if self.inference_log_path.exists():
                self.inference_log_path.unlink()
            return True
        except Exception as e:
            print(f"Error clearing prediction history: {e}")
            return False

