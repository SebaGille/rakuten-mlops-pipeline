"""Training orchestration and dataset management utilities"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json


class TrainingManager:
    """Manage dataset loading, training configuration, and execution"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_processed = project_root / "data" / "processed"
        self.data_interim = project_root / "data" / "interim"
        self.data_raw = project_root / "data" / "raw"
    
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
            # Prefer loading from interim (has original text) for better viewing
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
            # Try to load from interim folder first (has original text data)
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
            
            # Add hyperparameters
            if 'hyperparams' in config:
                for key, value in config['hyperparams'].items():
                    env_vars[f'HYPERPARAM_{key.upper()}'] = str(value)
            
            # Build command
            cmd = f"cd {self.project_root} && "
            
            # Export environment variables
            for key, value in env_vars.items():
                cmd += f"export {key}={value} && "
            
            # Run training script
            cmd += "python src/models/train_model.py"
            
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
                
                return True, "✅ Training completed successfully!", run_id
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

