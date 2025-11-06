"""MLflow tracking and model registry utilities"""
import mlflow
import pandas as pd
from typing import Dict, List, Optional
import requests
from urllib.parse import urljoin
from mlflow.tracking import MlflowClient


class MLflowManager:
    """Manage MLflow experiments, runs, and model registry"""
    
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
    
    def check_connection(self) -> bool:
        """Check if MLflow server is accessible"""
        # First try the official MLflow client API; this works even when the UI
        # is served behind a path prefix (e.g. ALB forwarding /mlflow → service).
        try:
            self.client.search_experiments(max_results=1)
            return True
        except Exception as client_error:
            last_error = client_error
        
        # Fallback to simple HTTP checks on common endpoints.
        base_url = self.tracking_uri.rstrip("/")
        urls_to_try = [
            urljoin(base_url + "/", "health"),
            base_url,
            base_url + "/",  # ensure trailing slash variant
        ]
        
        for url in urls_to_try:
            try:
                response = requests.get(url, timeout=5, allow_redirects=True)
                if response.status_code < 400:
                    return True
            except requests.RequestException as http_error:
                last_error = http_error
        
        print(f"MLflow connectivity check failed for {self.tracking_uri}: {last_error}")
        return False
    
    def get_experiments(self) -> List[Dict]:
        """Get all experiments"""
        try:
            experiments = self.client.search_experiments()
            return [
                {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'artifact_location': exp.artifact_location,
                    'lifecycle_stage': exp.lifecycle_stage
                }
                for exp in experiments
            ]
        except Exception as e:
            print(f"Error getting experiments: {e}")
            return []
    
    def get_runs(self, experiment_id: str, max_results: int = 10) -> pd.DataFrame:
        """Get runs for a specific experiment"""
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            data = []
            for run in runs:
                data.append({
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'status': run.info.status,
                    'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                    'duration_sec': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None,
                    'accuracy': run.data.metrics.get('accuracy', None),
                    'f1_weighted': run.data.metrics.get('f1_weighted', None),
                    'model_type': run.data.params.get('model', 'N/A'),
                    'git_commit': run.data.tags.get('git_commit', 'N/A'),
                    'auto_promotion_candidate': run.data.tags.get('auto_promotion_candidate', 'N/A'),
                    'auto_promotion_reason': run.data.tags.get('auto_promotion_reason', 'N/A')
                })
            
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error getting runs: {e}")
            return pd.DataFrame()
    
    def get_best_run(self, experiment_name: str, metric: str = 'f1_weighted') -> Optional[Dict]:
        """Get the best run for an experiment based on a metric"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                return None
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1,
                order_by=[f"metrics.{metric} DESC"]
            )
            
            if not runs:
                return None
            
            run = runs[0]
            return {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'accuracy': run.data.metrics.get('accuracy', None),
                'f1_weighted': run.data.metrics.get('f1_weighted', None),
                'params': run.data.params
            }
        except Exception as e:
            print(f"Error getting best run: {e}")
            return None
    
    def get_registered_models(self) -> List[Dict]:
        """Get all registered models"""
        try:
            models = self.client.search_registered_models()
            result = []
            for model in models:
                # Get latest versions
                versions = self.client.search_model_versions(f"name='{model.name}'")
                latest_version = max([int(v.version) for v in versions]) if versions else 0
                
                # Get champion and challenger by trying to fetch them directly by alias
                champion_version = None
                challenger_version = None
                
                # Method 1: Try to get versions by alias directly
                try:
                    champion_model = self.client.get_model_version_by_alias(model.name, "champion")
                    champion_version = champion_model.version
                except Exception:
                    pass
                
                try:
                    challenger_model = self.client.get_model_version_by_alias(model.name, "challenger")
                    challenger_version = challenger_model.version
                except Exception:
                    pass
                
                # Method 2: Fallback - search through versions for aliases
                if not champion_version or not challenger_version:
                    for version in versions:
                        if hasattr(version, 'aliases') and version.aliases:
                            if 'champion' in version.aliases and not champion_version:
                                champion_version = version.version
                            if 'challenger' in version.aliases and not challenger_version:
                                challenger_version = version.version
                
                result.append({
                    'name': model.name,
                    'latest_version': latest_version,
                    'champion_version': champion_version,
                    'challenger_version': challenger_version,
                    'description': model.description,
                    'all_versions': len(versions)
                })
            
            return result
        except Exception as e:
            print(f"Error getting registered models: {e}")
            return []
    
    def get_model_version_details(self, model_name: str, version: Optional[str] = None, alias: Optional[str] = None) -> Optional[Dict]:
        """Get details of a specific model version"""
        try:
            if alias:
                model_version = self.client.get_model_version_by_alias(model_name, alias)
            elif version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                return None
            
            # Get run details
            run = self.client.get_run(model_version.run_id)
            
            return {
                'version': model_version.version,
                'run_id': model_version.run_id,
                'status': model_version.status,
                'aliases': model_version.aliases if hasattr(model_version, 'aliases') else [],
                'creation_timestamp': pd.to_datetime(model_version.creation_timestamp, unit='ms'),
                'accuracy': run.data.metrics.get('accuracy', None),
                'f1_weighted': run.data.metrics.get('f1_weighted', None),
                'params': run.data.params,
                'tags': run.data.tags
            }
        except Exception as e:
            print(f"Error getting model version details: {e}")
            return None
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs side by side with all available data"""
        try:
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                
                # Extract all relevant data
                run_data = {
                    'run_id': run_id,  # Keep full run_id for matching
                    'run_id_short': run_id[:8],  # Short version for display
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'accuracy': run.data.metrics.get('accuracy', None),
                    'f1_weighted': run.data.metrics.get('f1_weighted', None),
                    'f1_macro': run.data.metrics.get('f1_macro', None),
                    'f1_micro': run.data.metrics.get('f1_micro', None),
                    'model': run.data.params.get('model', 'N/A'),
                    'max_features': run.data.params.get('vec_max_features', 'N/A'),
                    'ngram_range': run.data.params.get('ngram_range', 'N/A'),
                    'max_iter': run.data.params.get('max_iter', 'N/A'),
                    'solver': run.data.params.get('solver', 'N/A'),
                    'modality': run.data.params.get('modality', 'text'),
                    'sample_size': run.data.params.get('sample_size', 'N/A'),
                    'git_commit': run.data.tags.get('git_commit', 'N/A')[:7] if run.data.tags.get('git_commit', 'N/A') != 'N/A' else 'N/A',
                    'git_branch': run.data.tags.get('git_branch', 'N/A'),
                    'auto_promotion_candidate': run.data.tags.get('auto_promotion_candidate', 'N/A'),
                    'auto_promotion_reason': run.data.tags.get('auto_promotion_reason', 'N/A'),
                    'start_time': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S') if run.info.start_time else 'N/A',
                    'duration_min': round((run.info.end_time - run.info.start_time) / 1000 / 60, 2) if run.info.end_time and run.info.start_time else None,
                }
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
        except Exception as e:
            print(f"Error comparing runs: {e}")
            return pd.DataFrame()
    
    def log_run_metrics(self, experiment_name: str, params: Dict, metrics: Dict, tags: Dict = None) -> str:
        """Log a new run with metrics"""
        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                if tags:
                    mlflow.set_tags(tags)
                return run.info.run_id
        except Exception as e:
            print(f"Error logging run: {e}")
            return None

