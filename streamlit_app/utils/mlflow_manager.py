"""MLflow tracking and model registry utilities"""
import mlflow
import pandas as pd
from typing import Dict, List, Optional, Tuple
import requests
from urllib.parse import urljoin
from mlflow.tracking import MlflowClient
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
import logging

# Import MLFLOW_HOST from constants to ensure it reads from Streamlit secrets
try:
    from streamlit_app.utils.constants import MLFLOW_HOST
except ImportError:
    # Fallback if constants module is not available
    MLFLOW_HOST = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")


class MLflowManager:
    """Manage MLflow experiments, runs, and model registry"""
    
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri.rstrip("/")
        # Get MLflow host for host-based routing (if using ALB)
        # Use MLFLOW_HOST from constants.py which reads from Streamlit secrets
        self.mlflow_host = MLFLOW_HOST
        
        # Configure MLflow to use custom Host header for host-based routing
        # MLflow uses requests internally, so we need to set the Host header
        # We'll do this by configuring the requests session used by MLflow
        self._configure_mlflow_host_header()
        
        # Set tracking URI (this is non-blocking, just sets a variable)
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        except Exception as e:
            # If setting tracking URI fails, log but continue
            print(f"Warning: Failed to set MLflow tracking URI: {e}")
        # Don't initialize client immediately - create it lazily to avoid hanging
        self._client = None
    
    def _configure_mlflow_host_header(self):
        """Configure MLflow to use custom Host header for host-based routing"""
        # MLflow uses requests internally. We need to patch the requests library
        # to add the Host header when making requests to the tracking URI
        # This is done by monkey-patching the requests.Session class
        try:
            import requests
            from functools import wraps
            
            tracking_uri = self.tracking_uri.rstrip("/")
            mlflow_host = self.mlflow_host
            
            # Store original request method if not already stored
            if not hasattr(requests.Session, '_original_request'):
                requests.Session._original_request = requests.Session.request
            
            # Get the original request method
            original_request = requests.Session._original_request
            
            @wraps(original_request)
            def request_with_host_header(session_self, method, url, *args, **kwargs):
                # If the URL matches our tracking URI, add Host header
                # Normalize URLs for comparison (remove trailing slashes)
                url_normalized = url.rstrip("/")
                if tracking_uri and (url_normalized.startswith(tracking_uri) or url.startswith(tracking_uri)):
                    if 'headers' not in kwargs:
                        kwargs['headers'] = {}
                    # Always set Host header (don't check if already set, as it might be wrong)
                    kwargs['headers']['Host'] = mlflow_host
                    print(f"[MLflowManager] Added Host header: {mlflow_host} for URL: {url} (tracking_uri: {tracking_uri})")
                return original_request(session_self, method, url, *args, **kwargs)
            
            # Patch the Session class
            requests.Session.request = request_with_host_header
            print(f"[MLflowManager] Configured host-based routing: {tracking_uri} -> Host: {mlflow_host}")
        except Exception as e:
            print(f"[MLflowManager] Warning: Failed to configure Host header: {e}")
            import traceback
            traceback.print_exc()
    
    @property
    def client(self) -> MlflowClient:
        """Lazy initialization of MLflow client to avoid hanging during __init__"""
        if self._client is None:
            # Ensure host-based routing is configured before creating client
            # Re-apply the patch in case it was overwritten
            self._configure_mlflow_host_header()
            
            # Debug: Log client initialization
            print(f"[MLflowManager] Initializing MLflowClient with tracking_uri: {self.tracking_uri}")
            print(f"[MLflowManager] Host header will be: {self.mlflow_host}")
            
            # Initialize client with timeout protection
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(MlflowClient, self.tracking_uri)
                    self._client = future.result(timeout=3)  # 3 second timeout for initialization
                print(f"[MLflowManager] MLflowClient initialized successfully")
            except FutureTimeoutError:
                error_msg = f"MLflow client initialization timed out after 3 seconds for {self.tracking_uri}"
                print(f"[MLflowManager] {error_msg}")
                raise TimeoutError(error_msg)
            except Exception as e:
                error_msg = f"Failed to initialize MLflow client: {e}"
                print(f"[MLflowManager] {error_msg}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
        return self._client
    
    def _check_mlflow_client_with_timeout(self, timeout_seconds: int = 5) -> Tuple[bool, Optional[Exception]]:
        """Check MLflow client connection with timeout using ThreadPoolExecutor"""
        def _client_check():
            try:
                # Access client property (lazy initialization) and make API call
                client = self.client  # This will trigger lazy initialization with timeout
                client.search_experiments(max_results=1)
                return True, None
            except TimeoutError as e:
                # Client initialization timed out
                return False, e
            except Exception as e:
                # API call failed
                return False, e
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_client_check)
                success, error = future.result(timeout=timeout_seconds + 3)  # Add 3s for client init timeout
                return success, error
        except FutureTimeoutError:
            return False, TimeoutError(f"MLflow client check timed out after {timeout_seconds + 3} seconds")
        except Exception as e:
            return False, e
    
    def check_connection(self) -> bool:
        """Check if MLflow server is accessible with improved timeout handling"""
        # First try a quick HTTP health check (fast, has timeout)
        base_url = self.tracking_uri.rstrip("/")
        
        # Prepare headers for host-based routing
        headers = {}
        if self.mlflow_host and (base_url.startswith("http://") or base_url.startswith("https://")):
            # If using ALB URL, add Host header for host-based routing
            headers['Host'] = self.mlflow_host
        
        # Try different health check endpoints
        health_urls = [
            urljoin(base_url + "/", "health"),
            urljoin(base_url + "/", "api/2.0/mlflow/health"),
            base_url,
            base_url + "/",
        ]
        
        http_success = False
        for url in health_urls:
            try:
                response = requests.get(url, timeout=3, allow_redirects=True, headers=headers)
                if response.status_code < 400:
                    http_success = True
                    break
            except (requests.RequestException, requests.Timeout) as e:
                continue
        
        # If HTTP check succeeded, try MLflow client API with timeout
        if http_success:
            try:
                success, error = self._check_mlflow_client_with_timeout(timeout_seconds=5)
                if success:
                    return True
                else:
                    # HTTP worked but client API failed - log but consider server accessible
                    print(f"MLflow client API check failed (but HTTP succeeded): {error}")
                    return True  # Server is accessible even if client API has issues
            except Exception as e:
                print(f"MLflow client check exception (but HTTP succeeded): {e}")
                return True  # Server is accessible even if client API has issues
        
        # If HTTP checks failed, try MLflow client API as last resort (with timeout)
        try:
            success, error = self._check_mlflow_client_with_timeout(timeout_seconds=5)
            if success:
                return True
            else:
                print(f"MLflow connectivity check failed for {self.tracking_uri}: {error}")
                return False
        except Exception as e:
            print(f"MLflow connectivity check exception for {self.tracking_uri}: {e}")
            return False
    
    def get_experiments(self) -> Tuple[List[Dict], Optional[str]]:
        """
        Get all experiments
        
        Returns:
            tuple: (experiments_list, error_message)
            - experiments_list: List of experiment dictionaries
            - error_message: Error message if failed, None if successful
        """
        try:
            # Debug: Log tracking URI and host being used
            print(f"[MLflowManager] Getting experiments from: {self.tracking_uri}")
            print(f"[MLflowManager] Using Host header: {self.mlflow_host}")
            
            # Ensure host header is configured before making the call
            self._configure_mlflow_host_header()
            
            # Use max_results to ensure we get all experiments (default might be limited)
            # Also try without view_type first, then with view_type=ALL if needed
            print(f"[MLflowManager] Calling search_experiments(max_results=1000)...")
            try:
                # Try to get all experiments (active and deleted)
                experiments = self.client.search_experiments(max_results=1000, view_type="ALL")
                print(f"[MLflowManager] Successfully retrieved {len(experiments)} experiments (view_type=ALL)")
            except Exception as e:
                # Fallback to default (active only) if view_type=ALL is not supported
                print(f"[MLflowManager] view_type=ALL not supported, trying default: {e}")
                experiments = self.client.search_experiments(max_results=1000)
                print(f"[MLflowManager] Successfully retrieved {len(experiments)} experiments (default view)")
            
            # Debug: Log experiment names and details
            if experiments:
                exp_names = [exp.name for exp in experiments]
                print(f"[MLflowManager] Experiment names: {exp_names}")
                for exp in experiments:
                    print(f"[MLflowManager]   - {exp.name} (ID: {exp.experiment_id}, Stage: {exp.lifecycle_stage})")
            else:
                print(f"[MLflowManager] Warning: No experiments returned from search_experiments()")
                print(f"[MLflowManager] This might indicate:")
                print(f"[MLflowManager]   1. Host header not being set correctly")
                print(f"[MLflowManager]   2. Connecting to wrong MLflow instance")
                print(f"[MLflowManager]   3. ALB routing not working correctly")
            
            return [
                {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'artifact_location': exp.artifact_location,
                    'lifecycle_stage': exp.lifecycle_stage
                }
                for exp in experiments
            ], None
        except Exception as e:
            error_msg = f"Error getting experiments from {self.tracking_uri}: {e}"
            print(f"[MLflowManager] {error_msg}")
            import traceback
            traceback.print_exc()
            return [], error_msg
    
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

