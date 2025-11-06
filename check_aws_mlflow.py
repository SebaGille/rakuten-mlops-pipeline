#!/usr/bin/env python3
"""
Test script to verify AWS MLflow connectivity and fetch experiments/runs
Usage: python test_aws_mlflow.py [MLFLOW_TRACKING_URI] [MLFLOW_HOST]
"""
import os
import sys
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import requests
from functools import wraps

def _configure_mlflow_host_header(tracking_uri: str, mlflow_host: str):
    """Configure MLflow to use custom Host header for host-based routing"""
    try:
        # Store original request method
        original_request = requests.Session.request
        tracking_uri_base = tracking_uri.rstrip("/")
        
        @wraps(original_request)
        def request_with_host_header(session_self, method, url, *args, **kwargs):
            # If the URL matches our tracking URI, add Host header
            if tracking_uri_base and url.startswith(tracking_uri_base):
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers']['Host'] = mlflow_host
            return original_request(session_self, method, url, *args, **kwargs)
        
        # Patch the Session class
        requests.Session.request = request_with_host_header
        return True
    except Exception as e:
        print(f"Warning: Failed to configure Host header: {e}")
        return False

def check_mlflow_connection(tracking_uri: str, mlflow_host: str = None):
    """Test MLflow connection and fetch experiments/runs"""
    print("=" * 70)
    print("Testing MLflow Connection")
    print("=" * 70)
    print(f"Tracking URI: {tracking_uri}")
    if mlflow_host:
        print(f"MLflow Host: {mlflow_host}")
        print("Configuring host-based routing...")
        _configure_mlflow_host_header(tracking_uri, mlflow_host)
    print()
    
    try:
        print("Connecting to MLflow...")
        client = MlflowClient(tracking_uri=tracking_uri)
        
        print("Searching experiments...")
        experiments = client.search_experiments(max_results=100)
        
        print(f"✓ Found {len(experiments)} experiment(s):\n")
        
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
            # Get run count
            try:
                runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
                print(f"    Runs: {len(runs)}")
                
                # Show recent runs
                if runs:
                    print(f"    Recent runs:")
                    for run in runs[:5]:
                        status = run.info.status
                        run_name = run.data.tags.get('mlflow.runName', 'N/A')
                        print(f"      - {run.info.run_id[:8]}... {run_name} (Status: {status})")
            except Exception as e:
                print(f"    Runs: Error - {e}")
            print()
        
        # Check for the expected experiment
        expected_name = "rakuten-multimodal-text-image"
        found_exp = next((e for e in experiments if e.name == expected_name), None)
        
        if found_exp:
            print("=" * 70)
            print(f"Expected Experiment: {expected_name}")
            print("=" * 70)
            print(f"✓ Found experiment: {expected_name}")
            print(f"  ID: {found_exp.experiment_id}")
            
            # Get all runs for this experiment
            try:
                runs = client.search_runs(experiment_ids=[found_exp.experiment_id], max_results=1000)
                print(f"  Total runs: {len(runs)}")
                
                if runs:
                    print(f"\n  All runs for this experiment:")
                    for i, run in enumerate(runs, 1):
                        run_id = run.info.run_id
                        run_name = run.data.tags.get('mlflow.runName', 'N/A')
                        status = run.info.status
                        accuracy = run.data.metrics.get('accuracy', 'N/A')
                        f1 = run.data.metrics.get('f1_weighted', 'N/A')
                        print(f"    {i}. {run_id[:8]}... | {run_name} | Status: {status} | Acc: {accuracy} | F1: {f1}")
                else:
                    print("  No runs found for this experiment")
            except Exception as e:
                print(f"  ✗ Error getting runs: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("=" * 70)
            print(f"Expected Experiment: {expected_name}")
            print("=" * 70)
            print(f"✗ Expected experiment '{expected_name}' NOT FOUND")
            print(f"  Available experiments: {[e.name for e in experiments]}")
        
        # Test registered models
        print("\n" + "=" * 70)
        print("Registered Models")
        print("=" * 70)
        try:
            models = client.search_registered_models(max_results=100)
            if models:
                print(f"✓ Found {len(models)} registered model(s):\n")
                for model in models:
                    versions = client.search_model_versions(f"name='{model.name}'")
                    print(f"  - {model.name} ({len(versions)} versions)")
            else:
                print("  No registered models found")
        except Exception as e:
            print(f"  ✗ Error getting registered models: {e}")
        
        print("\n" + "=" * 70)
        print("✓ Test Complete - Connection Successful")
        print("=" * 70)
        return True
        
    except MlflowException as e:
        print(f"\n✗ MLflow Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Error connecting to MLflow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get tracking URI from command line argument or environment variable
    if len(sys.argv) > 1:
        tracking_uri = sys.argv[1]
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    
    # Get MLflow host from command line argument or environment variable
    if len(sys.argv) > 2:
        mlflow_host = sys.argv[2]
    else:
        mlflow_host = os.getenv("MLFLOW_HOST", None)
    
    success = check_mlflow_connection(tracking_uri, mlflow_host)
    sys.exit(0 if success else 1)

