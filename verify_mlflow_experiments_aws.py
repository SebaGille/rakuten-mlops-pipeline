#!/usr/bin/env python3
"""
Script to verify all experiments and runs in AWS MLflow via ALB
This helps debug why Streamlit only sees 1 experiment
"""
import os
import sys
from mlflow.tracking import MlflowClient
import requests
from functools import wraps

# Configuration from environment or defaults
ALB_URL = os.getenv("AWS_ALB_URL", "http://rakuten-shared-alb-217736274.eu-west-1.elb.amazonaws.com")
MLFLOW_HOST = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
TRACKING_URI = ALB_URL.rstrip("/")

def configure_host_header(tracking_uri: str, mlflow_host: str):
    """Configure Host header for ALB host-based routing"""
    try:
        # Store original request method
        if not hasattr(requests.Session, '_original_request'):
            requests.Session._original_request = requests.Session.request
        
        original_request = requests.Session._original_request
        tracking_uri_base = tracking_uri.rstrip("/")
        
        @wraps(original_request)
        def request_with_host_header(session_self, method, url, *args, **kwargs):
            # If the URL matches our tracking URI, add Host header
            if tracking_uri_base and url.startswith(tracking_uri_base):
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers']['Host'] = mlflow_host
                print(f"[DEBUG] Request: {method} {url}")
                print(f"[DEBUG] Host header: {mlflow_host}")
            return original_request(session_self, method, url, *args, **kwargs)
        
        # Patch the Session class
        requests.Session.request = request_with_host_header
        print(f"✓ Configured host-based routing: {tracking_uri} -> Host: {mlflow_host}\n")
        return True
    except Exception as e:
        print(f"✗ Failed to configure Host header: {e}")
        return False

def main():
    print("=" * 70)
    print("AWS MLflow Experiments & Runs Verification")
    print("=" * 70)
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"MLflow Host: {MLFLOW_HOST}")
    print()
    
    # Configure host-based routing
    if not configure_host_header(TRACKING_URI, MLFLOW_HOST):
        print("Failed to configure Host header. Exiting.")
        sys.exit(1)
    
    try:
        # Initialize MLflow client
        print("Connecting to MLflow...")
        client = MlflowClient(tracking_uri=TRACKING_URI)
        print("✓ Connected to MLflow\n")
        
        # Get all experiments
        print("=" * 70)
        print("Fetching ALL Experiments")
        print("=" * 70)
        
        # Try with view_type=ALL first
        try:
            experiments = client.search_experiments(max_results=1000, view_type="ALL")
            print(f"✓ Found {len(experiments)} experiment(s) with view_type=ALL\n")
        except Exception as e:
            print(f"view_type=ALL not supported: {e}")
            experiments = client.search_experiments(max_results=1000)
            print(f"✓ Found {len(experiments)} experiment(s) with default view\n")
        
        if not experiments:
            print("✗ No experiments found!")
            sys.exit(1)
        
        # Display all experiments
        for i, exp in enumerate(experiments, 1):
            print(f"\n{i}. {exp.name}")
            print(f"   ID: {exp.experiment_id}")
            print(f"   Lifecycle Stage: {exp.lifecycle_stage}")
            print(f"   Artifact Location: {exp.artifact_location}")
            
            # Get runs for this experiment
            try:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1000
                )
                print(f"   Runs: {len(runs)}")
                
                if runs:
                    print(f"   Recent runs:")
                    for run in runs[:5]:
                        run_name = run.data.tags.get('mlflow.runName', 'N/A')
                        status = run.info.status
                        accuracy = run.data.metrics.get('accuracy', 'N/A')
                        f1 = run.data.metrics.get('f1_weighted', 'N/A')
                        print(f"     - {run.info.run_id[:8]}... | {run_name} | Status: {status} | Acc: {accuracy} | F1: {f1}")
                    if len(runs) > 5:
                        print(f"     ... and {len(runs) - 5} more runs")
            except Exception as e:
                print(f"   ✗ Error getting runs: {e}")
        
        # Check for specific experiment
        print("\n" + "=" * 70)
        print("Checking for 'rakuten-multimodal-text-image' experiment")
        print("=" * 70)
        
        try:
            exp_by_name = client.get_experiment_by_name("rakuten-multimodal-text-image")
            if exp_by_name:
                print(f"✓ Found experiment by name: {exp_by_name.name} (ID: {exp_by_name.experiment_id})")
                
                # Get all runs for this experiment
                runs = client.search_runs(
                    experiment_ids=[exp_by_name.experiment_id],
                    max_results=1000
                )
                print(f"  Total runs: {len(runs)}")
                
                if runs:
                    print(f"\n  All runs:")
                    for i, run in enumerate(runs, 1):
                        run_name = run.data.tags.get('mlflow.runName', 'N/A')
                        status = run.info.status
                        accuracy = run.data.metrics.get('accuracy', 'N/A')
                        f1 = run.data.metrics.get('f1_weighted', 'N/A')
                        print(f"    {i}. {run.info.run_id[:8]}... | {run_name} | Status: {status} | Acc: {accuracy} | F1: {f1}")
            else:
                print("✗ Experiment not found by name")
        except Exception as e:
            print(f"✗ Error getting experiment by name: {e}")
        
        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total experiments: {len(experiments)}")
        exp_names = [exp.name for exp in experiments]
        print(f"Experiment names: {exp_names}")
        
        # Check if Default is the only one
        if len(experiments) == 1 and experiments[0].name == "Default":
            print("\n⚠️  WARNING: Only 'Default' experiment found!")
            print("   This suggests:")
            print("   1. Host header might not be routing correctly")
            print("   2. Connecting to wrong MLflow instance")
            print("   3. Experiments might be in a different lifecycle stage")
        else:
            print("\n✓ Multiple experiments found - connection is working correctly")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

