#!/usr/bin/env python3
"""
Diagnostic script to check MLflow experiments and configuration.
This helps identify why experiments might not be found.
"""
import os
import sys
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def main():
    # Get tracking URI from environment or use default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow_host = os.getenv("MLFLOW_HOST", "mlflow.rakuten.dev")
    
    print("=" * 70)
    print("MLflow Experiments Diagnostic Tool")
    print("=" * 70)
    print(f"\nTracking URI: {tracking_uri}")
    print(f"Environment variable MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}")
    print()
    
    # Configure host-based routing for AWS ALB (same fix as MLflowManager)
    if mlflow_host:
        try:
            import requests
            from functools import wraps
            
            # Store original request method
            original_request = requests.Session.request
            tracking_uri_base = tracking_uri.rstrip("/")
            
            @wraps(original_request)
            def request_with_host_header(session_self, method, url, *args, **kwargs):
                # If the URL matches our tracking URI, add Host header for host-based routing
                if tracking_uri_base and url.startswith(tracking_uri_base):
                    if 'headers' not in kwargs:
                        kwargs['headers'] = {}
                    kwargs['headers']['Host'] = mlflow_host
                return original_request(session_self, method, url, *args, **kwargs)
            
            # Patch the Session class to add Host header
            requests.Session.request = request_with_host_header
            print(f"Configured host-based routing with Host header: {mlflow_host}\n")
        except Exception as e:
            print(f"Warning: Failed to configure Host header for host-based routing: {e}\n")
    
    try:
        # Initialize client
        print("Connecting to MLflow...")
        client = MlflowClient(tracking_uri=tracking_uri)
        
        # List all experiments
        print("\n" + "=" * 70)
        print("Available Experiments:")
        print("=" * 70)
        
        try:
            experiments = client.search_experiments(max_results=1000)
            
            if not experiments:
                print("  ❌ No experiments found in this MLflow instance.")
                print("\n  Possible reasons:")
                print("    1. This is a new MLflow instance with no experiments yet")
                print("    2. The tracking URI points to a different backend store")
                print("    3. The MLflow server is not properly configured")
            else:
                print(f"  ✓ Found {len(experiments)} experiment(s):\n")
                
                for i, exp in enumerate(experiments, 1):
                    # Get run count for this experiment
                    try:
                        runs = client.search_runs(
                            experiment_ids=[exp.experiment_id],
                            max_results=1
                        )
                        run_count = len(client.search_runs(
                            experiment_ids=[exp.experiment_id],
                            max_results=10000
                        ))
                    except:
                        run_count = "?"
                    
                    print(f"  {i}. {exp.name}")
                    print(f"     ID: {exp.experiment_id}")
                    print(f"     Runs: {run_count}")
                    print(f"     Artifact Location: {exp.artifact_location}")
                    print(f"     Lifecycle Stage: {exp.lifecycle_stage}")
                    print()
                
                # Check for the expected experiment name
                expected_name = "rakuten-multimodal-text-image"
                found_expected = any(exp.name == expected_name for exp in experiments)
                
                print("=" * 70)
                print("Expected Experiment Check:")
                print("=" * 70)
                if found_expected:
                    print(f"  ✓ Found expected experiment: '{expected_name}'")
                else:
                    print(f"  ❌ Expected experiment '{expected_name}' NOT FOUND")
                    print(f"\n  Available experiment names:")
                    for exp in experiments:
                        print(f"    - {exp.name}")
                    print(f"\n  💡 Tip: If your experiments are in a different MLflow instance,")
                    print(f"     check the MLFLOW_TRACKING_URI environment variable.")
                    print(f"     Current value: {tracking_uri}")
        
        except MlflowException as e:
            print(f"  ❌ Error listing experiments: {e}")
            print(f"\n  This could indicate:")
            print(f"    1. Connection issue to MLflow server")
            print(f"    2. Authentication problem")
            print(f"    3. Backend store access issue")
        
        # Check registered models
        print("\n" + "=" * 70)
        print("Registered Models:")
        print("=" * 70)
        
        try:
            models = client.search_registered_models(max_results=100)
            if not models:
                print("  No registered models found.")
            else:
                print(f"  ✓ Found {len(models)} registered model(s):\n")
                for model in models:
                    versions = client.search_model_versions(f"name='{model.name}'")
                    print(f"  - {model.name} ({len(versions)} versions)")
        except MlflowException as e:
            print(f"  ❌ Error listing models: {e}")
        
        print("\n" + "=" * 70)
        print("Diagnostic Complete")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Failed to connect to MLflow at {tracking_uri}")
        print(f"   Error: {e}")
        print(f"\n  Troubleshooting steps:")
        print(f"    1. Verify MLflow server is running")
        print(f"    2. Check the MLFLOW_TRACKING_URI environment variable")
        print(f"    3. Test connectivity: curl {tracking_uri}/health")
        print(f"    4. Check firewall/network settings")
        sys.exit(1)

if __name__ == "__main__":
    main()

