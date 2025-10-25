"""
Basic tests to verify the project structure and imports
"""
import pytest


def test_imports_serve():
    """Test that the FastAPI app module can be imported"""
    try:
        import src.serve.fastapi_app
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import FastAPI app: {e}")


def test_imports_models():
    """Test that model modules can be imported"""
    try:
        import src.models.train_model
        import src.models.predict_model
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import model modules: {e}")


def test_imports_data():
    """Test that data modules can be imported"""
    try:
        import src.data.make_dataset
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import data modules: {e}")


def test_imports_features():
    """Test that feature modules can be imported"""
    try:
        import src.features.build_features
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import feature modules: {e}")


def test_requirements_installed():
    """Test that key dependencies are installed"""
    try:
        import fastapi
        import mlflow
        import pandas
        import sklearn
        import pydantic
        assert True
    except ImportError as e:
        pytest.fail(f"Missing required dependency: {e}")


def test_python_version():
    """Test that Python version is 3.11+"""
    import sys
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"

