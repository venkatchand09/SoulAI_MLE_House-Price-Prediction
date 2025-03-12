import os
import sys
import pytest
from fastapi.testclient import TestClient
import numpy as np
import joblib
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app
from api.app import app

# Create mock model and scaler
mock_model = MagicMock()
mock_model.predict.return_value = np.array([4.526])
mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

mock_scaler = MagicMock()
mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

# Set up app state with mocks
app.state.model = mock_model
app.state.scaler = mock_scaler

# Create test client
client = TestClient(app)

# Test data (features from California Housing dataset)
test_data = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    """Test the prediction endpoint with valid data."""
    response = client.post("/api/predict", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "predicted_price" in data
    assert "features" in data
    
    # Check that predicted price is a reasonable value (between 0 and 10)
    assert 0 <= data["predicted_price"] <= 10
    
    # Check that features match input data
    for key, value in test_data.items():
        assert data["features"][key] == value

def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid data."""
    # Test with negative values
    invalid_data = test_data.copy()
    invalid_data["MedInc"] = -5.0
    
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error
    
    # Test with missing values
    incomplete_data = {k: v for k, v in test_data.items() if k != "HouseAge"}
    
    response = client.post("/api/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error

def test_feature_importance_endpoint():
    """Test the feature importance endpoint."""
    response = client.get("/api/feature-importance")
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "feature_importance" in data
    
    # Check that feature importance is a list of dictionaries
    feature_importance = data["feature_importance"]
    assert isinstance(feature_importance, list)
    assert len(feature_importance) > 0
    
    # Check that each item has feature and importance
    for item in feature_importance:
        assert "feature" in item
        assert "importance" in item
        assert isinstance(item["importance"], float)