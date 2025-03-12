import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from src
from src.utils import load_model, load_scaler

# Paths to model and scaler
MODEL_PATH = "models/random_forest_model.joblib"
SCALER_PATH = "models/scaler.joblib"

@pytest.fixture
def california_data():
    """Load California Housing dataset for testing."""
    california = fetch_california_housing()
    X = california.data
    y = california.target
    feature_names = california.feature_names
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["MedHouseVal"] = y
    
    return df

def test_model_loading():
    """Test that the model loads correctly."""
    model = load_model(MODEL_PATH)
    assert model is not None
    assert hasattr(model, 'predict')

def test_scaler_loading():
    """Test that the scaler loads correctly."""
    scaler = load_scaler(SCALER_PATH)
    assert scaler is not None
    assert hasattr(scaler, 'transform')

def test_prediction_shape(california_data):
    """Test that the model predictions have the expected shape."""
    # Load model and scaler
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    
    # Get a small subset of data
    X = california_data.drop(columns=["MedHouseVal"]).values[:10]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Check shape
    assert predictions.shape == (10,)

def test_prediction_range(california_data):
    """Test that the model predictions are within an expected range."""
    # Load model and scaler
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    
    # Get a sample of data
    X = california_data.drop(columns=["MedHouseVal"]).values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # California housing prices should be within a reasonable range
    # The target is in units of $100,000s
    assert np.all(predictions >= 0)  # Non-negative prices
    assert np.all(predictions <= 10)  # Upper bound of $1M (in $100,000s)

def test_model_performance(california_data):
    """Test that the model performance is above a minimum threshold."""
    # Load model and scaler
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    
    # Get features and target
    X = california_data.drop(columns=["MedHouseVal"]).values
    y = california_data["MedHouseVal"].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    
    # Performance thresholds
    assert rmse < 1.0  # RMSE less than 1.0 (in $100,000s)
    assert r2 > 0.6    # R² greater than 0.6