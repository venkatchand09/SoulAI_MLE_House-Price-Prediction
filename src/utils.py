import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler


def setup_logging(log_file='logs/house_price_prediction.log'):
    """Set up logging configuration."""
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(_name_)


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'docs',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logger = logging.getLogger(_name_)
    logger.info("Created project directories")


def load_model(model_path='models/random_forest_model.joblib'):
    """Load a trained model from disk."""
    logger = logging.getLogger(_name_)
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_scaler(scaler_path='models/scaler.joblib'):
    """Load a fitted scaler from disk."""
    logger = logging.getLogger(_name_)
    
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        raise


def predict_house_price(features, model_path='models/random_forest_model.joblib', scaler_path='models/scaler.joblib'):
    """Predict house price based on input features."""
    logger = logging.getLogger(_name_)
    
    try:
        # Load model and scaler
        model = load_model(model_path)
        scaler = load_scaler(scaler_path)
        
        # Convert features to numpy array if needed
        if isinstance(features, dict):
            # Ensure correct order based on original feature names
            feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
            features = np.array([[features[name] for name in feature_names]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        logger.info(f"Prediction made: {prediction[0]}")
        
        return prediction[0]
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def get_feature_importance(model_path='models/random_forest_model.joblib'):
    """Get feature importance from a trained model."""
    logger = logging.getLogger(_name_)
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Check if model has feature_importance_ attribute
        if hasattr(model, 'feature_importances_'):
            # Get feature names (assuming California Housing dataset)
            feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
            
            # Create DataFrame with feature importances
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            logger.info("Feature importance extracted")
            
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise


def plot_feature_importance(importance_df, save_path='docs/feature_importance.png'):
    """Plot feature importance."""
    if importance_df is None:
        return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger = logging.getLogger(_name_)
    logger.info(f"Feature importance plot saved to {save_path}")


if _name_ == "_main_":
    # Setup logging
    logger = setup_logging()
    
    # Create directories
    create_directories()
    
    # Example of using utility functions
    try:
        model = load_model()
        importance_df = get_feature_importance()
        plot_feature_importance(importance_df)
        logger.info("Utility functions tested successfully")
    except Exception as e:
        logger.error(f"Error testing utility functions: {str(e)}")
