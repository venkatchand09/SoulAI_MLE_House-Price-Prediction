import numpy as np
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List

from api.schemas import HouseFeatures, PredictionResponse, ErrorResponse

# Create router
router = APIRouter()

# Get logger
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_price(request: Request, features: HouseFeatures):
    """
    Predict house price based on input features.
    
    Args:
        features: House features input
        
    Returns:
        PredictionResponse: Predicted house price and input features
    """
    try:
        # Log prediction request
        logger.info(f"Prediction request received: {features.dict()}")
        
        # Access model and scaler from app state
        model = request.app.state.model
        scaler = request.app.state.scaler
        
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or scaler not loaded")
        
        # Convert input data to NumPy array
        input_data = np.array([[
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]])
        
        # Scale input data if scaler is available and working
        try:
            input_scaled = scaler.transform(input_data)
        except Exception as e:
            logger.warning(f"Error scaling input data: {str(e)}. Using unscaled data.")
            input_scaled = input_data
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Log prediction result
        logger.info(f"Prediction result: {prediction[0]}")
        
        # Return prediction and input features
        return PredictionResponse(
            predicted_price=float(prediction[0]),
            features=features.dict()
        )
    
    except Exception as e:
        # Log error
        logger.error(f"Prediction error: {str(e)}")
        
        # Raise HTTP exception
        raise HTTPException(
            status_code=400,
            detail={"error": "Prediction failed", "details": str(e)}
        )


@router.get("/feature-importance", responses={500: {"model": ErrorResponse}})
async def get_feature_importance(request: Request):
    """
    Get feature importance from the model.
    
    Returns:
        dict: Feature names and their importance values
    """
    try:
        # Access model from app state
        model = request.app.state.model
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Get feature names
        feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        # Check if model has feature_importance_ attribute
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            
            # Create list of features and their importance
            feature_importance = [
                {"feature": name, "importance": float(importance)}
                for name, importance in zip(feature_names, importances)
            ]
            
            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            
            return {"feature_importance": feature_importance}
        else:
            raise HTTPException(
                status_code=500,
                detail="Model does not support feature importance"
            )
    
    except Exception as e:
        # Log error
        logger.error(f"Feature importance error: {str(e)}")
        
        # Raise HTTP exception
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get feature importance", "details": str(e)}
        )