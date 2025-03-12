import numpy as np
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, Form
from typing import Dict, Any

from api.schemas import HouseFeatures, ErrorResponse, ModelType

router = APIRouter()

logger = logging.getLogger(_name_)

async def get_house_features_basic(
    MedInc: float = Form(..., description="Median income in block group"),
    HouseAge: float = Form(..., description="Median house age in block group"),
    AveRooms: float = Form(..., description="Average number of rooms per household"),
    AveBedrms: float = Form(..., description="Average number of bedrooms per household"),
    Population: float = Form(..., description="Block group population"),
    AveOccup: float = Form(..., description="Average number of household members"),
    Latitude: float = Form(..., description="Block group latitude"),
    Longitude: float = Form(..., description="Block group longitude")
):
    """
    Parse house features from form data without model selection
    """
    return HouseFeatures(
        MedInc=MedInc,
        HouseAge=HouseAge,
        AveRooms=AveRooms,
        AveBedrms=AveBedrms,
        Population=Population,
        AveOccup=AveOccup,
        Latitude=Latitude,
        Longitude=Longitude,
        model_type=ModelType.XGBOOST 
    )

@router.post("/api/predict", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}})
async def predict_price(request: Request, features: HouseFeatures = Depends(get_house_features_basic)):
    """
    Predict house price based on input features using the XGBoost model.
    
    Form fields:
    - MedInc: Median income in block group
    - HouseAge: Median house age in block group
    - AveRooms: Average number of rooms per household
    - AveBedrms: Average number of bedrooms per household
    - Population: Block group population
    - AveOccup: Average number of household members
    - Latitude: Block group latitude
    - Longitude: Block group longitude
    """
    try:
        logger.info(f"Prediction request received: {features.model_dump()}")
        
        models = request.app.state.models
        scaler = request.app.state.scaler
        
        if not models or scaler is None:
            raise HTTPException(status_code=500, detail="Models or scaler not loaded")
        
        model_type = ModelType.XGBOOST
        if model_type not in models:
            raise HTTPException(status_code=500, detail="XGBoost model not available")
        
        model = models[model_type]
        
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
        
        try:
            input_scaled = scaler.transform(input_data)
        except Exception as e:
            logger.warning(f"Error scaling input data: {str(e)}. Using unscaled data.")
            input_scaled = input_data
        
        prediction = model.predict(input_scaled)
        predicted_price = float(prediction[0])
        
        logger.info(f"Prediction result using XGBoost: {predicted_price}")
        
        feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        importances = model.feature_importances_
        
        feature_importance = [
            {
                "feature": name, 
                "importance": float(importance), 
                "value": getattr(features, name),
                "importance_percentage": f"{float(importance) * 100:.1f}%"
            }
            for name, importance in zip(feature_names, importances)
        ]
        
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        top_features = feature_importance[:3]
        
        price_usd = predicted_price * 100000  
        
        response = {
            "prediction": {
                "price_numeric": predicted_price,
                "price_usd": price_usd,
                "price_formatted": f"${price_usd:,.2f}",
                "unit": "USD",
                "description": f"Predicted median house value is ${price_usd:,.2f}",
            },
            "model_info": {
                "model_type": "xgboost",
                "model_class": model._class.name_,
                "top_influential_features": top_features,
                "confidence_metrics": {
                    "model_confidence": "high"
                }
            },
            "input_features": features.model_dump(exclude={"model_type"}),
            "timestamp": str(model)
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        
        raise HTTPException(
            status_code=400,
            detail={"error": "Prediction failed", "details": str(e)}
        )


@router.get("/api/sample-input", response_model=Dict[str, Any])
async def get_sample_input():
    """
    Get a sample input format for the prediction endpoint.
    """
    sample = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.023,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23,
        "model_type": "random_forest"
    }
    
    return {
        "sample_input": sample,
        "usage_instructions": "Send a POST request to /api/predict with form data containing these fields",
    }


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
