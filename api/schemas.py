from pydantic import BaseModel, Field, validator
from typing import Dict, Any


class HouseFeatures(BaseModel):
    """
    Schema for house features input.
    
    Attributes:
        MedInc (float): Median income in block group
        HouseAge (float): Median house age in block group
        AveRooms (float): Average number of rooms per household
        AveBedrms (float): Average number of bedrooms per household
        Population (float): Block group population
        AveOccup (float): Average number of household members
        Latitude (float): Block group latitude
        Longitude (float): Block group longitude
    """
    MedInc: float = Field(..., description="Median income in block group", gt=0)
    HouseAge: float = Field(..., description="Median house age in block group", ge=0)
    AveRooms: float = Field(..., description="Average number of rooms per household", gt=0)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", ge=0)
    Population: float = Field(..., description="Block group population", ge=0)
    AveOccup: float = Field(..., description="Average number of household members", gt=0)
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")
    
    @validator('AveBedrms')
    def bedrooms_less_than_rooms(cls, v, values):
        if 'AveRooms' in values and v > values['AveRooms']:
            raise ValueError('Average bedrooms cannot exceed average rooms')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.023,
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    
    Attributes:
        predicted_price (float): Predicted median house value (in $100,000s)
        features (dict): Input features used for prediction
    """
    predicted_price: float = Field(..., description="Predicted median house value (in $100,000s)")
    features: Dict[str, Any] = Field(..., description="Input features used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 4.526,
                "features": {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.984,
                    "AveBedrms": 1.023,
                    "Population": 322.0,
                    "AveOccup": 2.555,
                    "Latitude": 37.88,
                    "Longitude": -122.23
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Schema for error response.
    
    Attributes:
        error (str): Error message
        details (dict, optional): Additional error details
    """
    error: str
    details: Dict[str, Any] = None