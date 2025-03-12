from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import field_validator

class ModelType(str, Enum):
    """Enum for available model types."""
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class HouseFeatures(BaseModel):
    """
    Schema for house features input.
    
    Attributes:
        MedInc (float): Median income in block group
        HouseAge (float): Median house age in block group
        AveRooms (float): Average number of rooms per household
        AveBedrms (float): Average number of bedrooms per household
        Population (float): Population in block group
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
    model_type: Optional[ModelType] = Field(ModelType.RANDOM_FOREST, description="Model type to use for prediction")
    
    @field_validator('AveBedrms')
    def bedrooms_less_than_rooms(cls, v, info):
        if 'AveRooms' in info.data and v > info.data['AveRooms']:
            raise ValueError('Average bedrooms cannot exceed average rooms')
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
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
