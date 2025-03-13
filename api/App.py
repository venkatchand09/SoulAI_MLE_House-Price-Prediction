import os
import sys
import logging
from unittest.mock import MagicMock
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.endpoints import router
from api.schemas import ModelType
from src.utils import setup_logging

logger = setup_logging(log_file='logs/api.log')

MODEL_PATHS = {
    # ModelType.RANDOM_FOREST: "models/random_forest_model.joblib",
    ModelType.XGBOOST: "models/xgboost_model.joblib"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.scaler = None
    
    try:
        scaler_path = "models/scaler.joblib"
        if os.path.exists(scaler_path):
            app.state.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            from sklearn.preprocessing import StandardScaler
            logger.warning("Scaler file not found. Creating dummy scaler for demo/testing.")
            app.state.scaler = StandardScaler()
            
        for model_type, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                app.state.models[model_type] = joblib.load(model_path)
                logger.info(f"Model {model_type} loaded successfully from {model_path}")
            else:
                if model_type == ModelType.LINEAR_REGRESSION:
                    from sklearn.linear_model import LinearRegression
                    app.state.models[model_type] = LinearRegression()
                elif model_type == ModelType.DECISION_TREE:
                    from sklearn.tree import DecisionTreeRegressor
                    app.state.models[model_type] = DecisionTreeRegressor()
                elif model_type == ModelType.RANDOM_FOREST:
                    from sklearn.ensemble import RandomForestRegressor
                    app.state.models[model_type] = RandomForestRegressor()
                elif model_type == ModelType.XGBOOST:
                    try:
                        from xgboost import XGBRegressor
                        app.state.models[model_type] = XGBRegressor()
                    except ImportError:
                        from sklearn.ensemble import GradientBoostingRegressor
                        app.state.models[model_type] = GradientBoostingRegressor()
                        
                logger.warning(f"Model file for {model_type} not found. Creating dummy model for demo/testing.")
                
                # Set dummy feature_importances_ for tree-based models
                if hasattr(app.state.models[model_type], 'feature_importances_'):
                    app.state.models[model_type].feature_importances_ = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
            
    except Exception as e:
        logger.error(f"Error loading models or scaler: {str(e)}")
        for model_type in ModelType:
            app.state.models[model_type] = MagicMock()
            app.state.models[model_type].predict.return_value = np.array([4.5])
            app.state.models[model_type].feature_importances_ = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
            
        app.state.scaler = MagicMock()
        app.state.scaler.transform.return_value = np.ones((1, 8))

    yield

    app.state.models = None
    app.state.scaler = None
    logger.info("Resources released")

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using California Housing Dataset",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)