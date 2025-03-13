#!/bin/bash
set -e

mkdir -p logs models data/raw data/processed

if [ ! -f models/xgboost_model.joblib ]; then
    echo "Model files not found. Running data preprocessing and model training..."
    
    echo "Starting data preprocessing..."
    python src/data_preprocessing.py
    
    echo "Starting model training..."
    python src/model_training.py
    
    echo "Model training completed."
else
    echo "Model files found. Skipping preprocessing and training."
fi

echo "Starting API server..."
exec uvicorn api.App:app --host 0.0.0.0 --port 8000
