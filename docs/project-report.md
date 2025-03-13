# House Price Prediction Project Report

## Introduction

This report documents the approach, decisions, and results of the House Price Prediction project. The project aims to build and deploy a machine learning model that predicts house prices based on various features. The California Housing dataset from scikit-learn was used for this project.

## Data Preprocessing

### Dataset Overview

The California Housing dataset contains 20,640 samples with 8 features:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude

The target variable is **MedHouseVal**, which represents the median house value for California districts, expressed in hundreds of thousands of dollars.

### Exploratory Data Analysis (EDA)

Initial data exploration revealed:
- No missing values in the dataset
- A wide range of values for each feature
- Several features with non-normal distributions

Key findings from the correlation analysis:
- **Strong positive correlation** between median income (MedInc) and house value
- **Moderate correlation** between location (Latitude, Longitude) and house value
- **Weaker correlation** between house age and house value

### Data Cleaning and Feature Engineering

The following preprocessing steps were performed:
1. **Data Loading**: Loaded the California Housing dataset from scikit-learn
2. **Missing Values Check**: Confirmed no missing values in the dataset
3. **Feature Scaling**: Applied StandardScaler to normalize all features
4. **Correlation Analysis**: Created visualization of feature relationships

## Model Training & Evaluation

### Baseline Models

Four regression models were trained and evaluated:
1. **Linear Regression**: Simple linear model as a baseline
2. **Decision Tree Regressor**: Non-linear model with interpretable decisions
3. **Random Forest Regressor**: Ensemble of decision trees
4. **XGBoost Regressor**: Gradient boosted trees implementation

The models were evaluated using the following metrics:
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of errors
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **R² Score**: Proportion of variance explained by the model

### Model Optimization

Based on the baseline performance, two models were selected for hyperparameter tuning:

1. **Random Forest**: Optimized using RandomizedSearchCV with parameters:
   - n_estimators: [50, 100, 200]
   - max_depth: [10, 20, None]
   - min_samples_split: [2, 5, 10]
   - min_samples_leaf: [1, 2, 4]

2. **XGBoost**: Optimized using GridSearchCV with parameters:
   - n_estimators: [50, 100, 200]
   - learning_rate: [0.01, 0.1, 0.2]
   - max_depth: [3, 6, 10]
   - subsample: [0.7, 0.8, 1.0]

### Results

The following table shows the performance of the optimized models:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Optimized XGBoost | 0.531 | 0.379 | 0.812 |
| Optimized Random Forest | 0.549 | 0.393 | 0.801 |

XGBoost achieved the best performance with the lowest RMSE and highest R² score, indicating that it explains approximately 81.2% of the variance in house prices.

Feature importance analysis from the XGBoost model revealed:
1. Median income (MedInc) is the most important feature
2. Geographical location (Latitude, Longitude) is also significant
3. Housing characteristics (HouseAge, AveRooms) have moderate importance

## Model Deployment

### API Development

The optimized XGBoost model was deployed as a REST API using FastAPI. The API includes:

1. **API Structure**:
   - `app.py`: Main application with model loading logic
   - `endpoints.py`: API endpoint definitions
   - `schemas.py`: Data validation schemas using Pydantic

2. **Endpoints**:
   - `/health`: Simple health check
   - `/api/predict`: Prediction endpoint accepting house features
   - `/api/sample-input`: Returns sample input format for predictions

3. **Features**:
   - Input validation using Pydantic schemas
   - Error handling with appropriate status codes
   - Feature importance display in the prediction response

### Containerization

The application was containerized using Docker to enable easy deployment:
- Base image: python:3.9-slim
- All dependencies installed from requirements.txt
- Application exposed on port 8000

### Testing and Performance

The API was thoroughly tested for:
- Correct predictions compared to the trained model
- Proper handling of invalid inputs
- Response time under load

## Conclusion

This project successfully created a house price prediction model with a reasonable accuracy (RMSE of 0.531, which corresponds to approximately $53,100 in prediction error). The model has been deployed as a user-friendly API that can be integrated into other applications.

### Insights

1. Median income is the strongest predictor of house prices
2. Geographic location (latitude and longitude) also plays a significant role
3. The number of rooms and house age have moderate predictive power
4. XGBoost outperformed other models, including Random Forest

### Future Improvements

1. **Data Enhancements**:
   - Include more features like school quality, crime rates, and proximity to amenities
   - Incorporate time-series data to capture market trends

2. **Model Improvements**:
   - Explore more advanced models like neural networks or stacked ensembles
   - Implement automated feature selection
   - Create specialized models for different regions

3. **Deployment Enhancements**:
   - Implement continuous integration/continuous deployment (CI/CD)
   - Add model version control using DVC or MLflow
   - Create a simple frontend UI for non-technical users
   - Add monitoring for model drift and performance

4. **Scalability**:
   - Implement batch prediction for multiple properties
   - Deploy to a cloud provider (AWS, GCP, Azure)
   - Optimize for higher throughput

## References

- California Housing dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- FastAPI documentation: https://fastapi.tiangolo.com/
- scikit-learn documentation: https://scikit-learn.org/stable/
- XGBoost documentation: https://xgboost.readthedocs.io/
