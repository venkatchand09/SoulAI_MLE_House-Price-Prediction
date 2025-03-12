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

### Data Cleaning and Feature Engineering

The following preprocessing steps were performed:
1. **Missing Values**: Confirmed no missing values in the dataset
2. **Feature Scaling**: Applied StandardScaler to normalize all features
3. **Feature Relationships**: Visualized correlations between features and the target variable

The correlation heatmap showed strong positive correlation between median income (MedInc) and house value (MedHouseVal), indicating that income is a significant predictor of house prices.

## Model Training & Evaluation

### Baseline Models

Four regression models were trained and evaluated:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost Regressor

The models were evaluated using the following metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

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

## Model Deployment

### API Development

The optimized model was deployed as a REST API using FastAPI. The API includes:
1. A prediction endpoint that accepts house features and returns the predicted price
2. A feature importance endpoint that provides insights into model decision-making
3. Health checks and error handling

### Testing

The API was thoroughly tested using:
- Unit tests for model loading and predictions
- Integration tests for API endpoints
- Performance tests to ensure the API can handle multiple concurrent requests

## Conclusion

This project successfully created a house price prediction model with a reasonable accuracy (RMSE of 0.531, which corresponds to approximately $53,100 in prediction error). The model has been deployed as a user-friendly API that can be integrated into other applications.

### Insights

1. Median income is the strongest predictor of house prices
2. Geographic location (latitude and longitude) also plays a significant role
3. The number of rooms and house age have moderate predictive power

### Future Improvements

1. Include more features like school quality, crime rates, and proximity to amenities
2. Explore more advanced models like neural networks or stacked ensembles
3. Add more robust data validation and preprocessing
4. Implement continuous model retraining as new data becomes available

## References

- California Housing dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- FastAPI documentation: https://fastapi.tiangolo.com/
- scikit-learn documentation: https://scikit-learn.org/stable/