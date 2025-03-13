# House Price Prediction API Usage Guide

This guide provides instructions on how to use the House Price Prediction API.

## API Overview

The API is built using FastAPI and provides predictions for house prices based on the California Housing dataset. It exposes the following endpoints:

- `GET /health`: Health check endpoint
- `POST /api/predict`: Prediction endpoint
- `GET /api/sample-input`: Sample input format for prediction

## Getting Started

### Running the API Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the API:
   ```bash
   python api\App.py
   ```

4. The API will be available at: `http://localhost:8000`

### Using Docker

If you prefer to use Docker:

1. Build the Docker image:
   ```bash
   docker build -t house-price-prediction .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 house-price-prediction
   ```

## API Documentation

Once the API is running, you can access the auto-generated documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Get Sample Input

```
GET /api/sample-input
```

**Response:**
```json
{
  "sample_input": {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23,
    "model_type": "random_forest"
  },
  "usage_instructions": "Send a POST request to /api/predict with form data containing these fields"
}
```

### Predict House Price

```
POST /api/predict
```

**Form Data Parameters:**
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

**Response:**
```json
{
  "prediction": {
    "price_numeric": 4.526,
    "price_usd": 452600.0,
    "price_formatted": "$452,600.00",
    "unit": "USD",
    "description": "Predicted median house value is $452,600.00"
  },
  "model_info": {
    "model_type": "xgboost",
    "model_class": "XGBRegressor",
    "top_influential_features": [
      {
        "feature": "MedInc",
        "importance": 0.3,
        "value": 8.3252,
        "importance_percentage": "30.0%"
      },
      {
        "feature": "HouseAge",
        "importance": 0.2,
        "value": 41.0,
        "importance_percentage": "20.0%"
      },
      {
        "feature": "Latitude",
        "importance": 0.1,
        "value": 37.88,
        "importance_percentage": "10.0%"
      }
    ],
    "confidence_metrics": {
      "model_confidence": "high"
    }
  },
  "input_features": {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  },
  "timestamp": "XGBRegressor(...)"
}
```

Note: The `price_numeric` is in units of $100,000 (e.g., 4.526 represents $452,600).

## Example Usage

### Using cURL

**Health Check:**
```bash
curl -X GET http://localhost:8000/health
```

**Get Sample Input:**
```bash
curl -X GET http://localhost:8000/api/sample-input
```

**Prediction:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "MedInc=8.3252" \
  -F "HouseAge=41.0" \
  -F "AveRooms=6.984" \
  -F "AveBedrms=1.023" \
  -F "Population=322.0" \
  -F "AveOccup=2.555" \
  -F "Latitude=37.88" \
  -F "Longitude=-122.23"
```

### Using Python (with requests library)

```python
import requests

# API endpoint
url = "http://localhost:8000/api/predict"

# Input data as form data
data = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
}

# Make POST request
response = requests.post(url, data=data)

# Print response
print(response.status_code)
print(response.json())
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200`: Successful request
- `400`: Bad request (invalid input)
- `422`: Validation error (missing or invalid data)
- `500`: Server error

Error Response Example:
```json
{
  "error": "Prediction failed",
  "details": "Error message"
}
```

## Input Validation

The API performs validation on input data:
- All features are required
- Numerical values must be within reasonable ranges
- Average bedrooms must be less than average rooms

## Support

If you encounter any issues or have questions, please:
1. Check the API documentation
2. Submit an issue on GitHub
3. Contact the development team
