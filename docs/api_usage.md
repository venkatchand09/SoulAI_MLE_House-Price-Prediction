# House Price Prediction API Usage Guide

This guide provides instructions on how to use the House Price Prediction API.

## API Overview

The API is built using FastAPI and provides predictions for house prices based on the California Housing dataset. It exposes the following endpoints:

- `GET /health`: Health check endpoint
- `POST /api/predict`: Prediction endpoint
- `GET /api/feature-importance`: Feature importance endpoint

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
   python api/app.py
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

### Predict House Price

```
POST /api/predict
```

**Request Body:**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322.0,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Response:**
```json
{
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
```

Note: The `predicted_price` is in units of $100,000 (e.g., 4.526 represents $452,600).

### Feature Importance

```
GET /api/feature-importance
```

**Response:**
```json
{
  "feature_importance": [
    {
      "feature": "MedInc",
      "importance": 0.322
    },
    {
      "feature": "Latitude",
      "importance": 0.153
    },
    {
      "feature": "Longitude",
      "importance": 0.142
    },
    {
      "feature": "AveRooms",
      "importance": 0.121
    },
    {
      "feature": "HouseAge",
      "importance": 0.118
    },
    {
      "feature": "AveBedrms",
      "importance": 0.094
    },
    {
      "feature": "Population",
      "importance": 0.029
    },
    {
      "feature": "AveOccup",
      "importance": 0.021
    }
  ]
}
```

## Example Usage

### Using cURL

**Health Check:**
```bash
curl -X GET http://localhost:8000/health
```

**Prediction:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

**Feature Importance:**
```bash
curl -X GET http://localhost:8000/api/feature-importance
```

### Using Python (with requests library)

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/predict"

# Input data
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
response = requests.post(url, json=data)

# Print response
print(response.status_code)
print(json.dumps(response.json(), indent=2))
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

## Rate Limiting

The API has rate limiting enabled to prevent abuse:
- 100 requests per minute per IP address

## Support

If you encounter any issues or have questions, please:
1. Check the API documentation
2. Submit an issue on GitHub
3. Contact the development team