# House Price Prediction

A machine learning project for predicting house prices using the California Housing dataset.

## Project Overview

This project demonstrates the end-to-end process of building and deploying a machine learning model:

1. Data preprocessing and exploratory data analysis
2. Model training and evaluation
3. Model optimization using hyperparameter tuning
4. API development with FastAPI
5. Testing and documentation

## Repository Structure

```
house-price-prediction/
├── data/               # Data related files
│   ├── raw/            # Raw, immutable data
│   └── processed/      # Processed data ready for modeling
├── models/             # Saved model files
├── notebooks/          # Jupyter notebooks
│   └── SouAI.ipynb     # Main development notebook
├── src/                # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── api/                # API related code
│   ├── app.py          # FastAPI application
│   ├── schemas.py      # Pydantic schemas
│   └── endpoints.py    # API endpoint handlers
├── tests/              # Test files
│   ├── test_api.py
│   └── test_model.py
├── docs/               # Documentation
│   ├── report.md       # Project report
│   └── api_usage.md    # API usage documentation
├── Dockerfile          # Dockerfile for containerization
├── requirements.txt    # Project dependencies
├── setup.py            # Installation script
└── README.md           # Project README
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Data Preprocessing

```bash
python src/data_preprocessing.py
```

This will:
- Load the California Housing dataset
- Perform exploratory data analysis
- Handle missing values
- Scale features
- Create visualizations
- Save preprocessed data

#### Model Training

```bash
python src/model_training.py
```

This will:
- Load preprocessed data
- Train baseline models (Linear Regression, Decision Tree, Random Forest, XGBoost)
- Optimize models using hyperparameter tuning
- Evaluate model performance
- Save the best models

#### API Deployment

```bash
python api/app.py
```

This will:
- Start the FastAPI application on http://localhost:8000
- Load the trained model
- Expose endpoints for predictions

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t house-price-prediction .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 house-price-prediction
   ```

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Documentation

- Project report: [docs/report.md](docs/report.md)
- API usage guide: [docs/api_usage.md](docs/api_usage.md)
- API documentation: http://localhost:8000/docs (when API is running)

## Model Performance

The best performing model is XGBoost with the following metrics:
- RMSE: 0.53
- MAE: 0.38
- R² Score: 0.81

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The California Housing dataset from scikit-learn
- FastAPI for API development
- scikit-learn and XGBoost for machine learning models