import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_preprocessed_data(scaled=True):
    """Load preprocessed data."""
    if scaled:
        df = pd.read_csv("data/processed/california_housing_scaled.csv")
        print("Loaded scaled preprocessed data.")
    else:
        df = pd.read_csv("data/processed/california_housing.csv")
        print("Loaded unscaled preprocessed data.")
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    # Split dataset into features and target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"\nData split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test


def train_baseline_models(X_train, y_train, X_test, y_test):
    """Train and evaluate baseline models."""
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    }
    
    results = []
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2
        })
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")
    
    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    
    # Visualize model performance
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Model", y="RMSE", data=results_df, hue="Model", palette="viridis", legend=False)
    plt.title("Baseline Model Comparison: RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("docs/baseline_model_comparison.png")
    plt.close()
    
    return models, results_df


def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest model using RandomizedSearchCV."""
    print("\nOptimizing Random Forest model...")
    
    # Define hyperparameter search space
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    # Initialize model
    rf = RandomForestRegressor(random_state=42)
    
    # Setup RandomizedSearchCV
    rf_search = RandomizedSearchCV(
        rf, 
        rf_params, 
        n_iter=10, 
        scoring="neg_mean_squared_error", 
        cv=3, 
        random_state=42, 
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    rf_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = rf_search.best_estimator_
    
    print(f"Best parameters: {rf_search.best_params_}")
    
    return best_rf


def optimize_xgboost(X_train, y_train):
    """Optimize XGBoost model using GridSearchCV."""
    print("\nOptimizing XGBoost model...")
    
    # Define hyperparameter search space
    xgb_params = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 10],
        "subsample": [0.7, 0.8, 1.0]
    }
    
    # Initialize model
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    
    # Setup GridSearchCV
    xgb_search = GridSearchCV(
        xgb, 
        xgb_params, 
        scoring="neg_mean_squared_error", 
        cv=3, 
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    xgb_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = xgb_search.best_estimator_
    
    print(f"Best parameters: {xgb_search.best_params_}")
    
    return best_xgb


def evaluate_optimized_models(best_rf, best_xgb, X_test, y_test):
    """Evaluate optimized models."""
    print("\nEvaluating optimized models...")
    
    # Initialize models dictionary
    models = {
        "Optimized Random Forest": best_rf,
        "Optimized XGBoost": best_xgb
    }
    
    results = []
    
    # Evaluate models
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2
        })
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")
    
    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    
    # Visualize model performance
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Model", y="RMSE", data=results_df, hue="Model", palette="viridis", legend=False)
    plt.title("Optimized Model Comparison: RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("docs/optimized_model_comparison.png")
    plt.close()
    
    return models, results_df


def save_models(best_rf, best_xgb):
    """Save trained models."""
    print("\nSaving models...")
    
    # Save using Pickle
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(best_rf, f)
    
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(best_xgb, f)
    
    # Save using Joblib (recommended for large models)
    joblib.dump(best_rf, "models/random_forest_model.joblib")
    joblib.dump(best_xgb, "models/xgboost_model.joblib")
    
    print("Models saved successfully!")


def train_and_optimize_models():
    """Main function to train and optimize models."""
    # Load preprocessed data
    df = load_preprocessed_data(scaled=True)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train baseline models
    baseline_models, baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # Optimize Random Forest model
    best_rf = optimize_random_forest(X_train, y_train)
    
    # Optimize XGBoost model
    best_xgb = optimize_xgboost(X_train, y_train)
    
    # Evaluate optimized models
    optimized_models, optimized_results = evaluate_optimized_models(best_rf, best_xgb, X_test, y_test)
    
    # Save models
    save_models(best_rf, best_xgb)
    
    # Save results to CSV
    baseline_results.to_csv("docs/baseline_model_results.csv", index=False)
    optimized_results.to_csv("docs/optimized_model_results.csv", index=False)
    
    print("\nResults saved to docs/ folder.")
    
    return baseline_models, baseline_results, optimized_models, optimized_results


if __name__ == "__main__":
    train_and_optimize_models()