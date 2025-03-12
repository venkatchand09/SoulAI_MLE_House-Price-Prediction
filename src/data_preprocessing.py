import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load California Housing dataset from sklearn."""
    california_housing = fetch_california_housing()
    
    # View dataset description
    print("Dataset Description:")
    print(california_housing.DESCR)
    
    # Get feature names
    print("\nFeature Names:")
    print(california_housing.feature_names)
    
    # Access data and target
    X = california_housing.data
    y = california_housing.target
    
    print(f"\nData Shape: {X.shape}, Target Shape: {y.shape}")
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=california_housing.feature_names)
    
    # Add target variable
    df["MedHouseVal"] = y
    
    return df


def explore_data(df):
    """Perform exploratory data analysis on the dataset."""
    print("\n== Exploratory Data Analysis ==")
    
    # Data types and missing values
    print("\nData Info:")
    print(df.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Drop any missing values (note: California Housing dataset typically has no missing values)
    df_clean = df.dropna()
    
    # Check if any rows were dropped
    if len(df) != len(df_clean):
        print(f"\nRemoved {len(df) - len(df_clean)} rows with missing values.")
    else:
        print("\nNo missing values found.")
    
    return df_clean


def scale_features(df, features_to_scale=None):
    """Scale numerical features using StandardScaler."""
    # If no specific features provided, scale all except target
    if features_to_scale is None:
        features_to_scale = [col for col in df.columns if col != "MedHouseVal"]
    
    # Extract features to scale
    X = df[features_to_scale].values
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform features
    X_scaled = scaler.fit_transform(X)
    
    # Create new DataFrame with scaled features
    df_scaled = pd.DataFrame(X_scaled, columns=features_to_scale)
    
    # Add target variable back
    df_scaled["MedHouseVal"] = df["MedHouseVal"].values
    
    print("\nFeatures scaled using StandardScaler.")
    
    return df_scaled, scaler


def visualize_correlations(df):
    """Visualize correlations between features and target variable."""
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("docs/correlation_heatmap.png")
    plt.close()
    
    # Plot pairplot for a sample of the data
    sns.pairplot(df.sample(500), diag_kind="kde", plot_kws={"alpha": 0.5})
    plt.savefig("docs/pairplot.png")
    plt.close()
    
    print("\nCorrelation visualizations saved to docs/ folder.")
    
    return corr_matrix


def preprocess_data():
    """Main function to preprocess the California Housing data."""
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Visualize correlations
    correlations = visualize_correlations(df)
    
    # Scale features
    df_scaled, scaler = scale_features(df)
    
    # Save preprocessed data
    df.to_csv("data/processed/california_housing.csv", index=False)
    df_scaled.to_csv("data/processed/california_housing_scaled.csv", index=False)
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, "models/scaler.joblib")
    
    print("\nPreprocessed data saved to data/processed/ folder.")
    print("Scaler saved to models/ folder.")
    
    return df, df_scaled, scaler


if __name__ == "__main__":
    preprocess_data()