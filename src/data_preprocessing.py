import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    """Load California Housing dataset from sklearn."""
    california_housing = fetch_california_housing()
    
    print("Dataset Description:")
    print(california_housing.DESCR)
    
    print("\nFeature Names:")
    print(california_housing.feature_names)
    
    X = california_housing.data
    y = california_housing.target
    
    print(f"\nData Shape: {X.shape}, Target Shape: {y.shape}")
    
    df = pd.DataFrame(X, columns=california_housing.feature_names)
    
    df["MedHouseVal"] = y
    
    return df


def explore_data(df):
    """Perform exploratory data analysis on the dataset."""
    print("\n== Exploratory Data Analysis ==")
    
    print("\nData Info:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df_clean = df.dropna()
    
    if len(df) != len(df_clean):
        print(f"\nRemoved {len(df) - len(df_clean)} rows with missing values.")
    else:
        print("\nNo missing values found.")
    
    return df_clean


def scale_features(df, features_to_scale=None):
    """Scale numerical features using StandardScaler."""
    if features_to_scale is None:
        features_to_scale = [col for col in df.columns if col != "MedHouseVal"]
    
    X = df[features_to_scale].values
    
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=features_to_scale)
    
    df_scaled["MedHouseVal"] = df["MedHouseVal"].values
    
    print("\nFeatures scaled using StandardScaler.")
    
    return df_scaled, scaler


def visualize_correlations(df):
    """Visualize correlations between features and target variable."""
    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("docs/correlation_heatmap.png")
    plt.close()
    
    sns.pairplot(df.sample(500), diag_kind="kde", plot_kws={"alpha": 0.5})
    plt.savefig("docs/pairplot.png")
    plt.close()
    
    print("\nCorrelation visualizations saved to docs/ folder.")
    
    return corr_matrix


def preprocess_data():
    """Main function to preprocess the California Housing data."""
    df = load_data()
    
    df = explore_data(df)
    
    df = handle_missing_values(df)
    
    correlations = visualize_correlations(df)
    
    df_scaled, scaler = scale_features(df)
    
    df.to_csv("data/processed/california_housing.csv", index=False)
    df_scaled.to_csv("data/processed/california_housing_scaled.csv", index=False)
    
    
    joblib.dump(scaler, "models/scaler.joblib")
    
    print("\nPreprocessed data saved to data/processed/ folder.")
    print("Scaler saved to models/ folder.")
    
    return df, df_scaled, scaler


if __name__ == "__main__":
    preprocess_data()