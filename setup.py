from setuptools import setup, find_packages

setup(
    name="house-price-prediction",
    version="0.1.0",
    description="Machine Learning model for predicting house prices",
    author="SouAI Developer",
    author_email="developer@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "fastapi>=0.75.0",
        "uvicorn>=0.17.0",
        "pydantic>=1.9.0",
        "joblib>=1.1.0",
        "python-dotenv>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "httpx>=0.22.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)