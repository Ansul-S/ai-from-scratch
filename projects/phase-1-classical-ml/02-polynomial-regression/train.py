import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from solution import PolynomialFeatures, StandardScaler, PolynomialRegression

def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description='Train Polynomial Regression')
    parser.add_argument('--degree',    type=int,   default=3)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    # --- Load data ---
    df = pd.read_csv('data/hour.csv')
    features = ['atemp', 'hum', 'windspeed', 'hr', 'season', 'workingday']
    target   = 'cnt'

    X = df[features].values
    y = df[target].values

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # --- Train your model ---
    model = PolynomialRegression(degree=args.degree)
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # --- Scores ---
    train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
    test_r2  = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    rmse     = np.sqrt(np.mean((y_test_pred - y_test)**2))

    # --- Sklearn benchmark ---
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    ridge_test_r2 = ridge.score(X_test, y_test)

    # --- Print results ---
    print(f"\nTraining Polynomial Regression (degree={args.degree})...")
    print(f"Features: {', '.join(features)}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"Ridge Test R² (benchmark): {ridge_test_r2:.4f}")

if __name__ == '__main__':
    main()