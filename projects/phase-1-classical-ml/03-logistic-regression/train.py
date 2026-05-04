import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from solution import LogisticRegression

def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description='Train Logistic Regression')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # --- Load data ---
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # --- Scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Train your model ---
    model = LogisticRegression(
        learning_rate=args.learning_rate,
        n_iterations=args.iterations,
        threshold=args.threshold
    )
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # --- Evaluation ---
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))


if __name__ == '__main__':
    main()