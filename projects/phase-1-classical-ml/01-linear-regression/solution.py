import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, l2=0.0):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error  = y_pred - y

            dw = (1 / n_samples) * X.T @ error + (l2 * self.weights)
            db = (1 / n_samples) * np.sum(error)

            self.weights = self.weights - self.learning_rate * dw
            self.bias    = self.bias    - self.learning_rate * db

            self.loss_history.append(self.mse(X, y))

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)

    def fit_normal_equation(self, X, y):
        X_b   = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.bias    = theta[0]
        self.weights = theta[1:]
        return self


class PolynomialFeatures:

    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X, feature_idx=0):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            new_col = X[:, feature_idx] ** d
            X_poly  = np.c_[X_poly, new_col]
        return X_poly