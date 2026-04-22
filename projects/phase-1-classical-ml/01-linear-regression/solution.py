import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Step 1 - prediction
            y_pred = X @ self.weights + self.bias

            # Step 2 - error
            error = y_pred - y

            # Step 3 - gradients
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            # Step 4 - update
            self.weights = self.weights - dw
            self.bias = self.bias - db

            # Track loss
            self.loss_history.append(self.mse(X, y))
        return self    
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y)**2)
    
    def fit_normal_equation(self, X, y):
        # θ = (XᵀX)⁻¹Xᵀy
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias column
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.bias    = theta[0]
        self.weights = theta[1:]
        return self