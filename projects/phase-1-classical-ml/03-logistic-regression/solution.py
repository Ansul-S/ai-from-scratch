import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        self.loss_history = []

        for _ in range(self.n_iterations):
            ŷ = self._sigmoid(X @ self.weights + self.bias)
            ŷ = np.clip(ŷ, 1e-15, 1 - 1e-15)  # prevent log(0) or this will explode your loss to inf

            loss = -np.mean(y * np.log(ŷ) + (1 - y) * np.log(1 - ŷ))
            self.loss_history.append(loss)             

            dw = (1/n_samples) * X.T @ (ŷ - y)
            db = (1/n_samples) * np.sum(ŷ - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias    = self.bias    - self.learning_rate * db

        return self

    def predict_proba(self, X):
        ŷ = self._sigmoid(X @ self.weights + self.bias)
        return ŷ
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)