import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=3000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_loss_history = [] # Added to track validation progress

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y, y_pred):
        # Helper to keep the loop clean and prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        self.loss_history = []
        self.val_loss_history = []

        for _ in range(self.n_iterations):
            # 1. Forward Pass (Training)
            y_pred_train = self._sigmoid(X @ self.weights + self.bias)
            
            # 2. Compute & Store Training Loss
            train_loss = self._compute_loss(y, y_pred_train)
            self.loss_history.append(train_loss)             

            # 3. Backward Pass (Update weights based on TRAIN only)
            dw = (1/n_samples) * X.T @ (y_pred_train - y)
            db = (1/n_samples) * np.sum(y_pred_train - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # 4. Optional: Validation Loss (Post-update)
            if X_val is not None and y_val is not None:
                y_pred_val = self._sigmoid(X_val @ self.weights + self.bias)
                val_loss = self._compute_loss(y_val, y_pred_val)
                self.val_loss_history.append(val_loss)

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)