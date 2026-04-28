import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge as SklearnLinearRegression

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        cols = [X]

        for d in range(2, self.degree + 1):
            cols.append(X ** d)

            if d == 2:
                for i, j in combinations(range(n_features), 2):
                    interaction = X[:, i] * X[:, j]
                    cols.append(interaction.reshape(-1, 1))

        return np.hstack(cols)
    
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Learn mean and std from training data
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

        return self

    def transform(self, X):
        """
        Apply scaling using learned parameters
        """
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """
        Fit + transform (used during training)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Convert scaled data back to original scale
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.array(X)
        return (X * self.std_) + self.mean_

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model = SklearnLinearRegression(alpha=0.1)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_poly = self.poly.fit_transform(X)   # expand features
        X_scaled = self.scaler.transform(X_poly)  # use learned scaling
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.score(X_scaled, y)