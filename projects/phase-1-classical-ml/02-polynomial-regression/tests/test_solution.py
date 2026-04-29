import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution import PolynomialFeatures, StandardScaler, PolynomialRegression

def test_degree1_equals_linear():
    # setup data
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + np.random.randn(200) * 0.1
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler as SklearnScaler
    
    # your model at degree=1
    poly_model = PolynomialRegression(degree=1)
    poly_model.fit(X, y)
    poly_r2 = poly_model.score(X, y)
    
    # sklearn Ridge baseline on raw features
    scaler = SklearnScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_scaled, y)
    ridge_r2 = ridge.score(X_scaled, y)
    
    # assert they are close within tolerance 0.01
    assert abs(poly_r2 - ridge_r2) < 0.01

def test_higher_degree_higher_train_r2():
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = X[:, 0]**2 + np.random.randn(200) * 0.1

    model1 = PolynomialRegression(degree=1)
    model3 = PolynomialRegression(degree=3)
    
    model1.fit(X, y)
    model3.fit(X, y)

    # assert train R² of degree=3 is greater than degree=1
    assert model3.score(X, y) > model1.score(X, y)

def test_standard_scaler():
    X = np.random.randn(100, 3) * 50 + 20  # messy scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # assert mean of each column is close to 0
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    # assert std of each column is close to 1
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

def test_polynomial_beats_linear():
    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = X[:, 0]**2 + X[:, 1]**2 + np.random.randn(300) * 0.1

    model1 = PolynomialRegression(degree=1)
    model2 = PolynomialRegression(degree=2)

    model1.fit(X, y)
    model2.fit(X, y)

    # assert degree=2 scores higher than degree=1 on training data
    assert model2.score(X, y) > model1.score(X, y)

def test_polynomial_features_shape():
    X = np.random.randn(100, 2)
    pf = PolynomialFeatures(degree=2)
    X_out = pf.fit_transform(X)
    assert X_out.shape == (100, 5)