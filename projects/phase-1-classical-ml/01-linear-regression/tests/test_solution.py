import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import LinearRegression, PolynomialFeatures

def test_predict_shape():
    model = LinearRegression()
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (100,), "Prediction shape should match number of samples"

def test_score_range():
    model = LinearRegression()
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert 0 <= r2 <= 1, "R² should be between 0 and 1"

def test_normal_equation_vs_gradient_descent():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.5, -2.0, 3.0]) + 0.5

    model_gd = LinearRegression(learning_rate=0.01, n_iterations=2000)
    model_gd.fit(X, y)

    model_ne = LinearRegression()
    model_ne.fit_normal_equation(X, y)

    np.testing.assert_allclose(model_gd.weights, model_ne.weights, atol=0.1)

def test_ridge_shrinks_weights():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)

    model_no_reg = LinearRegression()
    model_no_reg.fit(X, y, l2=0.0)

    model_ridge = LinearRegression()
    model_ridge.fit(X, y, l2=100.0)

    assert np.sum(np.abs(model_ridge.weights)) < np.sum(np.abs(model_no_reg.weights)), \
        "Ridge should shrink weights"

def test_loss_decreases():
    model = LinearRegression(learning_rate=0.01, n_iterations=100)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    model.fit(X, y)
    assert model.loss_history[0] > model.loss_history[-1], \
        "Loss should decrease over iterations"
    
