import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution import LogisticRegression


# 1. Predictions are binary (0 or 1 only)
def test_predictions_are_binary():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)

    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})


# 2. predict_proba outputs are in range (0, 1)
def test_predict_proba_range():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert np.all(probs > 0) and np.all(probs < 1)


# 3. Loss decreases over training iterations
def test_loss_decreases():
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    model = LogisticRegression(n_iterations=200, learning_rate=0.1)
    model.fit(X, y)

    losses = model.loss_history

    assert losses[-1] < losses[0]


# 4. Accuracy beats random baseline (> 0.5)
def test_accuracy_beats_random():
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)

    acc = model.score(X, y)

    assert acc > 0.5


# 5. sklearn benchmark within tolerance (2%)
def test_against_sklearn():
    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    from sklearn.linear_model import LogisticRegression as SklearnLR

    # your model
    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)
    acc_custom = model.score(X, y)

    # sklearn model
    sk_model = SklearnLR()
    sk_model.fit(X, y)
    acc_sklearn = sk_model.score(X, y)

    assert abs(acc_custom - acc_sklearn) < 0.02