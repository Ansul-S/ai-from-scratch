import numpy as np
import pytest
import sys
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution import KNNClassifier


# 1. k=1 should memorise the training data
def test_k1_memorises_training():
    np.random.seed(42)

    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 3, size=50)

    model = KNNClassifier(k=1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    accuracy = np.mean(predictions == y_train)

    assert accuracy == 1.0, (
        f"k=1 should memorise training data, but got accuracy {accuracy}"
    )


# 2. All predictions should be valid classes
def test_predictions_are_valid_classes():
    np.random.seed(42)

    X_train = np.random.rand(100, 4)
    y_train = np.random.randint(0, 3, size=100)
    X_test = np.random.rand(20, 4)

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(np.unique(predictions)).issubset({0, 1, 2}), (
        f"Invalid predicted classes: {np.unique(predictions)}"
    )


# 3. Accuracy should beat random guessing
def test_accuracy_beats_random():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = np.mean(predictions == y_test)

    assert accuracy > 0.33, (
        f"Accuracy should beat random guessing. Got {accuracy:.2f}"
    )


# 4. Euclidean distance should be correct
def test_euclidean_distance_correct():
    model = KNNClassifier(k=3)

    point_a = np.array([0, 0])
    point_b = np.array([3, 4])

    distance = model._euclidean_distance(point_a, point_b)

    assert distance == pytest.approx(5.0), (
        f"Expected distance 5.0, got {distance}"
    )


# 5. Custom KNN should perform within 2% of sklearn KNN
def test_against_sklearn():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )

    k = 5

    # Your implementation
    custom_knn = KNNClassifier(k=k)
    custom_knn.fit(X_train, y_train)
    custom_predictions = custom_knn.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)

    # sklearn implementation
    sklearn_knn = SklearnKNN(
        n_neighbors=k,
        metric="euclidean"
    )
    sklearn_knn.fit(X_train, y_train)
    sklearn_predictions = sklearn_knn.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    difference = abs(custom_accuracy - sklearn_accuracy)

    assert difference <= 0.02, (
        f"Custom KNN accuracy ({custom_accuracy:.4f}) differs from "
        f"sklearn accuracy ({sklearn_accuracy:.4f}) by {difference:.4f}, "
        f"which is more than 2%"
    )g