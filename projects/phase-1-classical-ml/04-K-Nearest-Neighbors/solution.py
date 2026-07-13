import numpy as np

class KNNClassifier():
    def __init__(self, k=3, distance='euclidean', weighted=False):
        self.k = k
        self.distance = distance
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train, dtype=int)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _get_k_nearest(self, x):
        distances = []

        for x_train in self.X_train:

            if self.distance == 'euclidean':
                dist = self._euclidean_distance(x, x_train)

            elif self.distance == 'manhattan':
                dist = self._manhattan_distance(x, x_train)

            distances.append(dist)

        return np.argsort(distances)[:self.k]

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:

            # get k nearest neighbours
            k_indices = self._get_k_nearest(x)

            # get labels of nearest neighbours
            k_nearest_labels = self.y_train[k_indices]

            # majority voting
            prediction = np.bincount(k_nearest_labels).argmax()

            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        X = np.array(X)

        n_classes = len(np.unique(self.y_train))

        probabilities = []

        for x in X:

            # get k nearest neighbours
            k_indices = self._get_k_nearest(x)

            # get neighbour labels
            k_nearest_labels = self.y_train[k_indices]

            # count class occurrences
            class_counts = np.bincount(
                k_nearest_labels,
                minlength=n_classes
            )

            # convert counts into probabilities
            probs = class_counts / self.k

            probabilities.append(probs)

        return np.array(probabilities)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)