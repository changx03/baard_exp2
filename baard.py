"""
Implementing the algorithm of Blocking Adversarial Examples by Testing 
Applicability, Reliability and Decidability.
"""
import numpy as np


def dummy_encoder(X):
    """A dummy encoder which returns the inputs."""
    return X


def flatten(X):
    if len(X.shape[1:]) > 1:
        return X.reshape((X.shape[0], -1))
    else:
        return X


class ApplicabilityStage:
    """Testing Applicability in the BAARD framework.
    """

    def __init__(self, *, n_classes=10, quantile=0.999):
        self.n_classes = n_classes
        self.quantile = quantile

    def fit(self, X=None, y=None):
        X = flatten(X)
        self.n_features_ = X.shape[1]
        # Holds min and max
        thresholds = []
        low = (1 - self.quantile) / 2.0
        high = 1 - low
        for i in range(self.n_classes):
            indices = np.where(y == i)[0]
            x_subset = X[indices]
            thresholds.append(np.quantile(x_subset, [low, high], axis=0))
        self.thresholds_ = np.array(thresholds)
        return self

    def predict(self, X, y):
        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.long)
        for i in range(self.n_classes):
            indices = np.where(y == i)[0]
            x_subset = X[indices]
            lower = self.thresholds_[i, 0]
            upper = self.thresholds_[i, 1]
            blocked_indices = np.where(
                np.logical_or(
                    np.any(x_subset < lower, axis=1),
                    np.any(x_subset > upper, axis=1)))[0]
            results[indices[blocked_indices]] = 1
        return results

    def score(self, X, y, labels_adv):
        n = X.shape[0]
        results = self.predict(X, y)
        return len(np.where(results == labels_adv)[0]) / float(n)


class ReliabilityStage:
    """Testing Reliability in the BAARD framework
    """
