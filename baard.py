"""
Implementing the algorithm of Blocking Adversarial Examples by Testing 
Applicability, Reliability and Decidability.
"""
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree


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
            if len(indices) == 0:
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
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
            if len(indices) == 0:
                continue
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

    def __init__(self, *, n_classes, k):
        self.n_classes = n_classes
        self.k = k

        self.trees_ = []
        self.detectors_ = []
        self.means_ = np.zeros(self.n_classes, dtype=np.float32)
        self.stds_ = np.zeros(self.n_classes, dtype=np.float32)

    def fit(self, X, y):
        X = flatten(X)
        for i in range(self.n_classes):
            indices = np.where(y == i)[0]
            if len(indices) == 0:
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
            x_subset = X[indices]
            tree = BallTree(x_subset, leaf_size=64)
            self.trees_.append(tree)
        return self

    def search_thresholds(self, X, y, labels_adv):
        X = flatten(X)
        for i in range(self.n_classes):
            indices = np.where(y == i)[0]
            if len(indices) == 0:
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
            x_subset = X[indices]
            label_subset = labels_adv[indices]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            avg_dist = np.expand_dims(avg_dist, axis=1)
            detector = LogisticRegressionCV(cv=5).fit(avg_dist, label_subset)
            self.detectors_.append(detector)

    def predict(self, X, y):
        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.long)
        for i in range(self.n_classes):
            indices = np.where(y == i)[0]
            if len(indices) == 0:
                continue
            x_subset = X[indices]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            avg_dist = np.expand_dims(avg_dist, axis=1)
            predictions = self.detectors_[i].predict(avg_dist)
            results[indices] = predictions
        return results

    def predict_proba(self, X, y):
        n = X.shape[0]
        X = flatten(X)
        # Binary classification. 1: Adversarial example; 0: Benign sample.
        results = np.zeros((n, 2), dtype=np.float32)
        for i in range(self.n_classes):
            indices = np.where(y == i)[0]
            if len(indices) == 0:
                continue
            x_subset = X[indices]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            avg_dist = np.expand_dims(avg_dist, axis=1)
            probs = self.detectors_[i].predict_proba(avg_dist)
            results[indices] = probs
        return results

    def score(self, X, y, labels_adv):
        prob = self.predict_proba(X, y)[:, 1]
        return roc_auc_score(labels_adv, prob)


class DecidabilityStage:
    """Testing Decidability in the BAARD framework
    """

    def __init__(self, *, n_classes, k):
        self.n_classes = n_classes
        self.k = k

    def fit(self, X=None, y=None):
        return self

    def predict(self, X, y):
        pass

    def score(self, X, y, labels_adv):
        pass
