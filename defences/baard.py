"""
Implementing the algorithm of Blocking Adversarial Examples by Testing 
Applicability, Reliability and Decidability.
"""
import numpy as np
import torch
from sklearn.neighbors import BallTree
from tqdm import trange


def flatten(X):
    """Converts multi-dimensional samples into vectors."""
    if len(X.shape[1:]) > 1:
        return X.reshape((X.shape[0], -1))
    else:
        return X


class ApplicabilityStage:
    """The 1st stage of BAARD framework. It tests the applicability of given 
    samples.

    This is an outlier detector based on per-class quantile bounding boxes.

    Parameters
    ----------
    n_classes : int, default=10
        The number of output classes.

    quantile : float, default=0.99
        Quantile to compute, which must be between 0 and 1 inclusive.
    """

    def __init__(self, n_classes=10, quantile=0.99, verbose=True):
        self.n_classes = n_classes
        self.quantile = quantile
        self.verbose = verbose

        self.k = 0  # A dummy variable

    def fit(self, X=None, y=None):
        """Fits the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, )
            Target labels.

        Returns
        -------
        self : object
        """
        X = flatten(X)
        self.n_features_ = X.shape[1]
        # Holds min and max
        thresholds = []
        low = (1 - self.quantile) / 2.0
        high = 1 - low
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                print('Class {:d} has no training samples!'.format(i))
                continue
            x_subset = X[idx]
            thresholds.append(np.quantile(x_subset, [low, high], axis=0))
        self.thresholds_ = np.array(thresholds)
        return self

    def search_thresholds(self, X=None, y=None, labels_adv=None):
        """Applicability stage does not require searching for thresholds.
        """
        pass

    def predict(self, X, y):
        """Detects outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        Returns
        -------
        labels : array of shape (n_samples,)
            Returns labels for outliers. 1 is outlier, 0 is benign.
        """
        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.long)
        for i in trange(self.n_classes, desc='Applicability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            lower = self.thresholds_[i, 0]
            upper = self.thresholds_[i, 1]
            # blocked_idx_of_idx = np.where(np.logical_or(np.any(x_subset < lower, axis=1), np.any(x_subset > upper, axis=1)))[0]
            below = np.any(x_subset < lower, axis=1)
            above = np.any(x_subset > upper, axis=1)
            out_of_box = np.logical_or(below, above)
            blocked_idx_of_idx = np.where(out_of_box)[0]
            blocked_idx = idx[blocked_idx_of_idx]
            results[blocked_idx] = 1
        return results


class ReliabilityStage:
    """The 2nd stage of BAARD framework. It tests the reliability of given 
    samples.

    This is a neighbour distance based detector.

    Parameters
    ----------
    n_classes : int, default=10
        Number of output classes.

    k : int, default=10
        Number of neighbours required for each sample.

    quantile : float, default=0.99
        Quantile to compute, which must be between 0 and 1 inclusive.
    """

    def __init__(self, n_classes=10, k=10, quantile=0.99, verbose=True):
        self.n_classes = n_classes
        self.k = k
        self.quantile = quantile
        self.verbose = verbose

        self.trees_ = []
        self.means_ = np.zeros(self.n_classes, dtype=np.float32)
        self.stds_ = np.zeros(self.n_classes, dtype=np.float32)
        self.thresholds_ = np.zeros(self.n_classes, dtype=np.float32)

    def fit(self, X, y):
        """Fits the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, )
            Target labels.

        Returns
        -------
        self : object
        """
        X = flatten(X)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                print('Class {:d} has no training samples!'.format(i))
                continue
            x_subset = X[idx]
            tree = BallTree(x_subset, leaf_size=64)
            self.trees_.append(tree)
        return self

    def search_thresholds(self, X, y, labels_adv):
        """Find thresholds using a linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation set which contains adversarial examples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.
        """
        X = flatten(X)
        for i in trange(self.n_classes, desc='Reliability', disable=not self.verbose):
            idx = np.where(np.logical_and(y == i, labels_adv == 0))[0]
            if len(idx) == 0:
                print('Class {:d} has no training samples!'.format(i))
                continue
            x_subset = X[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            self.thresholds_[i] = np.quantile(avg_dist, self.quantile, axis=0)

    def predict(self, X, y):
        """Detect adversarial examples for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        Returns
        -------
        labels : array of shape (n_samples,)
            Returns labels for adversarial examples. 1 is adversarial example, 
            0 is benign.
        """
        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.long)
        for i in trange(self.n_classes, desc='Reliability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            pred = avg_dist > self.thresholds_[i]
            results[idx] = pred
        return results

    def predict_proba(self, X, y):
        """Predict probability estimates of adversarial examples for samples in
        X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        Returns
        -------
        labels : array of shape (n_samples, 2)
            Returns probability estimates of adversarial examples.
        """
        n = X.shape[0]
        X = flatten(X)
        # Binary classification. 1: Adversarial example; 0: Benign sample.
        results = np.zeros(n, dtype=np.float32)
        for i in trange(self.n_classes, desc='Reliability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            results[idx] = avg_dist
        return results


class DecidabilityStage:
    """The 3rd stage of BAARD framework. It tests the decidability of given 
    samples.

    This is a maximum likelihood based detector. This stage is less effective as
    a standalone test. It should be only used after the samples are failed to 
    reject by the Reliability Stage (2nd stage of BAARD).

    Parameters
    ----------
    n_classes : int, default=10
        The number of output classes.

    k : int, default=100
        Number of neighbours required for each sample.

    quantile : float, default=0.99
        Quantile to compute, which must be between 0 and 1 inclusive.
    """

    def __init__(self, n_classes=10, k=100, quantile=0.99, verbose=True):
        self.n_classes = n_classes
        self.k = k
        self.quantile = quantile
        self.verbose = verbose

        self.likelihoods_mean_ = np.zeros((n_classes, n_classes), dtype=np.float32)
        self.thresholds_ = np.zeros(n_classes, dtype=np.float32)

    def fit(self, X, y):
        """Fits the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, )
            Target labels.

        Returns
        -------
        self : object
        """
        X = flatten(X)
        self.tree_ = BallTree(X, leaf_size=128)
        self.y_train_ = y
        return self

    def search_thresholds(self, X, y=None, labels_adv=None):
        """Find thresholds using a linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation set which contains adversarial examples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.
        """
        X = flatten(X)
        for i in trange(self.n_classes, desc='Decidability', disable=not self.verbose):
            idx = np.where(np.logical_and(y == i, labels_adv == 0))[0]
            n = len(idx)
            if n == 0:
                print('Class {:d} has no training samples!'.format(i))
                continue
            x_sub = X[idx]
            y_sub = y[idx]
            likelihood = self.__get_likelihoods(x_sub, y_sub)
            self.thresholds_[i] = np.quantile(likelihood, 1 - self.quantile, axis=0)

    def predict(self, X, y):
        """Detect adversarial examples for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        Returns
        -------
        labels : array of shape (n_samples,)
            Returns labels for adversarial examples. 1 is adversarial example, 
            0 is benign.
        """
        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.long)
        for i in trange(self.n_classes, desc='Decidability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_sub = X[idx]
            y_sub = y[idx]
            likelihood = self.__get_likelihoods(x_sub, y_sub)
            results[idx] = likelihood < self.thresholds_[i]
        return results

    def predict_proba(self, X, y):
        """Predict probability estimates of adversarial examples for samples in 
        X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        Returns
        -------
        labels : array of shape (n_samples, 2)
            Returns probability estimates of adversarial examples.
        """
        likelihoods = self.__get_likelihoods(X, y)
        return 1.0 - likelihoods

    def __get_likelihoods(self, x, y):
        neigh_idx = self.tree_.query(x, self.k, return_distance=False)
        neigh_y = np.array([self.y_train_[i] for i in neigh_idx])
        y_grid = np.repeat(np.expand_dims(y, axis=0).transpose(), self.k, axis=1)
        likelihood = np.sum(neigh_y == y_grid, axis=1) / self.k
        return likelihood


class BAARDOperator:
    """BAARD framework. It runs each pre-trained stage in sequence.

    Parameters
    ----------
    stages : array of {ApplicabilityStage, ReliabilityStage, DecidabilityStage}
        An array of the BAARD stages
    """

    def __init__(self, stages):
        self.stages = stages

    def fit(self, X, y):
        """Fits models for each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, )
            Target labels.

        Returns
        -------
        self : object
        """
        for stage in self.stages:
            stage.fit(X, y)
        return self

    def search_thresholds(self, X, y, labels_adv):
        """Search thresholds for each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation set which contains adversarial examples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.
        """
        for stage in self.stages:
            stage.search_thresholds(X, y, labels_adv)

    def detect(self, X, y):
        """Detect adversarial examples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model. NOTE: Since the detector 
            has no access to the model, we do not have the true labels at the 
            run time. Therefore, y is the predictions rather than the true 
            labels.

        Returns
        -------
        labels : array of shape (n_samples,)
            Returns labels for adversarial examples. 1 is adversarial example, 
            0 is benign.
        """
        n = len(X)
        labels = np.zeros(n, dtype=np.long)
        for stage in self.stages:
            uncertain_idx = np.where(labels == 0)[0]
            # No benign samples left in the queue
            if np.sum(uncertain_idx) == 0:
                return labels
            X_uncertain = X[uncertain_idx]
            y_uncertain = y[uncertain_idx]
            results = stage.predict(X_uncertain, y_uncertain)
            positive_results = np.where(results == 1)[0]
            positive_idx = uncertain_idx[positive_results]
            labels[positive_idx] = 1
        return labels

    def save(self, path):
        thresholds = []
        quantiles = np.zeros(3, dtype=np.float32)
        ks = np.zeros(3, dtype=np.long)
        for i, stage in enumerate(self.stages):
            thresholds.append(stage.thresholds_)
            ks[i] = stage.k
            quantiles[i] = stage.quantile
        obj = {
            "thresholds": thresholds,
            "quantiles": quantiles,
            "ks": ks}
        torch.save(obj, path)
        print('Save to:', path)

    def load(self, path):
        obj = torch.load(path)
        for i in range(len(self.stages)):
            self.stages[i].thresholds_ = obj['thresholds'][i]
            self.stages[i].quantile = obj['quantiles'][i]
            self.stages[i].k = obj['ks'][i]
        print('Load from:', path)
