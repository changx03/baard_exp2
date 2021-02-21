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

    fpr : float, default=0.001
        Determines the False Positive Rate in the validation set.
    """

    def __init__(self, n_classes=10, fpr=0.001, verbose=True):
        self.n_classes = n_classes
        self.fpr = fpr
        self.verbose = verbose

        self.k = 0  # A dummy variable
        self.n_tolerance_ = np.zeros(self.n_classes, dtype=np.int)

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
        self.dtype_ = X.dtype
        self.X_train_ = flatten(X)
        self.y_train_ = y
        return self

    def search_thresholds(self, X, y, labels_adv=None):
        """Find tolerate based on False Positive Rate.
        """
        # Only uses bengin samples.
        idx = np.where(labels_adv == 0)[0]
        X = X[idx]
        y = y[idx]

        # First, find thresholds.
        print('[BAARD] s1 self.fpr:', self.fpr)
        thresholds = self.__search_boundingboxes(self.fpr)
        self.thresholds_ = thresholds
        pred = self.predict_prob(X, y) > 0
        print('[BAARD] s1 eval. fpr (only thresholds):', np.mean(pred == 1))

        # Second, find tolerance.
        fpr = self.fpr
        n_outs = self.predict_prob(X, y)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            self.n_tolerance_[i] = np.quantile(n_outs[idx], 1 - fpr)
        print('[BAARD] s1 tolerance:', self.n_tolerance_)

        # Show FPR on validation set
        pred = self.predict(X, y)
        print('[BAARD] s1 eval. fpr (thresholds + tolerance):', np.mean(pred == 1))

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
        n_outs = self.predict_prob(X, y)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            n_tolerance = self.n_tolerance_[i]
            blocked = n_outs[idx] > n_tolerance
            results[idx] = blocked
        return results

    def predict_prob(self, X, y, thresholds=None):
        """Detects outliers. It returns the number of features which is outside 
        the bounding boxes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        Returns
        -------
        n_features : array of shape (n_samples,)
            Returns the number of features which is outside the bounding boxes 
            for each sample.
        """
        if thresholds is None:
            thresholds = self.thresholds_

        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.float)
        for i in trange(self.n_classes, desc='Applicability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            lower = thresholds[i, 0]
            upper = thresholds[i, 1]
            n_below = np.sum(x_subset < lower, axis=1)
            n_above = np.sum(x_subset > upper, axis=1)
            n_out = n_below + n_above
            results[idx] = n_out
        return results

    def __search_boundingboxes(self, t):
        low = t / 2.
        high = 1. - low
        thresholds = []
        for i in range(self.n_classes):
            idx = np.where(self.y_train_ == i)[0]
            if len(idx) == 0:
                print('[BAARD] Class {:d} has no training samples!'.format(i))
                continue
            x_subset = self.X_train_[idx]
            thresholds.append(np.quantile(x_subset, [low, high], axis=0))
        return np.array(thresholds, dtype=self.dtype_)


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

    fpr : float, default=0.001
        Determines the False Positive Rate in the validation set.
    """

    def __init__(self, n_classes=10, k=10, fpr=0.001, verbose=True):
        self.n_classes = n_classes
        self.k = k
        self.fpr = fpr
        self.verbose = verbose

        self.trees_ = []
        self.means_ = np.zeros(self.n_classes, dtype=np.float)
        self.stds_ = np.zeros(self.n_classes, dtype=np.float)
        self.thresholds_ = np.zeros(self.n_classes, dtype=np.float)

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
                print('[BAARD] Class {:d} has no training samples!'.format(i))
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
        # Only uses bengin samples.
        idx = np.where(labels_adv == 0)[0]
        X = X[idx]
        X = flatten(X)
        y = y[idx]

        quantile = 1. - self.fpr
        for i in trange(self.n_classes, desc='Reliability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                print('[BAARD] Class {:d} has no training samples!'.format(i))
                continue
            x_subset = X[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            self.thresholds_[i] = np.quantile(avg_dist, quantile, axis=0)

        # Show FPR on validation set
        pred = self.predict(X, y)
        print('[BAARD] s2 eval. fpr:', np.mean(pred == 1))

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
        results = np.zeros(n, dtype=np.float)
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

    fpr : float, default=0.001
        Determines the False Positive Rate in the validation set.
    """

    def __init__(self, n_classes=10, k=100, fpr=0.001, verbose=True):
        self.n_classes = n_classes
        self.k = k
        self.fpr = fpr
        self.verbose = verbose

        self.likelihoods_mean_ = np.zeros((n_classes, n_classes), dtype=np.float)
        self.thresholds_ = np.zeros(n_classes, dtype=np.float)

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
        # Only uses bengin samples.
        idx = np.where(labels_adv == 0)[0]
        X = X[idx]
        X = flatten(X)
        y = y[idx]

        for i in trange(self.n_classes, desc='Decidability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            n = len(idx)
            if n == 0:
                print('[BAARD] Class {:d} has no training samples!'.format(i))
                continue
            x_sub = X[idx]
            y_sub = y[idx]
            likelihood = self.__get_likelihoods(x_sub, y_sub)
            self.thresholds_[i] = np.quantile(likelihood, self.fpr, axis=0)

        # Show FPR on validation set
        pred = self.predict(X, y)
        print('[BAARD] s3 eval. fpr:', np.mean(pred == 1))

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
        n = X.shape[0]
        X = flatten(X)
        results = np.zeros(n, dtype=np.float)
        for i in trange(self.n_classes, desc='Decidability', disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_sub = X[idx]
            y_sub = y[idx]
            likelihood = self.__get_likelihoods(x_sub, y_sub)
            results[idx] = 1 - likelihood
        return results

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

    def fit(self, X, y, X_s1=None):
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
        if X_s1 is not None:
            self.stages[0].fit(X_s1, y)
            for stage in self.stages[1:]:
                stage.fit(X, y)
        else:
            for stage in self.stages:
                stage.fit(X, y)
        return self

    def search_thresholds(self, X, y, labels_adv=None):
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
        if labels_adv is None:
            labels_adv = np.zeros_like(y)
        for stage in self.stages:
            stage.search_thresholds(X, y, labels_adv)

    def detect(self, X, y, per_stage=False):
        """Detect adversarial examples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        per_stage : bool, default=False
            If per_stage is True, the functions will return the results per 
            stage.

        Returns
        -------
        *labels : array of shape (n_samples,)
            Returns labels for adversarial examples. 1 is adversarial example, 
            0 is benign.
        """
        n = len(X)
        labelled_as_adv = np.zeros(n, dtype=np.long)
        labelled_as_adv_per_stage = []
        for stage in self.stages:
            uncertain_idx = np.where(labelled_as_adv == 0)[0]
            # No benign samples left in the queue
            if np.sum(uncertain_idx) == 0:
                labelled_as_adv_per_stage.append(labelled_as_adv.copy())
                continue
            X_uncertain = X[uncertain_idx]
            y_uncertain = y[uncertain_idx]
            results = stage.predict(X_uncertain, y_uncertain)
            positive_results = np.where(results == 1)[0]
            positive_idx = uncertain_idx[positive_results]
            labelled_as_adv[positive_idx] = 1
            # Save a copy
            labelled_as_adv_per_stage.append(labelled_as_adv.copy())

        if per_stage:
            return tuple(labelled_as_adv_per_stage)
        else:
            return labelled_as_adv

    def save(self, path, without_s1=False):
        thresholds = []
        fprs = np.zeros(3, dtype=np.float)
        ks = np.zeros(3, dtype=np.long)
        for i, stage in enumerate(self.stages):
            thresholds.append(stage.thresholds_)
            ks[i] = stage.k
            fprs[i] = stage.fpr
        if without_s1:
            obj = {
                'thresholds': thresholds,
                'fprs': fprs,
                'ks': ks}
        else:
            obj = {
                'thresholds': thresholds,
                'fprs': fprs,
                'ks': ks,
                'n_tolerance': self.stages[0].n_tolerance_}
        torch.save(obj, path)
        print('[BAARD] Save to:', path)

    def load(self, path):
        obj = torch.load(path)
        for i in range(len(self.stages)):
            self.stages[i].thresholds_ = obj['thresholds'][i]
            self.stages[i].fpr = obj['fprs'][i]
            self.stages[i].k = obj['ks'][i]
        self.stages[0].n_tolerance_ = obj['n_tolerance']
        print('[BAARD] Load from:', path)

# Testing
if __name__ == '__main__':
    k = 3
    neigh_idx = np.array([[0, 1, 2], [3, 4, 0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 1])
    neigh_y = np.array([y_train[i] for i in neigh_idx])
    print('neigh_y', neigh_y)
    y_grid = np.repeat(np.expand_dims(y, axis=0).transpose(), k, axis=1)
    print('y_grid', y_grid)
    likelihood = np.sum(neigh_y == y_grid, axis=1) / k
    print('likelihood', likelihood)
