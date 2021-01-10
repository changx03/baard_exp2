"""
Implementing the algorithm of Blocking Adversarial Examples by Testing 
Applicability, Reliability and Decidability.
"""
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree


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

    def __init__(self, n_classes=10, quantile=0.99):
        self.n_classes = n_classes
        self.quantile = quantile

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
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
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
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            lower = self.thresholds_[i, 0]
            upper = self.thresholds_[i, 1]
            blocked_idx = np.where(
                np.logical_or(
                    np.any(x_subset < lower, axis=1),
                    np.any(x_subset > upper, axis=1)))[0]
            results[idx[blocked_idx]] = 1
        return results

    def score(self, X, y, labels_adv):
        """Returns the accuracy score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target labels for outliers. 1 is outlier, 0 is benign.
        """
        n = X.shape[0]
        results = self.predict(X, y)
        correct_results = results == labels_adv
        return np.sum(correct_results) / n


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
    """

    def __init__(self, n_classes=10, k=10):
        self.n_classes = n_classes
        self.k = k

        self.trees_ = []
        self.detectors_ = []
        self.means_ = np.zeros(self.n_classes, dtype=np.float32)
        self.stds_ = np.zeros(self.n_classes, dtype=np.float32)

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
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
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
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
            x_subset = X[idx]
            label_subset = labels_adv[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            avg_dist = np.expand_dims(avg_dist, axis=1)
            detector = LogisticRegressionCV(cv=5)
            shuffle_idx = np.random.permutation(avg_dist.shape[0])
            detector.fit(avg_dist[shuffle_idx],
                         label_subset[shuffle_idx])
            self.detectors_.append(detector)

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
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            avg_dist = np.expand_dims(avg_dist, axis=1)
            pred = self.detectors_[i].predict(avg_dist)
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
        results = np.zeros((n, 2), dtype=np.float32)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            x_subset = X[idx]
            tree = self.trees_[i]
            dist, _ = tree.query(x_subset, k=self.k)
            avg_dist = np.sum(dist, axis=1) / self.k
            avg_dist = np.expand_dims(avg_dist, axis=1)
            probs = self.detectors_[i].predict_proba(avg_dist)
            results[idx] = probs
        return results

    def score(self, X, y, labels_adv):
        """Returns the ROC AUC score given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.

        Returns
        -------
        score : float
            The ROC AUC score based on the linear model.
        """
        prob = self.predict_proba(X, y)[:, 1]
        return roc_auc_score(labels_adv, prob)


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

    n_bins : int, default=2
        Number of likelihoods for defining the threshold. n_bins must be less or
        equal to n_classes.
    """

    def __init__(self, n_classes=10, k=100, n_bins=2):
        if n_bins > n_classes:
            raise ValueError('n_bins must less or equal to n_classes!')

        self.n_classes = n_classes
        self.k = k
        self.n_bins = n_bins

        self.detectors_ = []

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
        likelihoods = self.__get_likelihoods(X)[:, -self.n_bins:]
        # create detectors for each class
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                raise ValueError(
                    'Class {:d} has no training samples!'.format(i))
            label_subset = labels_adv[idx]
            detector = LogisticRegressionCV(cv=5)
            detector.fit(likelihoods[idx], label_subset)
            self.detectors_.append(detector)

    def predict(self, X, y=None):
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
        n = len(X)
        likelihoods = self.__get_likelihoods(X)[:, -self.n_bins:]
        results = np.zeros(n, dtype=np.long)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            pred = self.detectors_[i].predict(likelihoods[idx])
            results[idx] = pred
        return results

    def predict_proba(self, X, y=None):
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
        likelihoods = self.__get_likelihoods(X)[:, -self.n_bins:]
        results = np.zeros((n, 2), dtype=np.float32)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            prob = self.detectors_[i].predict_proba(likelihoods[idx])
            results[idx] = prob
        return results

    def score(self, X, y=None, labels_adv=None):
        """Returns the ROC AUC score given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.

        Returns
        -------
        score : float
            The ROC AUC score based on the linear model.
        """
        prob = self.predict_proba(X, y)[:, 1]
        return roc_auc_score(labels_adv, prob)

    def __get_likelihoods(self, X):
        n = X.shape[0]
        X = flatten(X)
        neighbors_idx = self.tree_.query(X, self.k, return_distance=False)
        neighbors_y = np.array([self.y_train_[i] for i in neighbors_idx])
        bins = np.zeros((n, self.n_classes), dtype=np.float32)
        for i in range(n):
            frequency = stats.relfreq(
                neighbors_y[i],
                numbins=self.n_classes,
                defaultreallimits=(0, self.n_classes-1)
            )[0]
            bins[i] = np.sort(frequency)
        return bins


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

    def score(self, X, y, pred, labels_adv):
        """Rate of success. The success means (1) correctly blocked by detector.
        (2) If an adversarial example does not alter the prediction, we allow it
        to pass the detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, )
            Target labels.

        pred : array-like of shape (n_samples, )
            Predicted labels from the initial model.

        labels_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.

        Returns
        -------
        success_rate : float
            The fraction of correctly classified samples.
        """
        n = len(X)
        blocked_labels = self.detect(X, pred)
        matched_adv = blocked_labels == labels_adv
        unmatched_idx = np.where(matched_adv == False)[0]
        # Two situations for unmatched samples:
        # 1. false positive (FP): Mislabel benign samples as advasarial examples.
        # 2. false negative (FN): Fail to reject the sample.
        # FP is always wrong. FN is ok, only if the prediction matches the true
        # label.
        fn_idx = unmatched_idx[np.where(labels_adv[unmatched_idx] == 1)[0]]
        if len(fn_idx) == 0:
            matched_label = 0
        else:
            matched_label = np.sum(y[fn_idx] == pred[fn_idx])
        total_correct = np.sum(matched_adv) + matched_label
        success_rate = total_correct / n
        return success_rate
