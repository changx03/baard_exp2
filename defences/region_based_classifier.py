"""
Region-based classification using PyTorch.
"""
import datetime
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def generate_random_samples(x, x_min, x_max, r, size):
    """Generates uniformly distributed random samples around x within hypercube
    B(x, r), where r is L-infinity distance.
    """
    shape = tuple([size] + list(x.shape))
    dtype = x.dtype
    noise = np.random.uniform(low=-abs(r), high=abs(r), size=shape).astype(dtype)
    rng_samples = np.repeat([x], repeats=size, axis=0) + noise
    rng_samples = np.minimum(np.maximum(rng_samples, x_min), x_max)
    return rng_samples


class RegionBasedClassifier:
    """Region Based Classifier for robust classification. 

    This classifier is used to restore the true label of adversarial examples.

    Parameters
    ----------
    model : torch.nn.Module object, default=None
        The classifier.

    r : float, default=0.2
        The radius from the sample.

    sample_size : int, default=1000
        The number of samples generated within the radius.

    n_classes : int, default=10
        The number of output classes.

    x_min : float or array, default=0.0

    x_max : float or array, default=1.0

    batch_size : int, default=128
        Mini batch size for training the autoencoder.

    r0 : float, default=0.0
        The initial radius for the binary search.

    step_size : float, default=0.01
            Step size for each iteration.

    stop_value : float, default=None
        Maximum searching radius.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, *, model=None, r=0.2, sample_size=1000, n_classes=10,
                 x_min=0.0, x_max=1.0, batch_size=128, r0=0.0, step_size=0.01,
                 stop_value=0.4, device='gpu'):
        self.model = model
        self.r = r
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.x_min = x_min
        self.x_max = x_max
        self.batch_size = batch_size
        self.r0 = r0
        self.step_size = step_size
        self.stop_value = stop_value
        self.device = device

    def search_r(self, X, y, r0=0.0, step_size=0.01, stop=0.4, update=True,
                 verbose=0, exit_early=True):
        n = len(X)
        r = r0
        time_start = time.time()
        tensor_X = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        tensor_predictions_point = self.__predict(tensor_X)
        corrects = tensor_predictions_point.eq(
            tensor_y.view_as(tensor_predictions_point)).sum().item()
        acc_point = corrects / n
        acc_region = self.score(X, y, r=r)
        results = [[r, acc_region]]

        if verbose > 0:
            print('Accuracy on validation set: {:.4f}'.format(acc_point))

        # If stop is not None, keep running until reaches stop value.
        if stop is None:
            stop = r0

        r_best = r0
        acc_best = 0.0
        while acc_region >= acc_point or r <= stop:
            time_elapsed = time.time() - time_start
            if verbose > 0:
                print('[{:s}] Accuracy on region-based: {:.4f}, r: {:.2f}'.format(
                    str(datetime.timedelta(seconds=time_elapsed)), acc_region, r))
            time_start = time.time()
            acc_region = self.score(X, y, r=r)
            results.append([r, acc_region])
            if acc_best <= acc_region:
                acc_best = acc_region
                r_best = r
            if exit_early and acc_best > acc_region:
                break
            r += step_size

        r_best = r
        for res in reversed(results):
            if res[1] >= acc_point:
                break
            r_best = res[0]

        if update:
            self.r = r_best

        return r_best, np.array(results, dtype=np.float32)

    def fit(self, X=None, y=None):
        """Region-based classifier does not require fit."""
        return self

    def search_thresholds(self, X, y=None, labels_adv=None, verbose=1):
        # find all indices are not adversarial examples
        indices = np.where(labels_adv == 0)[0]
        X = X[indices]
        y = y[indices]
        r_best, _ = self.search_r(
            X,
            y,
            r0=self.r0,
            step_size=self.step_size,
            stop=self.stop_value,
            update=True,
            verbose=verbose)
        print('Best r =', r_best)
        return r_best

    def predict(self, X, r=None):
        """Predicts class labels for samples in X."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        if r is None:
            r = self.r

        n = X.shape[0]
        pred = -np.ones(n, dtype=np.long)

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                x_rng = generate_random_samples(
                    X[i],
                    x_min=self.x_min,
                    x_max=self.x_max,
                    r=r,
                    size=self.sample_size)
                tensor_x_rng = torch.tensor(x_rng, dtype=torch.float32)
                tensor_pred_rng = self.__predict(tensor_x_rng)
                pred_rng = tensor_pred_rng.cpu().detach().numpy()
                pred[i] = np.bincount(pred_rng).argmax()

        return pred

    def detect(self, X, y):
        return self.predict(X)

    def predict_proba(self, X, r=None):
        """Returns probability estimates."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        if r is None:
            r = self.r

        n = X.shape[0]
        probabilities = np.zeros((n, self.n_classes), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                x_rng = generate_random_samples(
                    X[i], x_min=self.x_min, x_max=self.x_max, r=r,
                    size=self.sample_size)
                tensor_x_rng = torch.tensor(x_rng, dtype=torch.float32)
                tensor_preds_rng = self.__predict(tensor_x_rng)
                preds_rng = tensor_preds_rng.cpu().detach().numpy()
                prob = np.bincount(preds_rng, minlength=self.n_classes).astype(np.float32)
                prob = prob / np.sum(prob)
                probabilities[i] = prob

        return probabilities

    def score(self, X, y, r=None):
        if r is None:
            r = self.r
        pred = self.predict(X, r)
        acc = np.mean(pred == y)
        return acc

    def __predict(self, tensor_X):
        n = len(tensor_X)
        dataset = TensorDataset(tensor_X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        tensor_predictions = -torch.ones(n, dtype=torch.long)
        start = 0

        for x in loader:
            x = x[0].to(self.device)
            n = x.size(0)
            end = start + n
            outputs = self.model(x)
            pred = outputs.max(1)[1].type(torch.long)
            tensor_predictions[start:end] = pred
            start += n

        return tensor_predictions


class SklearnRegionBasedClassifier:
    """Region Based Classifier for robust classification. (scikit-learn version)

    This classifier is used to restore the true label of adversarial examples.

    Parameters
    ----------
    model : torch.nn.Module object, default=None
        The classifier.

    r : float, default=0.2
        The radius from the sample.

    sample_size : int, default=1000
        The number of samples generated within the radius.

    n_classes : int, default=10
        The number of output classes.

    x_min : float or array, default=0.0

    x_max : float or array, default=1.0

    r0 : float, default=0.0
        The initial radius for the binary search.

    step_size : float, default=0.01
            Step size for each iteration.

    stop_value : float, default=None
        Maximum searching radius.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, model=None, r=0.2, sample_size=1000, n_classes=10,
                 x_min=0.0, x_max=1.0, r0=0.0, step_size=0.01,
                 stop_value=0.4, device='gpu'):
        self.model = model
        self.r = r
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.x_min = x_min
        self.x_max = x_max
        self.r0 = r0
        self.step_size = step_size
        self.stop_value = stop_value
        self.device = device

    def search_r(self, X, y, r0=0.0, step_size=0.01, stop=0.4, update=True,
                 verbose=0, exit_early=True):
        r = r0
        time_start = time.time()
        pred_pt = self.model.predict(X)
        acc_pt = np.mean(pred_pt == y)
        acc_region = self.score(X, y, r=r)
        results = [[r, acc_region]]

        if verbose > 0:
            print('Accuracy on validation set: {:.4f}'.format(acc_pt))

        # If stop is not None, keep running until reaches stop value.
        if stop is None:
            stop = r0

        r_best = r0
        acc_best = 0.0
        while acc_region >= acc_pt or r <= stop:
            time_elapsed = time.time() - time_start
            if verbose > 0:
                print('[{:s}] Accuracy on region-based: {:.4f}, r: {:.2f}'.format(
                    str(datetime.timedelta(seconds=time_elapsed)), acc_region, r))
            time_start = time.time()
            acc_region = self.score(X, y, r=r)
            results.append([r, acc_region])
            if acc_best <= acc_region:
                acc_best = acc_region
                r_best = r
            if exit_early and acc_best > acc_region:
                break
            r += step_size

        r_best = r
        for res in reversed(results):
            if res[1] >= acc_pt:
                break
            r_best = res[0]

        if update:
            self.r = r_best

        return r_best, np.array(results, dtype=np.float32)

    def fit(self, X=None, y=None):
        """Region-based classifier does not require fit."""
        return self

    def search_thresholds(self, X, y=None, labels_adv=None, verbose=1):
        # find all indices are not adversarial examples
        indices = np.where(labels_adv == 0)[0]
        X = X[indices]
        y = y[indices]
        r_best, _ = self.search_r(
            X,
            y,
            r0=self.r0,
            step_size=self.step_size,
            stop=self.stop_value,
            update=True,
            verbose=verbose)
        print('Best r =', r_best)
        return r_best

    def predict(self, X, r=None):
        """Predicts class labels for samples in X."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        if r is None:
            r = self.r

        n = X.shape[0]
        pred = -np.ones(n, dtype=np.long)

        for i in range(n):
            x_rng = generate_random_samples(
                X[i],
                x_min=self.x_min,
                x_max=self.x_max,
                r=r,
                size=self.sample_size)
            pred_rng = self.model.predict(x_rng)
            pred[i] = np.bincount(pred_rng).argmax()
        return pred

    def detect(self, X, y=None):
        return self.predict(X)

    def predict_proba(self, X, r=None):
        """Returns probability estimates."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        if r is None:
            r = self.r

        n = X.shape[0]
        probabilities = np.zeros((n, self.n_classes), dtype=np.float32)

        for i in range(n):
            x_rng = generate_random_samples(
                X[i],
                x_min=self.x_min,
                x_max=self.x_max,
                r=r,
                size=self.sample_size)
            pred_rng = self.model.predict(x_rng)
            prob = np.bincount(pred_rng).astype(np.float32)
            prob = prob / np.sum(prob)
            probabilities[i] = prob
        return probabilities

    def score(self, X, y, r=None):
        if r is None:
            r = self.r
        pred = self.predict(X, r)
        acc = np.mean(pred == y)
        return acc
