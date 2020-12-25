"""
Region-based classification using PyTorch.
"""
import logging

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils import DataLoader, TensorDataset

from util import generate_random_samples

logger = logging.getLogger(__name__)


class RegionBasedClassifier(BaseEstimator, ClassifierMixin):
    """Region Based Classifier for robust classification. This classifier is 
    used to restore the true label of adversarial examples.
    """

    def __init__(self, *, model=None, r=0.2, sample_size=1000, n_class=10,
                 clip_values=None, batch_size=128, device='gpu'):
        self.model = model
        self.r = r
        self.sample_size = sample_size
        self.n_class = n_class
        self.clip_values = clip_values
        self.batch_size = batch_size
        self.device = device

    def search_r(self, X, y, r0=0, step_size=0.01, update=True):
        r = r0
        n = len(X)
        tensor_X = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        tensor_predictions_point = self.__predict(tensor_X)
        corrects = tensor_predictions_point.eq(
            tensor_y.view_as(tensor_predictions_point)).sum().item()
        acc_point = corrects / float(n)
        acc_region = self.score(X, y, r=r)

        while acc_region >= acc_point:
            r += step_size
            acc_region = self.score(X, y, r=r)

        if update:
            self.r = r - step_size
        return self.r

    def fit(self, X=None, y=None):
        """Region-based classifier does not require fit."""
        logger.info('Region-based classifier does not require retraining.')
        return self

    def predict(self, X, r=None):
        """Predicts class labels for samples in X."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        if r is None:
            r = self.r
        n = X.shape[0]
        predictions = -np.ones(n, dtype=np.long)

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                x_rng = generate_random_samples(
                    X[i], clip_values=self.clip_values, r=r,
                    size=self.sample_size)
                tensor_x_rng = torch.tensor(x_rng, dtype=torch.float32)
                tensor_preds_rng = self.__predict(tensor_x_rng)
                preds_rng = tensor_preds_rng.cpu().detach().numpy()
                predictions[i] = np.bincount(preds_rng).argmax()

        return predictions

    def predict_proba(self, X):
        """Returns probability estimates."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        n = X.shape[0]
        probabilities = np.zeros((n, self.n_class), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                x_rng = generate_random_samples(
                    X[i], clip_values=self.clip_values, r=self.r,
                    size=self.sample_size)
                tensor_x_rng = torch.tensor(x_rng, dtype=torch.float32)
                tensor_preds_rng = self.__predict(tensor_x_rng)
                preds_rng = tensor_preds_rng.cpu().detach().numpy()
                prob = np.bincount(preds_rng).astype(np.float32)
                prob = prob / float(np.sum(prob))
                probabilities[i] = prob

        return probabilities

    def score(self, X, y, r=None):
        """Returns the accuracy score."""
        n = X.shape[0]
        preds = self.predict(X, r=r)
        accuracy = np.sum(preds == y) / float(n)
        return accuracy

    def __predict(self, tensor_X):
        dataset = TensorDataset(tensor_X)
        loader = DataLoader(dataset, batchsize=self.batch_size, shuffle=False)
        tensor_predictions = -torch.ones(len(tensor_X), dtype=torch.long)
        start = 0

        for x in loader:
            x = x[0].to(self.device)
            n = x.size(0)
            end = start + n
            outputs = self.model(x)
            preds = outputs.max(1)[1].type(torch.long)
            tensor_predictions[start:end] = preds
            start += n

        return tensor_predictions
