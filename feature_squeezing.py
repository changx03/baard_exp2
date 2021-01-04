import abc
import copy
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from sklearn.preprocessing import MinMaxScaler


class Squeezer(abc.ABC):
    """Base class for squeezers."""

    def __init__(self, name, x_min, x_max):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max

    @abc.abstractmethod
    def transform(self, X):
        raise NotImplementedError


class GaussianSqueezer(Squeezer):
    """Gaussian Noise Squeezer"""

    def __init__(self, x_min, x_max, noise_strength=0.025, std=1.0):
        super().__init__('gaussian', x_min, x_max)
        self.noise_strength = noise_strength
        self.std = std

    def transform(self, X):
        noise = np.random.normal(0, scale=self.std, size=X.shape)
        X_transformed = X + self.noise_strength * noise
        return np.clip(X_transformed, self.x_min, self.x_max)


class MedianSqueezer(Squeezer):
    """Median Filter Squeezer"""

    def __init__(self, x_min, x_max, kernel_size=3):
        super().__init__('median', x_min, x_max)
        self.kernel_size = kernel_size

    def transform(self, X):
        X_transformed = np.zeros_like(X, dtype=np.float32)
        for i in range(len(X)):
            X_transformed[i] = ndimage.median_filter(
                X[i], size=self.kernel_size)
        return np.clip(X_transformed, self.x_min, self.x_max)


class DepthSqueezer(Squeezer):
    """Bit Depth Squeezer"""

    def __init__(self, x_min, x_max, bit_depth=8):
        super().__init__('depth', x_min, x_max)
        self.bit_depth = bit_depth

    def transform(self, X):
        max_val = np.rint(2 ** self.bit_depth - 1)
        X_transformed = np.rint(X * max_val) / max_val
        X_transformed = X_transformed * (self.x_max - self.x_min)
        X_transformed += self.x_min
        return np.clip(X_transformed, self.x_min, self.x_max)


class FeatureSqueezingTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, *, classifier=None, lr=0.001, momentum=0.9,
                 loss=nn.CrossEntropyLoss(), batch_size=128, x_min=0.0,
                 x_max=1.0, squeezers=[], n_class=10, device='cuda'):
        self.classifier = classifier
        self.lr = lr
        self.momentum = momentum
        self.loss = loss
        self.batch_size = batch_size
        self.x_min = x_min
        self.x_max = x_max
        self.squeezers = squeezers
        self.n_class = n_class
        self.device = device

        self.squeezed_models_ = []
        self.__history_losses = []
        for s in squeezers:
            self.squeezed_models_.append(copy.deepcopy(classifier))
            self.__history_losses.append([])

    @property
    def history_losses(self):
        return np.array(self.__history_losses)

    def fit(self, X, y, epochs=50, verbose=0):
        """Train squeezed models"""
        squeezed_data = self.__get_squeezed_data(X)
        for i in range(len(self.squeezers)):
            squeezer = self.squeezers[i]
            model = self.squeezed_models_[i]
            samples = squeezed_data[i]
            optimizer = SGD(model.parameters(), lr=self.lr,
                            momentum=self.momentum)
            losses = []
            dataset = TensorDataset(
                torch.from_numpy(samples.astype(np.float32)),
                torch.from_numpy(y.astype(np.long)))
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)

            for e in range(epochs):
                time_start = time.time()
                current_loss, accuracy = self.__train(
                    loader, model, self.loss, optimizer)
                losses.append(current_loss)
                time_elapsed = time.time() - time_start

                if verbose > 0:
                    print('{:2d}/{:2d} [{:s}] Squeezer: {:s} Train loss: {:.4f} acc: {:.4f}%'.format(
                        e+1, epochs,
                        str(datetime.timedelta(seconds=time_elapsed)),
                        squeezer.name,
                        current_loss,
                        accuracy*100))
            self.__history_losses[i] = losses

    def search_threshold(self, X, y_adv):
        """Train a logistic regression model to find the threshold"""
        l1_scores = self.get_l1_score(X)
        self.scaler_ = MinMaxScaler().fit(l1_scores)
        characteristics = self.scaler_.transform(l1_scores)
        self.detector_ = LogisticRegressionCV(cv=5)
        self.detector_.fit(characteristics, y_adv)

    def get_l1_score(self, X):
        n_squeezer = len(self.squeezers)
        n_samples = len(X)
        squeezed_data = self.__get_squeezed_data(X)
        outputs_squeezed = np.zeros(
            (n_squeezer, n_samples, self.n_class), dtype=np.long)

        for i in range(n_squeezer):
            dataset = TensorDataset(
                torch.from_numpy(squeezed_data[i].astype(np.float32)))
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False)
            model = self.squeezed_models_[i]
            outputs_squeezed[i] = self.__predict(loader, model)

        # Also use the original classifier.
        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        initial_outputs = self.__predict(loader, self.classifier)
        l1 = np.sum(np.abs(outputs_squeezed - initial_outputs), axis=2)
        return np.transpose(l1)

    def predict(self, X):
        l1_scores = self.get_l1_score(X)
        characteristics = self.scaler_.transform(l1_scores)
        return self.detector_.predict(characteristics)

    def predict_proba(self, X):
        l1_scores = self.get_l1_score(X)
        characteristics = self.scaler_.transform(l1_scores)
        return self.detector_.predict_proba(characteristics)

    def score(self, X, y):
        """Returns the ROC AUC score"""
        prob = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, prob)

    def __get_squeezed_data(self, X):
        squeezed_data = []
        for squeezer in self.squeezers:
            X_transformed = squeezer.transform(X)
            squeezed_data.append(X_transformed)
        return np.array(squeezed_data, dtype=np.float32)

    def __train(self, loader, model, loss, optimizer):
        n = len(loader.dataset)
        model.train()
        total_loss = 0.0
        corrects = 0.0

        for x, y in loader:
            batch_size = x.size(0)
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            outputs = model(x)
            l = loss(outputs, y)
            l.backward()
            optimizer.step()

            total_loss += l.item() * batch_size
            predictions = outputs.max(1, keepdim=True)[1]
            corrects += predictions.eq(y.view_as(predictions)).sum().item()
        total_loss = total_loss / float(n)
        accuracy = corrects / float(n)
        return total_loss, accuracy

    def __predict(self, loader, model):
        model.eval()
        tensor_pred = torch.zeros(
            (len(loader.dataset), self.n_class), dtype=torch.float32)
        start = 0
        with torch.no_grad():
            for mini_batch in loader:
                x = mini_batch[0].to(self.device)
                n = x.size(0)
                end = start + n
                outputs = model(x)
                tensor_pred[start:end] = outputs
                start = end
        return tensor_pred.cpu().detach().numpy()


class FeatureSqueezingSklearn(BaseEstimator, ClassifierMixin):
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y, epochs=50):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
    
    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError
