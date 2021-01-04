import abc
import copy
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


def majority_vote(predictions, n_class):
    """Get predictions by majority votes"""
    counts = np.array([np.bincount(prediction, minlength=n_class)
                       for prediction in np.transpose(predictions)])
    return np.argmax(counts, axis=1)


class Squeezer(abc.ABC):
    """Base class for squeezers."""

    def __init__(self, name, x_min, x_max):
        self.name
        self.x_min = x_min
        self.x_max = x_max

    @abc.abstractmethod
    def transform(self, X):
        raise NotImplementedError


class GaussianSqueezer(Squeezer):
    """Gaussian Noise Squeezer"""

    def __init__(self, name, x_min, x_max, noise_strength=0.025, std=1.0):
        super().__init__(name, x_min, x_max)
        self.noise_strength = noise_strength
        self.std = std

    def transform(self, X):
        noise = np.random.normal(0, scale=self.std, size=X.shape)
        X_transformed = X + self.noise_strength * noise
        return np.clip(X_transformed, self.x_min, self.x_max)


class MedianSqueezer(Squeezer):
    """Median Filter Squeezer"""

    def __init__(self, name, x_min, x_max, kernel_size):
        super().__init__(name, x_min, x_max)
        self.kernel_size = kernel_size

    def transform(self, X):
        X_transformed = np.zeros_like(X, dtype=np.float32)
        for i in range(len(X)):
            X_transformed[i] = ndimage.median_filter(
                X[i], size=self.kernel_size)
        return np.clip(X_transformed, self.x_min, self.x_max)


class DepthSqueezer(Squeezer):
    """Bit Depth Squeezer"""

    def __init__(self, name, x_min, x_max, bit_depth):
        super().__init__(name, x_min, x_max)
        self.bit_depth = bit_depth

    def transform(self, X):
        max_val = np.rint(2 ** self.bit_depth - 1)
        X_transformed = np.rint(X * max_val) / max_val
        X_transformed = X_transformed * (self.x_max - self.x_min)
        X_transformed += self.x_min
        return np.clip(X_transformed, self.x_min, self.x_max)


class FeatureSqueezingTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, *, classifier=None, lr=0.001, momentum=0.9,
                 loss=nn.MSELoss(), batch_size=128, x_min=0.0, x_max=1.0,
                 squeezers=[], n_class=10, device='cuda'):
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

        self.squeezed_models = []
        self.history_losses = []
        for s in squeezers:
            self.squeezed_models.append(copy.deepcopy(classifier))
            self.history_losses.append([])

    def fit(self, X, y, epochs=50, verbose=0):
        squeezed_data = self.__get_squeezed_data(X)
        for i in range(len(self.squeezers)):
            model = self.squeezed_models[i]
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
                    print('{:2d}/{:2d}[{:s}] Train loss: {:.4f} acc: {:.4f}%'.format(
                        e+1, epochs,
                        str(datetime.timedelta(seconds=time_elapsed)),
                        current_loss,
                        accuracy*100))
            self.history_losses[i] = losses

    def predict(self, X):
        n_squeezer = len(self.squeezers)
        n_samples = len(X)
        squeezed_data = self.__get_squeezed_data(X)
        squeezed_preditions = -np.zeros((n_squeezer, n_samples), dtype=np.long)

        for i in range(n_squeezer):
            dataset = TensorDataset(
                torch.from_numpy(squeezed_data[i].astype(np.float32)))
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)
            model = self.squeezed_models[i]
            squeezed_preditions[i] = self.__predict(loader, model)

        return majority_vote(squeezed_preditions, self.n_class)

    def score(self, X, y):
        n = y.shape[0]
        predictions = self.predict(X)
        accuracy = np.sum(np.equal(predictions, y)) / float(n)
        return accuracy

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
        tensor_pred = -torch.ones(len(loader.dataset), dtype=torch.long)
        start = 0
        with torch.no_grad():
            for mini_batch in loader:
                x = mini_batch[0].to(self.device)
                n = x.size(0)
                end = start + n
                outputs = model(x)
                tensor_pred[start:end] = outputs.max(1)[1].type(torch.int64)
                start = end
        return tensor_pred


class FeatureSqueezingSklearn(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y, epochs=50):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass
