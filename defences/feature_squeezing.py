import abc
import copy
import datetime
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


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

    def __init__(self, x_min=0.0, x_max=1.0, noise_strength=0.025, std=1.0):
        super().__init__('gaussian', x_min, x_max)
        self.noise_strength = noise_strength
        self.std = std

    def transform(self, X):
        noise = np.random.normal(0, scale=self.std, size=X.shape)
        X_transformed = X + self.noise_strength * noise
        return np.clip(X_transformed, self.x_min, self.x_max)


class MedianSqueezer(Squeezer):
    """Median Filter Squeezer"""

    def __init__(self, x_min=0.0, x_max=1.0, kernel_size=2):
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

    def __init__(self, x_min=0.0, x_max=1.0, bit_depth=4):
        super().__init__('depth', x_min, x_max)
        self.bit_depth = bit_depth

    def transform(self, X):
        max_val = np.rint(2 ** self.bit_depth - 1)
        X_transformed = np.rint(X * max_val) / max_val
        X_transformed = X_transformed * (self.x_max - self.x_min)
        X_transformed += self.x_min
        return np.clip(X_transformed, self.x_min, self.x_max)


class NLMeansColourSqueezer(Squeezer):
    """OpenCV FastNLMeansDenoisingColored Squeezer"""

    def __init__(self, x_min=0.0, x_max=1.0, h=2, templateWindowsSize=3, searchWindowSize=13):
        super().__init__('NLMeans', x_min, x_max)
        self.h = h
        self.templateWindowsSize = templateWindowsSize
        self.searchWindowSize = searchWindowSize

    def transform(self, X):
        X = np.moveaxis(X, 1, -1)
        outputs = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            img = (X[i].copy() * 255.0).astype('uint8')
            outputs[i] = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=self.h,
                hColor=self.h,
                templateWindowSize=self.templateWindowsSize,
                searchWindowSize=self.searchWindowSize)
        outputs = np.moveaxis(outputs, -1, 1) / 255.0
        return np.clip(outputs, self.x_min, self.x_max)


class FeatureSqueezingTorch:
    def __init__(self, *, classifier=None, lr=0.001, momentum=0.9,
                 weight_decay=5e-4, loss=nn.CrossEntropyLoss(), batch_size=128,
                 x_min=0.0, x_max=1.0, squeezers=[], n_classes=10, fpr=0.05, device='cuda'):
        self.classifier = classifier
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss = loss
        self.batch_size = batch_size
        self.x_min = x_min
        self.x_max = x_max
        self.squeezers = squeezers
        self.n_classes = n_classes
        self.fpr = fpr
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
        squeezed_data = self.__get_squeezed_data(X, verbose)
        for i in range(len(self.squeezers)):
            squeezer = self.squeezers[i]
            if verbose > 0:
                print('Training {}...'.format(squeezer.name))
            model = self.squeezed_models_[i]
            samples = squeezed_data[i]
            optimizer = SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
            losses = []
            dataset = TensorDataset(
                torch.from_numpy(samples.astype(np.float32)),
                torch.from_numpy(y.astype(np.long)))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for e in range(epochs):
                time_start = time.time()
                current_loss, accuracy = self.__train(
                    loader, model, self.loss, optimizer)
                scheduler.step()
                losses.append(current_loss)
                time_elapsed = time.time() - time_start

                if verbose > 0:
                    print('{:2d}/{:2d} [{:s}] Squeezer: {:s} Train loss: {:.4f} acc: {:.4f}%'.format(
                        e + 1, epochs,
                        str(datetime.timedelta(seconds=time_elapsed)),
                        squeezer.name,
                        current_loss,
                        accuracy * 100))
                if accuracy >= 0.9999 and e >= 10:
                    print('Training set is converged at:', e)
                    break
            self.__history_losses[i] = losses
        return self

    def search_thresholds(self, X, y=None, labels_adv=None):
        """Train a logistic regression model to find the threshold"""
        idx = np.where(labels_adv == 0)[0]  # Only clean images
        l1_scores = self.get_l1_score(X[idx])
        self.threshold_ = np.quantile(l1_scores, 1 - self.fpr)
        return self.threshold_

    def get_l1_score(self, X):
        n_squeezer = len(self.squeezers)
        squeezed_data = self.__get_squeezed_data(X)
        outputs_squeezed = np.zeros((n_squeezer, X.shape[0], self.n_classes), dtype=np.float32)

        for i in range(n_squeezer):
            dataset = TensorDataset(torch.from_numpy(squeezed_data[i].astype(np.float32)))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            model = self.squeezed_models_[i]
            outputs_squeezed[i] = self.__predict(loader, model)

        # Also use the original classifier.
        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        initial_outputs = self.__predict(loader, self.classifier)
        l1 = np.sum(np.abs(outputs_squeezed - initial_outputs), axis=2)
        score = np.max(l1, axis=0)
        return np.transpose(score)

    def predict(self, X):
        l1_scores = self.get_l1_score(X)
        return l1_scores > self.threshold_

    def detect(self, X, y=None):
        return self.predict(X)

    def predict_proba(self, X):
        l1_scores = self.get_l1_score(X)
        return l1_scores

    def save(self, path):
        data = []
        for model in self.squeezed_models_:
            data.append(model.state_dict())
        torch.save(data, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(len(self.squeezed_models_)):
            self.squeezed_models_[i].load_state_dict(checkpoint[i])

    def __get_squeezed_data(self, X, verbose=0):
        squeezed_data = []
        for squeezer in self.squeezers:
            if verbose > 0:
                print('Transforming {}...'.format(squeezer.name))
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
            pred = outputs.max(1, keepdim=True)[1]
            corrects += pred.eq(y.view_as(pred)).sum().item()
        total_loss = total_loss / n
        accuracy = corrects / n
        return total_loss, accuracy

    def __predict(self, loader, model):
        model.eval()
        tensor_pred = torch.zeros((len(loader.dataset), self.n_classes), dtype=torch.float32)
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
