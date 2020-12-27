import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Autoencoder1(nn.Module):
    def __init__(self, n_channel=1):
        super(Autoencoder1, self).__init__()
        self.n_channel = n_channel
        self.conv1 = nn.Conv2d(self.n_channel, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv5 = nn.Conv2d(3, self.n_channel, 3, padding=1)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))
        return x


class Autoencoder2(nn.Module):
    def __init__(self, n_channel=1):
        super(Autoencoder2, self).__init__()
        self.n_channel = n_channel
        self.conv1 = nn.Conv2d(self.n_channel, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(3, self.n_channel, 3, padding=1)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


def torch_add_noise(X, x_min, x_max, epsilon, device='cpu'):
    """Returns X with Gaussian noise and clip."""
    normal = torch.distributions.normal.Normal(
        loc=torch.zeros(X.size(), dtype=torch.float32),
        scale=1.0)
    noise = normal.sample().to(device)
    X_noisy = X + epsilon * noise
    X_noisy = torch.clamp(X_noisy, x_min, x_max)
    return X_noisy


class MagNetDetector():
    def __init__(self, *, model=None, lr=0.001, batch_size=256, regularization=1e-9,
                 noise_strength=0.025, norm='l2', device='cpu'):
        if norm not in ['l1', 'l2']:
            raise ValueError('Norm can only be either l1 or l2.')

        self.model = model
        self.lr = lr
        self.batch_size = batch_size,
        self.regularization = regularization
        self.noise_strength = noise_strength
        self.norm = norm
        self.device = device

    def fit(self, X, y, epochs=100, disable_progress_bar=True):
        pass

    def search_threshold(self, X, y, fp=0.001):
        pass

    def predict(self, X):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class MagNetNoiseReformer():
    def __init__(self, noise_strength, device):
        self.noise_strength = noise_strength
        self.device = device

    def reform(self, X, x_min, x_max):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            X = torch_add_noise(
                X, x_min, x_max, self.noise_strength, self.device)
            X = X.cpu().detach().numpy()
        elif isinstance(X, torch.Tensor):
            X = torch_add_noise(
                X, x_min, x_max, self.noise_strength, self.device)
        else:
            raise ValueError('X must be either a ndarray or a Tensor.')
        return X


class MagNetAutoencoderReformer():
    def __init__(self, model, batch_size=512, device='cpu'):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def reform(self, X):
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X)
        elif isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            raise ValueError('X must be either a ndarray or a Tensor.')
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        X_ae = torch.zeros_like(X_tensor)
        self.model = self.model.to(self.device)
        self.model.eval()
        start = 0
        for x in loader:
            x = x.to(self.device)
            end = start + x.size(0)
            X_ae[start:end] = self.model(x)
            start += end
        if isinstance(X, np.ndarray):
            X_ae = X_ae.cpu().detach().numpy()
        return X_ae
