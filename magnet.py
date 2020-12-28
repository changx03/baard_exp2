import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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


def get_m(p, q):
    """Average 2 probabilities."""
    return 0.5 * (p + q)


def kl_divergence(p, q):
    """Returns Kullback–Leibler divergence between p and q, D_KL(P||Q)"""
    return np.sum(p * np.log(p/q), axis=1)


def js_divergence(p, q):
    """Returns Jensen–Shannon divergence between p and q, D_JS(P||Q)"""
    m = get_m(p, q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def smooth_softmax(X, t):
    """The smoother softmax function in distillation network."""
    X_exp = torch.exp(X/t)
    return X_exp / torch.sum(X_exp, 1).view(X.size(0), 1)


class MagNetDetector():
    """MagNet Detector which supports PyTorch.

    This implements a single detector in MagNet framework.

    Parameters
    ----------
    encoder : torch.nn.Module object
        An denoising autoencoder. The outputs must be the same shape as the 
        input.

    classifier : torch.nn.Module object, default=None
        The classifier is only required when using probability divergence based
        detector. This classifier must implement 'before_softmax' method.

    lr : float, default=0.001
        Learning rate for training the autoencoder.

    batch_size : int, default=256
        Mini batch size for training the autoencoder.

    weight_decay : float, default=1e-9
        Weight decay coefficient for AdamW optimizer.

    x_min : float or array, default=0.0

    x_max : float or array, default=1.0

    noise_strength : float, default=0.025
        the strength of noise level. In range [0, 1). When noise is 0, no noise
        will add during training.

    algorithm : {'error', 'prob'}, default='error'
        - 'error' will use reconstruction error to compute threshold.
        - 'prob' will use probability divergence to compute threshold.

    p : {1, 2, None}, default=1
        P-norm for computing the reconstruction error. Only used when 
        algorithm='error'.

    temperature : float, default=10
        The temperature parameter, T, in smoother softmax function. 
        Only used when algorithm='prob. temperature >= 1.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, *, encoder=None, classifier=None, lr=0.001,
                 batch_size=256, weight_decay=1e-9, x_min=0.0, x_max=1.0,
                 noise_strength=0.025, algorithm='error', p=1, temperature=10.0,
                 device='cpu'):
        self.encoder = encoder
        self.classifier = classifier
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.x_min = x_min
        self.x_max = x_max
        self.noise_strength = noise_strength
        self.algorithm = algorithm
        self.p = p
        self.temperature = temperature
        self.device = device

        self.history_train_loss = []
        self.encoder = self.encoder.to(device)
        self.threshold = None

    def fit(self, X, y=None, epochs=100, disable_progress_bar=True):
        """Fits the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Dummy variable.

        epochs : int, default=100
            Number of epochs to train.

        disable_progress_bar : bool, default=True
            Show progress bar.

        Returns
        -------
        self : object
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        X_tensor = torch.from_numpy(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW(self.encoder.parameters(),
                          lr=self.lr,
                          weight_decay=self.weight_decay)
        loss = nn.MSELoss()
        temp_train_loss = np.zeros(epochs, dtype=np.float32)

        for e in tqdm(range(epochs), disable=disable_progress_bar):
            temp_train_loss[e] = self.__train(loader, loss, optimizer)

        self.history_train_loss += temp_train_loss.tolist()
        return self

    def predict(self, X):
        """Returns reconstructed samples via the autoencoder."""
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        X_tensor = torch.from_numpy(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        outputs = torch.zeros_like(X_tensor)
        self.encoder.eval()

        start = 0
        with torch.no_grad():
            for x in loader:
                x = x[0].to(self.device)
                end = start + x.size(0)
                outputs[start:end] = self.encoder(x).cpu()
                start = end
        return outputs.detach().numpy()

    def score(self, X, y=None):
        """Returns the MSE loss between X and reconstructed samples."""
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        X_tensor = torch.from_numpy(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        loss = nn.MSELoss()
        total_loss = self.__validate(loader, loss)
        return total_loss

    def search_threshold(self, X, fp=0.001, update=True):
        """Returns the threshold based on false positive rate."""
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        X_ae = self.predict(X)
        if self.algorithm == 'error':
            diff = np.abs(X - X_ae)
            diff = diff.reshape(diff.shape[0], -1)
            scores = np.mean(np.power(diff, self.p), axis=1)
        else:  # self.algorithm == 'prob'
            scores = self.__get_js_divergence(
                torch.from_numpy(X).type(torch.float32),
                torch.from_numpy(X_ae).type(torch.float32))
        index = int(np.round((1-fp) * len(X)))
        threshold = np.sort(scores)[index]
        if update:
            self.threshold = threshold
        return threshold

    def detect(self, X):
        """Returns binary labels with adversarial examples are labled as 1, 
        0 otherwise.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        n = X.shape[0]
        X_ae = self.predict(X)
        if self.algorithm == 'error':
            X = X.reshape(n, -1)
            X_ae = X_ae.reshape(n, -1)
            diff = np.abs(X - X_ae)
            scores = np.mean(np.power(diff, self.p), axis=1)
        else:  # self.algorithm == 'prob'
            scores = self.__get_js_divergence(
                torch.from_numpy(X).type(torch.float32),
                torch.from_numpy(X_ae).type(torch.float32))
        labels = np.zeros(n, dtype=np.bool)
        indices = np.where(scores > self.threshold)[0]
        labels[indices] = 1
        return labels

    def save(self, path):
        """Save parameters"""
        data = {
            'lr': self.lr,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'noise_strength': self.noise_strength,
            'algorithm': self.algorithm,
            'p': self.p,
            'temperature': self.temperature,
            'history_train_loss': self.history_train_loss,
            'threshold': self.threshold,
            'encoder_state_dict': self.encoder.state_dict(),
        }
        if self.algorithm == 'prob' and self.classifier is not None:
            data['classifier_state_dict'] = self.classifier.state_dict()
        torch.save(data, path)

    def load(self, path):
        """Load parameters"""
        checkpoint = torch.load(path)
        self.lr = checkpoint['lr']
        self.batch_size = checkpoint['batch_size']
        self.weight_decay = checkpoint['weight_decay']
        self.x_min = checkpoint['x_min']
        self.x_max = checkpoint['x_max']
        self.noise_strength = checkpoint['noise_strength']
        self.algorithm = checkpoint['algorithm']
        self.p = checkpoint['p']
        self.temperature = checkpoint['temperature']
        self.history_train_loss = checkpoint['history_train_loss']
        self.threshold = checkpoint['threshold']

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if self.algorithm == 'prob':
            self.classifier.load_state_dict(
                checkpoint['classifier_state_dict'])

    def __train(self, loader, loss, optimizer):
        n = len(loader.dataset)
        self.encoder.train()
        total_loss = 0.0

        for x in loader:
            x = x[0].to(self.device)
            batch_size = x.size(0)
            if self.noise_strength != 0:
                x_noisy = torch_add_noise(x, self.x_min, self.x_max,
                                          self.noise_strength, self.device)
            else:
                x_noisy = x
            optimizer.zero_grad()
            outputs = self.encoder(x_noisy)
            l = loss(outputs, x)
            l.backward()
            optimizer.step()
            total_loss += l.item() * batch_size
        total_loss = total_loss / float(n)
        return total_loss

    def __validate(self, loader, loss):
        n = len(loader.dataset)
        self.encoder.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x in loader:
                x = x[0].to(self.device)
                batch_size = x.size(0)
                outputs = self.encoder(x)
                l = loss(outputs, x)
                total_loss += l.item() * batch_size
        total_loss = total_loss / float(n)
        return total_loss

    def __get_js_divergence(self, A, B):
        self.classifier.to(self.device)
        self.classifier.eval()
        before_softmax = self.classifier.before_softmax
        dataset = TensorDataset(A, B)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            output = self.classifier(A[:1].to(self.device))
            n_classes = output.size(1)
            shape_output = (A.size(0), n_classes)
            prob_A = torch.zeros(shape_output, dtype=torch.float32)
            prob_B = torch.zeros(shape_output, dtype=torch.float32)
            start = 0
            for a, b in loader:
                end = start + a.size(0)
                a = a.to(self.device)
                b = b.to(self.device)
                prob_A[start:end] = smooth_softmax(
                    before_softmax(a), self.temperature).cpu()
                prob_B[start:end] = smooth_softmax(
                    before_softmax(b), self.temperature).cpu()
                start = end
        prob_A = prob_A.detach().numpy()
        prob_B = prob_B.detach().numpy()
        diff = js_divergence(prob_A, prob_B)
        return diff


class MagNetNoiseReformer():
    """MagNet Noise-based reformer"""

    def __init__(self, noise_strength=0.025, device='cpu'):
        self.noise_strength = noise_strength
        self.device = device

    def reform(self, X, x_min, x_max):
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        X_tensor = torch.from_numpy(X)
        X_noisy = torch_add_noise(
            X_tensor, x_min, x_max, self.noise_strength, self.device)
        X_noisy = X_noisy.cpu().detach().numpy()
        return X_noisy


class MagNetAutoencoderReformer():
    """MagNet Autoencoder-based reformer"""

    def __init__(self, encoder, batch_size=512, device='cpu'):
        self.encoder = encoder
        self.batch_size = batch_size
        self.device = device

        self.encoder = self.encoder.to(self.device)

    def reform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a ndarray.')

        X_tensor = torch.from_numpy(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        X_ae = torch.zeros_like(X_tensor)
        self.encoder.eval()

        start = 0
        for x in loader:
            x = x.to(self.device)
            end = start + x.size(0)
            X_ae[start:end] = self.encoder(x).cpu()
            start += end
        if isinstance(X, np.ndarray):
            X_ae = X_ae.detach().numpy()
        return X_ae


class MagNetOperator():
    """MageNet framework. It combines multiple detectors and one reformer.

    Parameters
    ----------
    classifier : torch.nn.Module object
        The classifier is only required when using probability divergence based
        detector.

    detectors : array of MagNetDetector

    reformer : {MagNetNoiseReformer, MagNetAutoencoderReformer}

    batch_size : int, default=512
        Number of samples in each batch.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, classifier, detectors, reformer, batch_size=512,
                 device='cpu'):
        self.classifier = classifier
        self.detectors = detectors
        self.reformer = reformer
        self.batch_size = batch_size
        self.device = device

    def detect(self, X):
        """Testing samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        X_reformed : array-like of shape (n_samples, n_features)
            Same shape as X. Returns all reformed samples.

        labels : array of shape (n_samples,)
            Returns labels for adversarial examples. 1 is adversarial example, 
            0 is benign.
        """
        n = len(X)
        labels = np.zeros(n, dtype=np.bool)
        # Go through all detectors
        for detector in self.detectors:
            result = detector.detect(X)
            labels = np.logical_or(labels, result)
        # Reform all samples in X
        X_reformed = self.reformer.reform(X)
        return X_reformed, labels

    def score(self, X, y_label, y_adv):
        """Rate of success. The success means (1) correctly blocked by detector.
        (2) Failed blocked by detector, but the reformed sample is correctly 
        classified by the original classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y_label : array-like of shape (n_samples, )
            Target labels.

        y_adv : array-like of shape (n_samples, )
            Target adversarial labels. 1 is adversarial example, 0 is benign.

        Returns
        -------
        success_rate : float
            The fraction of correctly classified samples.
        """
        # Get reformed samples and the samples which are blocked by detectors.
        X_reformed, blocked_labels = self.detect(X)
        matched_adv = blocked_labels == y_adv
        # 1 is adversarial example, 0 is benign sample.
        uncertain_indices = np.where(matched_adv != True)[0]
        predictions = self.__predict(X_reformed[uncertain_indices])
        matched_label = y_label[uncertain_indices] == predictions
        # Count correctly detected and correctly predicted after reformed.
        total_correct = np.sum(matched_adv) + np.sum(matched_label)
        success_rate = total_correct / float(len(X))
        return success_rate

    def __predict(self, X):
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        tensor_pred = -torch.ones(len(X), dtype=torch.int64)

        start = 0
        with torch.no_grad():
            for x in loader:
                n = x.size(0)
                x = x[0].to(self.device)
                end = start + n
                outputs = self.classifier(x).cpu()
                tensor_pred[start:end] = outputs.max(1)[1].type(torch.int64)
                start = end
        return tensor_pred.detach().numpy()
