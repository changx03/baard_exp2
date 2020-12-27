import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from util import merge_and_generate_labels


def get_hidden_layers(sequence, device):
    """Returns a list of functions to compute the outpus in each hidden layer.
    """
    n_hidden_layers = len(list(sequence.children())) - 1
    # get deep representations
    hidden_layers = []
    for i in range(1, n_hidden_layers+1):
        layer = nn.Sequential(*list(sequence.children())[:i]).to(device)
        hidden_layers.append(layer)
    return hidden_layers, n_hidden_layers


def mle_batch(data, batch, k):
    """Computes Maximum Likelihood Estimator of LID within k nearlest neighbours.
    """
    k = min(k, len(data)-1)

    def mle(v):
        # v[-1] is the max of the neighbour distances
        return - k / np.sum(np.log(v/v[-1]))

    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(mle, axis=1, arr=a)
    return a


def get_lid_random_batch(sequence, X, X_noisy, X_adv, k, batch_size, device,
                         disable_progress_bar=True):
    """Trains the local intrinsic dimensionality (LID) using samples in X and 
    its coresponding noisy and adversarial examples.
    Estimated by k close neighbours in the random batch it lies in.
    """
    hidden_layers, n_hidden_layers = get_hidden_layers(sequence, device)
    # print('Number of hidden layers: {}'.format(len(hidden_layers)))

    # Convert numpy Array to PyTorch Tensor
    indices = np.random.permutation(X.shape[0])
    X = torch.tensor(X[indices], dtype=torch.float32).to(device)
    X_noisy = torch.tensor(X_noisy[indices], dtype=torch.float32).to(device)
    X_adv = torch.tensor(X_adv[indices], dtype=torch.float32).to(device)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch_benign = np.zeros(
            (n_feed, n_hidden_layers), dtype=np.float32)
        lid_batch_noisy = np.zeros((n_feed, n_hidden_layers), dtype=np.float32)
        lid_batch_adv = np.zeros((n_feed, n_hidden_layers), dtype=np.float32)

        for i, layer in enumerate(hidden_layers):
            layer.eval()
            batch_data = torch.cat(
                (X[start: end], X_noisy[start: end], X_adv[start: end]))
            output = layer(batch_data)
            output = output.view(output.size(0), -1).cpu().detach().numpy()

            output_benign = output[:n_feed]
            output_noisy = output[n_feed: n_feed+n_feed]
            output_adv = output[n_feed+n_feed:]

            lid_batch_benign[:, i] = mle_batch(
                output_benign, output_benign, k=k)
            lid_batch_noisy[:, i] = mle_batch(output_benign, output_noisy, k=k)
            lid_batch_adv[:, i] = mle_batch(output_benign, output_adv, k=k)
        return lid_batch_benign, lid_batch_noisy, lid_batch_adv

    lid_benign = []
    lid_adv = []
    lid_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    with torch.no_grad():
        for i_batch in tqdm(range(n_batches), disable=disable_progress_bar):
            batch_benign, batch_noisy, batch_adv = estimate(i_batch)
            lid_benign.extend(batch_benign)
            lid_noisy.extend(batch_noisy)
            lid_adv.extend(batch_adv)

    lid_benign = np.asarray(lid_benign, dtype=np.float32)
    lid_noisy = np.asarray(lid_noisy, dtype=np.float32)
    lid_adv = np.asarray(lid_adv, dtype=np.float32)
    return lid_benign, lid_noisy, lid_adv


def train_lid(sequence, X, X_noisy, X_adv, k=20, batch_size=100, device='cpu',
              disable_progress_bar=True):
    """Gets local intrinsic dimensionality (LID)."""
    lid_benign, lid_noisy, lid_adv = get_lid_random_batch(
        sequence, X, X_noisy, X_adv, k, batch_size, device,
        disable_progress_bar=disable_progress_bar)
    lid_pos = lid_adv
    lid_neg = np.concatenate((lid_benign, lid_noisy))
    artifacts, labels = merge_and_generate_labels(lid_pos, lid_neg)
    return artifacts, labels


def eval_lid(sequence, X_train, X_eval, k=20, batch_size=100,
             device='cpu', disable_progress_bar=True):
    """Evaluates samples in X using LID characteristics.
    TODO: Consider multithreading
    """
    def mle(v):
        return - k / np.sum(np.log(v/v[-1]))

    n_examples = X_eval.shape[0]
    hidden_layers, n_hidden_layers = get_hidden_layers(sequence, device)
    results = np.zeros((n_examples, n_hidden_layers), dtype=np.float32)

    for j in tqdm(range(n_examples), disable=disable_progress_bar):
        x = X_eval[j]
        indices = np.random.choice(
            len(X_train), size=batch_size, replace=False)
        samples = X_train[indices]
        samples = np.concatenate((np.expand_dims(x, axis=0), samples))
        samples = torch.tensor(samples, dtype=torch.float32).to(device)
        single_lid = np.zeros(n_hidden_layers, dtype=np.float32)

        for i, layer in enumerate(hidden_layers):
            layer.eval()
            output = layer(samples)
            output = output.view(output.size(0), -1).cpu().detach().numpy()
            tree = BallTree(output, leaf_size=2)
            dist, _ = tree.query(output[:1], k=k+1)
            dist = np.squeeze(dist, axis=0)[1:]
            single_lid[i] = mle(dist)

        results[j] = single_lid

    return results


def merge_adv_data(X_benign, X_noisy, X_adv):
    """Merges benign, noisy and adversarial examples into one dataset."""
    assert X_benign.shape == X_noisy.shape and X_noisy.shape == X_adv.shape, \
        'All 3 datasets must have same shape'

    n = X_benign.shape[0]
    x_shape = tuple(list(X_benign.shape)[1:])
    output_x_shape = tuple([n, 3] + list(x_shape))
    output = np.zeros(output_x_shape, dtype=np.float32)
    Y = np.repeat([[0, 0, 1]], repeats=n, axis=0).astype(np.long)

    for i in range(n):
        output[i, 0] = X_benign[i]
        output[i, 1] = X_noisy[i]
        output[i, 2] = X_adv[i]

    return output, Y


class LidDetector(BaseEstimator, ClassifierMixin):
    """LID Detector for detecting adversarial examples.
    
    Parameters
    ----------
    model : torch.nn.Sequential
        A PyTorch neural network sequential model.

    k : int
        Number of nearlest neighbours.
    
    batch_size : 100
        Number of random samples in each batch.

    device : {'cpu', 'cuda'}, default='cpu'
    """

    def __init__(self, *, model=None, k=20, batch_size=100, device='cpu'):
        self.model = model
        self.k = k
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y=None, disable_progress_bar=True):
        """Fits the model according to the given training data.
        Expecting each X contains bengin, noisy and adversarial examples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 3, n_features)
            Training vector, Each sample contains benign, noisy and adversarial 
            examples as a tuple. Use the `merge_adv_data` function to create X.

        y : array-like of shape (n_samples, 3), default=None
            Benign, noisy are labeld as 0. Adversarial examples are labeld as 1.

        Returns
        -------
        self : object
        """
        if self.model is None:
            raise ValueError('Model cannot be None.')

        n = X.shape[0]
        single_shape = list(X.shape)[2:]
        X_shape = tuple([n] + list(single_shape))
        X_benign = np.zeros(X_shape, dtype=np.float32)
        X_noisy = np.zeros(X_shape, dtype=np.float32)
        X_adv = np.zeros(X_shape, dtype=np.float32)

        for i in range(n):
            X_benign[i] = X[i, 0]
            X_noisy[i] = X[i, 1]
            X_adv[i] = X[i, 2]

        self.characteristics_, labels = train_lid(
            self.model,
            X=X_benign,
            X_noisy=X_noisy,
            X_adv=X_adv,
            k=self.k,
            batch_size=self.batch_size,
            device=self.device,
            disable_progress_bar=disable_progress_bar
        )
        self.scaler_ = MinMaxScaler().fit(self.characteristics_)
        self.characteristics_ = self.scaler_.transform(self.characteristics_)
        self.X_train_ = X_benign

        self.detector_ = LogisticRegressionCV(cv=5)
        self.detector_.fit(self.characteristics_, labels)

        return self  # Must return the classifier

    def predict(self, X):
        """Predicts class labels for samples in X."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        characteristics = eval_lid(
            self.model,
            X_train=self.X_train_,
            X_eval=X,
            k=self.k,
            batch_size=self.batch_size,
            device=self.device
        )
        characteristics = self.scaler_.transform(characteristics)
        return self.detector_.predict(characteristics)

    def predict_proba(self, X):
        """Returns probability estimates."""
        if self.model is None:
            raise ValueError('Model cannot be None.')

        characteristics = eval_lid(
            self.model,
            X_train=self.X_train_,
            X_eval=X,
            k=self.k,
            batch_size=self.batch_size,
            device=self.device
        )
        characteristics = self.scaler_.transform(characteristics)
        return self.detector_.predict_proba(characteristics)

    def score(self, X, y):
        """Returns the ROC AUC score.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, 3, n_features)
            Training vector, Each sample contains benign, noisy and adversarial 
            examples as a tuple. Use the `merge_adv_data` function to create X.

        y : array-like of shape (n_samples, 3), default=None
            Benign, noisy are labeld as 0. Adversarial examples are labeld as 1.

        Returns
        -------
        score : float
            The ROC AUC score.
        """
        if self.model is None:
            raise ValueError('Model cannot be None.')

        n = X.shape[0]
        single_shape = list(X.shape)[2:]
        X_shape = tuple([n] + list(single_shape))
        X_benign = np.zeros(X_shape, dtype=np.float32)
        X_noisy = np.zeros(X_shape, dtype=np.float32)
        X_adv = np.zeros(X_shape, dtype=np.float32)

        for i in range(n):
            X_benign[i] = X[i, 0]
            X_noisy[i] = X[i, 1]
            X_adv[i] = X[i, 2]

        characteristics, labels = train_lid(
            self.model,
            X=X_benign,
            X_noisy=X_noisy,
            X_adv=X_adv,
            k=self.k,
            batch_size=self.batch_size,
            device=self.device
        )
        characteristics = self.scaler_.transform(characteristics)
        prob = self.detector_.predict_proba(characteristics)[:, 1]
        return roc_auc_score(labels, prob)

    def save(self, path):
        pass

    def load(self, path):
        pass