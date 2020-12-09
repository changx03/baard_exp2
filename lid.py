import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from tqdm import tqdm


def mle_batch(data_benign, data_current, k):
    """Compute Maximum Likelihood Estimator of LID within k nearlest neighbours
    """
    k = min(k, len(data_benign)-1)
    def estimator_func(v): return - k / np.sum(np.log(v/v[-1]))
    dist = cdist(data_benign, data_current)
    dist = np.apply_along_axis(np.sort, axis=1, arr=dist)[:, 1:k+1]
    dist = np.apply_along_axis(estimator_func, axis=1, arr=dist)
    return dist


def get_lid_random_batch(sequence, X, X_noisy, X_adv, k, batch_size):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    """
    n_layers = len(list(sequence.children()))
    # get deep representations
    hidden_layers = []
    for i in range(1, n_layers + 1):
        layer = nn.Sequential(*list(sequence.children())[:i])
        hidden_layers.append(layer)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros((n_feed, n_layers))
        lid_batch_noisy = np.zeros((n_feed, n_layers))
        lid_batch_adv = np.zeros((n_feed, n_layers))
        # We don't use GPU at this stage
        for i, layer in enumerate(hidden_layers):
            layer.eval()
            output = layer(torch.tensor(X[start: end]))
            output = output.cpu().detach().numpy().reshape((n_feed, -1))

            output_noisy = layer(torch.tensor(X_noisy[start: end]))
            output_noisy = output_noisy.cpu().detach().numpy().reshape((n_feed, -1))

            output_adv = layer(torch.tensor(X_adv[start: end]))
            output_adv = output_adv.cpu().detach().numpy().reshape((n_feed, -1))

            lid_batch[:, i] = mle_batch(output, output, k=k)
            lid_batch_noisy[:, i] = mle_batch(output, output_noisy, k=k)
            lid_batch_adv[:, i] = mle_batch(output, output_adv, k=k)
        return lid_batch, lid_batch_noisy, lid_batch_adv

    lid_benign = []
    lid_adv = []
    lid_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch_benign, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lid_benign.extend(lid_batch_benign)
        lid_noisy.extend(lid_batch_noisy)
        lid_adv.extend(lid_batch_adv)

    lid_benign = np.asarray(lid_benign, dtype=np.float32)
    lid_noisy = np.asarray(lid_noisy, dtype=np.float32)
    lid_adv = np.asarray(lid_adv, dtype=np.float32)
    return lid_benign, lid_noisy, lid_adv


def merge_and_generate_labels(X_pos, X_neg):
    """Merge positive and negative artifact and generate labels
    """
    X_pos = X_pos.reshape(X_pos.shape[0], -1)
    X_neg = X_neg.reshape(X_pos.shape[0], -1)
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))
    return X, y


def get_lid(sequence, X, X_noisy, X_adv, k=20, batch_size=100):
    """Get local intrinsic dimensionality (LID)"""
    lid_benign, lid_noisy, lid_adv = get_lid_random_batch(
        sequence, X, X_noisy, X_adv, k, batch_size)
    lid_pos = lid_adv
    lid_neg = np.concatenate((lid_benign, lid_noisy))
    artifacts, labels = merge_and_generate_labels(lid_pos, lid_neg)
    return artifacts, labels
