import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from tqdm import tqdm


def get_hidden_layers(sequence, device):
    n_hidden_layers = len(list(sequence.children())) - 1
    # get deep representations
    hidden_layers = []
    for i in range(1, n_hidden_layers+1):
        layer = nn.Sequential(*list(sequence.children())[:i]).to(device)
        hidden_layers.append(layer)
    return hidden_layers, n_hidden_layers


def mle_batch(data, batch, k):
    """Compute Maximum Likelihood Estimator of LID within k nearlest neighbours
    """
    k = min(k, len(data)-1)

    def mle(v):
        # v[-1] is the max of the neighbour distances
        return - k / np.sum(np.log(v/v[-1]))

    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(mle, axis=1, arr=a)
    return a


def get_lid_random_batch(sequence, X, X_noisy, X_adv, k, batch_size, device):
    """
    Train the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    """
    hidden_layers, n_hidden_layers = get_hidden_layers(sequence, device)
    print('Number of hidden layers: {}'.format(len(hidden_layers)))

    # Convert numpy Array to PyTorch Tensor
    X = torch.tensor(X, dtype=torch.float32).to(device)
    X_noisy = torch.tensor(X_noisy, dtype=torch.float32).to(device)
    X_adv = torch.tensor(X_adv, dtype=torch.float32).to(device)

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
        for i_batch in tqdm(range(n_batches)):
            batch_benign, batch_noisy, batch_adv = estimate(i_batch)
            lid_benign.extend(batch_benign)
            lid_noisy.extend(batch_noisy)
            lid_adv.extend(batch_adv)

    lid_benign = np.asarray(lid_benign, dtype=np.float32)
    lid_noisy = np.asarray(lid_noisy, dtype=np.float32)
    lid_adv = np.asarray(lid_adv, dtype=np.float32)
    return lid_benign, lid_noisy, lid_adv


def merge_and_generate_labels(X_pos, X_neg):
    """Merge positive and negative artifact and generate labels
    """
    X_pos = X_pos.reshape(X_pos.shape[0], -1)
    X_neg = X_neg.reshape(X_neg.shape[0], -1)
    # print(X_pos.shape, X_neg.shape)
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.astype(np.long)
    return X, y


def get_lid(sequence, X, X_noisy, X_adv, k=20, batch_size=100, device='cpu'):
    """Get local intrinsic dimensionality (LID)"""
    lid_benign, lid_noisy, lid_adv = get_lid_random_batch(
        sequence, X, X_noisy, X_adv, k, batch_size, device)
    lid_pos = lid_adv
    lid_neg = np.concatenate((lid_benign, lid_noisy))
    artifacts, labels = merge_and_generate_labels(lid_pos, lid_neg)
    return artifacts, labels


def eval_single_lid(sequence, X_train, x, k=20, batch_size=100, device='cpu'):
    """Evaluate LID for a single example"""
    def mle(v):
        return - k / np.sum(np.log(v/v[-1]))

    samples = np.random.choice(X_train, size=batch_size, replace=False)
    samples = np.concatenate((np.expand_dims(x, axis=0), samples))
    samples = torch.tensor(samples, dtype=torch.float32).to(device)
    hidden_layers, n_hidden_layers = get_hidden_layers(sequence, device)
    single_lid = np.zeros(n_hidden_layers, dtype=np.float32)
    
    for i, layer in enumerate(hidden_layers):
        layer.eval()
        output = layer(samples)
        output = output.view(output.size(0), -1).cpu().detach().numpy()
        dist = cdist(output, output)
        dist = np.sort(dist[0])[1:k+1]
        single_lid[i] = mle(dist)

    # TODO: Not tested yet!
    return single_lid
