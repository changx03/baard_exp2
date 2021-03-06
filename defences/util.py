import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


def get_range(data):
    """Returns column-wise min and max values"""
    x_max = np.max(data, axis=0).astype(np.float)
    x_min = np.min(data, axis=0).astype(np.float)
    return (x_min, x_max)


def normalize(data, xmin, xmax):
    """Rescales the data to [0, 1]"""
    return (data - xmin) / (xmax - xmin)


def unnormalize(data, xmin, xmax):
    """Reverses the data to normal range"""
    return data * (xmax - xmin) + xmin


def get_roc(y_true, y_prob, show_plot=False):
    """Returns False-Positive-Rate, True-Positive-Rate, AUC score and thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score, thresholds


def get_shape(dataset):
    """Retruns the shape of the data in a PyTorch Dataset object."""
    # shape = list(next(iter(dataset))[0].size())
    X, _ = next(iter(dataset))
    shape = list(X.size())
    shape = [len(dataset)] + shape
    return tuple(shape)


def get_correct_examples(model, dataset, device='cuda',
                         batch_size=512, return_tensor=True):
    """Removes incorrect predictions."""
    model.eval()
    shape = get_shape(dataset)
    X = torch.zeros(shape, dtype=torch.float32)
    Y = torch.zeros(len(dataset), dtype=torch.long)
    corrects = torch.zeros(len(dataset), dtype=torch.bool)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = 0
    with torch.no_grad():
        for x, y in dataloader:
            n = x.size(0)
            end = start + n
            X[start:end] = x
            Y[start:end] = y
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            pred = outputs.max(1, keepdim=True)[1]
            corrects[start:end] = y.eq(pred.view_as(y)).cpu()
            start += n
    indices = torch.squeeze(torch.nonzero(corrects), 1)
    if return_tensor:
        return X[indices], Y[indices]
    dataset = TensorDataset(X[indices], Y[indices])
    return dataset


def dataset2tensor(dataset, batch_size=512):
    """Extracts X and y from a PyTorch dataset object."""
    shape = get_shape(dataset=dataset)
    n_samples = shape[0]
    X = torch.zeros(shape, dtype=torch.float32)
    Y = torch.zeros(n_samples, dtype=torch.long)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = 0
    for x, y in dataloader:
        n = x.size(0)
        end = start + n
        X[start:end] = x
        Y[start:end] = y
        start += n
    return X, Y


def get_binary_labels(adv, benign):
    """Creates binary labels with adversarial examples assign to 1 and benign 
    examples assign to 0.
    """
    y = np.concatenate((np.ones(adv.shape[0]), np.zeros(benign.shape[0])))
    return y.astype(np.long)


def merge_and_generate_labels(X_adv, X_benign, flatten=True):
    """Merges positive and negative artifact and generate labels."""
    if flatten:
        X_adv = X_adv.reshape(X_adv.shape[0], -1)
        X_benign = X_benign.reshape(X_benign.shape[0], -1)
    X = np.concatenate((X_adv, X_benign)).astype(np.float)
    y = get_binary_labels(X_adv, X_benign)
    return X, y


def acc_on_adv(y_pred, y_true, detected_as_adv):
    """Compute the accuracy on the adversarial examples.

    Parameters
    ----------
    y_pred: numpy array of integers
        labels predicted by the classifier for the adversarial examples.

    y_true: numpy array of integers
        true labels.

    detected_as_advx: numpy array of boolean value
        the i-th value is true if the sample has been detected as an adversarial
        example, false otherwise.

    Returns
    -------
    accuracy: float
        Accuracy on adversarial examples.
    """
    correct_classified = y_pred == y_true
    correct_classified_and_detected = np.logical_or(correct_classified, detected_as_adv)
    return np.mean(correct_classified_and_detected)
