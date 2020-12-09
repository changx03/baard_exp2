import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


def get_range(data):
    """Get column-wise min and max values"""
    x_max = np.max(data, axis=0)
    x_min = np.min(data, axis=0)
    return (x_min, x_max)


def normalize(data, xmin, xmax):
    """Scale the data to [0, 1]"""
    return (data - xmin) / (xmax - xmin)


def unnormalize(data, xmin, xmax):
    """Rescale the data to normal range"""
    return data * (xmax - xmin) + xmin


def get_roc(y_true, y_pred, show_plot=False):
    """Get ROC AUC scores"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    if show_plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score, thresholds


def get_shape(dataset):
    """Get the shape of the data (This ignores the label)"""
    shape = list(next(iter(dataset))[0].size())
    shape = [len(dataset)] + shape
    return tuple(shape)


def get_correct_examples(model, dataset, device='cuda',
                         batch_size=512, return_tensor=True):
    """Remove incorrect predictions"""
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
            preds = outputs.max(1, keepdim=True)[1]
            corrects[start:end] = y.eq(preds.view_as(y)).cpu()
            start += n
    indices = torch.squeeze(torch.nonzero(corrects), 1)
    if return_tensor:
        return X[indices], Y[indices]
    dataset = TensorDataset(X[indices], Y[indices])
    return dataset
