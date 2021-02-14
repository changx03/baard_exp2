import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .get_dataset_shape import get_dataset_shape


def get_correct_examples(model, dataset, device='cuda',
                         batch_size=512, return_tensor=True):
    """Removes incorrect predictions."""
    model.eval()
    shape = get_dataset_shape(dataset)
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


def get_correct_examples_sklearn(estimator, X, y):
    pred = estimator.predict(X)
    idx = np.where(pred == y)[0]
    return X[idx], y[idx]
