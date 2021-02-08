import torch
from torch.utils.data import DataLoader

from .get_correct_samples import get_correct_samples


def dataset2tensor(dataset, batch_size=512):
    """Extracts X and y from a PyTorch dataset object."""
    shape = get_correct_samples(dataset=dataset)
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
