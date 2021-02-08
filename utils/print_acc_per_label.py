import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from models.torch_util import predict_numpy


def print_acc_per_label(model, X, y, device):
    labels = np.unique(y)
    for i in labels:
        idx = np.where(y == i)[0]
        x_subset = X[idx]
        y_subset = y[idx]
        pred = predict_numpy(model, x_subset, device)
        correct = np.sum(pred == y_subset)
        print('[{}] {}/{}'.format(i, correct, len(x_subset)))