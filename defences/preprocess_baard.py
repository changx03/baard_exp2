import os
import sys

import numpy as np
import torchvision as tv

sys.path.append(os.getcwd())
from models import AddGaussianNoise


def preprocess_baard_img(data_name, tensor_X):
    """Preprocess training data"""
    if data_name == 'cifar10':
        # return tensor_X
        transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            # tv.transforms.RandomCrop(32, padding=4),
            AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)
    elif data_name == 'mnist':
        # return tensor_X
        transform = tv.transforms.Compose([
            tv.transforms.RandomRotation(5)
            # AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)
    else:
        # return tensor_X
        transform = tv.transforms.Compose([
            AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)


def preprocess_baard_numpy(X, std=1., eps=0.025):
    X_noisy = X + eps * np.random.normal(0, std, size=X.shape)
    return X_noisy
