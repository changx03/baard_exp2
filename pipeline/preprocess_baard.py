import os
import sys

import torchvision as tv

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from models.torch_util import AddGaussianNoise


def preprocess_baard(data, tensor_X):
    """Preprocess training data"""
    if data == 'cifar10':
        # return tensor_X
        transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            # tv.transforms.RandomCrop(32, padding=4),
            AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)
    elif data == 'mnist':
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
