import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.datasets as datasets
from art.attacks.evasion import (AutoProjectedGradientDescent, BasicIterativeMethod,
                                 BoundaryAttack, DeepFool,
                                 FastGradientMethod, SaliencyMapMethod)
from art.estimators.classification import PyTorchClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .util import load_csv

# Adding the parent directory.
sys.path.append(os.getcwd())
from models.mnist import BaseModel
from models.cifar10 import Resnet, Vgg
from models.numeric import NumericModel
from experiments.train_pt import validate
from defences.util import get_shape, get_correct_examples


# This seed ensures the pre-trained models have the same train and test sets.
RANDOM_STATE = int(2**12)

DATA_NAMES = ['mnist', 'cifar10', 'banknote', 'htru2', 'segment', 'texture']
DATA = {
    'mnist': {'n_features': (1, 28, 28), 'n_classes': 10},
    'cifar10': {'n_features': (3, 32, 32), 'n_classes': 10},
    'banknote': {'file_name': 'banknote_preprocessed.csv', 'n_features': 4, 'n_test': 400, 'n_classes': 2},
    'htru2': {'file_name': 'htru2_preprocessed.csv', 'n_features': 8, 'n_test': 4000, 'n_classes': 2},
    'segment': {'file_name': 'segment_preprocessed.csv', 'n_features': 18, 'n_test': 400, 'n_classes': 7},
    'texture': {'file_name': 'texture_preprocessed.csv', 'n_features': 40, 'n_test': 600, 'n_classes': 11},
}
ATTACKS = ['apgd', 'bim', 'boundary', 'cw2', 'deepfool', 'fgsm', 'jsma']