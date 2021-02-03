import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from defences.util import get_correct_examples
from experiments.util import set_seeds
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.torch_util import validate


class SurrogateModel(nn.Module):
    """This is the surrogate model for BAARD"""

    def __init__(self, in_channels=1, use_prob=True):
        super(SurrogateModel, self).__init__()
        self.use_prob = use_prob

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(9216, 200)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(200, 2)
        self.softmax = nn.Softmax(dim=1)

    def before_softmax(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.before_softmax(x)
        if self.use_prob:
            x = self.softmax(x)
        return x


def train_surrogate(data='mnist', path_trainset='results', batch_size=128, device='cpu'):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, required=True, choices=['basic', 'resnet', 'vgg'])
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--eps', type=float, default=2.)
    parser.add_argument('--test', type=int, default=0, choices=[0, 1])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    time_start = time.time()
    train_surrogate(args.data, args.output_path, args.batch_size, device)
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
    print()


if __name__ == '__main__':
    main()
