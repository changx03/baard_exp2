import datetime
import os
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from models.torch_util import train, validate

BATCH_SIZE = 128


class SurrogateModel(nn.Module):
    """This is the surrogate model for BAARD"""

    def __init__(self, in_channels=1, use_prob=True):
        super(SurrogateModel, self).__init__()
        self.use_prob = use_prob

        # Compute nodes after flatten
        if in_channels == 1:
            n_flat = 9216
        elif in_channels == 3:
            n_flat = 12544
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(n_flat, 200)
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


def train_surrogate(X_train, X_test, y_train, y_test, epochs, device):
    n_channels = X_train.shape[1]
    print('n_channels', n_channels)

    model = SurrogateModel(in_channels=n_channels, use_prob=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset_train = TensorDataset(
        torch.from_numpy(X_train).type(torch.float32),
        torch.from_numpy(y_train).type(torch.long))
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataset_test = TensorDataset(
        torch.from_numpy(X_test).type(torch.float32),
        torch.from_numpy(y_test).type(torch.long))
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    time_start = time.time()
    for e in range(epochs):
        start = time.time()
        tr_loss, tr_acc = train(model, loader_train, loss, optimizer, device)
        va_loss, va_acc = validate(model, loader_test, loss, device)
        scheduler.step()
        time_elapsed = time.time() - start
        print(('{:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, Test Loss: {:.4f} Acc: {:.4f}%').format(
            e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100, va_loss, va_acc * 100))
    time_elapsed = time.time() - time_start
    print('Total runing time:', str(datetime.timedelta(seconds=time_elapsed)))

    return model


def get_pretrained_surrogate(file, device):
    file_arr = (file.split('/')[-1]).split('_')
    data = file_arr[0]

    if data == 'mnist':
        n_channels = 1
    elif data == 'cifar10':
        n_channels = 3
    else:
        raise NotImplementedError
    model = SurrogateModel(in_channels=n_channels, use_prob=True).to(device)
    model.load_state_dict(torch.load(file))
    return model
