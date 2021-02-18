import datetime
import os
import sys
import time

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import torch
import torch.nn as nn
from models.torch_util import train, validate
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 192


class SurrogateModel(nn.Module):
    """This is the surrogate model for BAARD"""

    def __init__(self, in_channels=1):
        super(SurrogateModel, self).__init__()
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

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_prob(self, x):
        x = self.forward(x)
        x = self.softmax(x)
        return x


def train_surrogate(X_train, X_test, y_train, y_test, epochs, device):
    n_channels = X_train.shape[1]
    print('[SURROGATE] n_channels', n_channels)

    model = SurrogateModel(in_channels=n_channels).to(device)
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
        print(('[SURROGATE] {:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.2f}%, Test Loss: {:.4f} Acc: {:.2f}%').format(
            e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100, va_loss, va_acc * 100))
    time_elapsed = time.time() - time_start
    print('[SURROGATE] Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
    return model


# Testing
if __name__ == '__main__':
    x = torch.randn((64, 3, 32, 32))
    net = SurrogateModel(in_channels=3)
    out = net(x)
    print(out.size())
    print(out[:5])
