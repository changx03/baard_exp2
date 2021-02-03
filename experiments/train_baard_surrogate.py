import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from models.torch_util import train, validate

from experiments.util import set_seeds


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


def train_surrogate(model, loader_train, loader_test, epochs, device):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for e in range(epochs):
        start = time.time()
        tr_loss, tr_acc = train(model, loader_train, loss, optimizer, device)
        va_loss, va_acc = validate(model, loader_test, loss, device)
        scheduler.step()
        time_elapsed = time.time() - start
        print(('{:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, Test Loss: {:.4f} Acc: {:.4f}%').format(
            e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100, va_loss, va_acc * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, required=True, choices=['basic', 'resnet', 'vgg'])
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    if not os.path.exists(args.output_path):
        print('Output folder does not exist. Create:', args.output_path)
        os.mkdir(args.output_path)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    n_channels = 1 if args.data == 'mnist' else 3

    # Load training data (Shuffle is required)
    path_train = os.path.join(args.output_path, args.train)
    obj_train = torch.load(path_train)
    X_train = np.concatenate((obj_train['X'], obj_train['adv']))
    y_train = np.concatenate((obj_train['baard_label_x'], obj_train['baard_label_adv']))
    dataset_train = TensorDataset(
        torch.from_numpy(X_train).type(torch.float32),
        torch.from_numpy(y_train).type(torch.long))
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    print('[Train set] Adv: {}, Benign: {}'.format(np.sum(y_train == 1), np.sum(y_train == 0)))

    # Load test data (Do not shuffle)
    path_test = os.path.join(args.output_path, args.test)
    obj_test = torch.load(path_test)
    X_test = np.concatenate((obj_test['X'], obj_test['adv']))
    y_test = np.concatenate((obj_test['baard_label_x'], obj_test['baard_label_adv']))
    dataset_test = TensorDataset(
        torch.from_numpy(X_test).type(torch.float32),
        torch.from_numpy(y_test).type(torch.long))
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    print('[Test set] Adv: {}, Benign: {}'.format(np.sum(y_test == 1), np.sum(y_test == 0)))

    model = SurrogateModel(in_channels=n_channels, use_prob=True)

    time_start = time.time()
    train_surrogate(model, loader_train, loader_test, args.epochs, device)
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))

    file_name = os.path.join(args.output_path, '{}_{}_surrogate_{}.pt'.format(args.data, args.model, args.epochs))
    torch.save(model.state_dict(), file_name)
    print('Save to:', file_name)
    print()


if __name__ == '__main__':
    main()
