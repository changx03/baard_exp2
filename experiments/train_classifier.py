"""
Training a classifier.
"""
import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import train, validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from utils import load_csv, set_seeds

from experiments import get_output_path

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 192


def pytorch_train_classifier(data_name, model_name, idx):
    print('Runing train_classifier.py')
    seed = SEEDS[idx]
    set_seeds(seed)

    path_results = get_output_path(idx, data_name, model_name)
    if not os.path.exists(path_results):
        print('[DATA] Output folder does not exist. Create:', path_results)
        path = Path(os.path.join(path_results, 'data'))
        print('[DATA] Create folder:', path)
        path.mkdir(parents=True, exist_ok=True)
        path = Path(os.path.join(path_results, 'results'))
        path.mkdir(parents=True, exist_ok=True)
        print('[DATA] Create folder:', path)

    # Step 1: Load data
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data_name == 'mnist':
        dataset_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)
    elif data_name == 'cifar10':
        transform_train = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor()])
        dataset_train = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10(DATA_PATH, train=False, download=True, transform=transform)
    else:
        data_path = os.path.join(DATA_PATH, METADATA['data'][data_name]['file_name'])
        n_test = METADATA['data'][data_name]['n_test']
        print('[DATA] Read file:', data_path)
        X, y = load_csv(data_path)
        scalar = MinMaxScaler().fit(X, y)
        X = scalar.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=seed)
        dataset_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        dataset_test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    ############################################################################
    # Step 2: Train model
    epochs = 200 if data_name in ['mnist', 'cifar10'] else 400
    print('[CLASSIFIER] epochs:', epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[CLASSIFIER] device:', device)

    if data_name == 'mnist':
        model = BaseModel(use_prob=False).to(device)
    elif data_name == 'cifar10':
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
            model = Resnet(use_prob=False).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
        else:
            raise NotImplementedError
    else:
        n_features = METADATA['data'][data_name]['n_features']
        n_hidden = n_features * 4
        n_classes = METADATA['data'][data_name]['n_classes']
        model = NumericModel(n_features=n_features, n_hidden=n_hidden, n_classes=n_classes, use_prob=False).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print('[CLASSIFIER] Start training {} model on {}...'.format(model_name, data_name))
    time_start = time.time()
    for e in range(epochs):
        start = time.time()
        tr_loss, tr_acc = train(model, dataloader_train, loss, optimizer, device)
        va_loss, va_acc = validate(model, dataloader_test, loss, device)
        scheduler.step()
        time_elapsed = time.time() - start
        print(('[CLASSIFIER] {:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, Test Loss: {:.4f} Acc: {:.4f}%').format(
            e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100., va_loss, va_acc * 100.))
    time_elapsed = time.time() - time_start
    print('[CLASSIFIER] Time spend on training classifier: {}'.format(str(datetime.timedelta(seconds=time_elapsed))))

    _, acc = validate(model, dataloader_train, loss, device)
    print('[CLASSIFIER] accuracy on training set:', acc)
    _, acc = validate(model, dataloader_test, loss, device)
    print('[CLASSIFIER] accuracy on test set:    ', acc)

    file_model = os.path.join(path_results, 'data', '{}_{}_model.pt'.format(data_name, model_name))
    torch.save(model.state_dict(), file_model)
    print('[CLASSIFIER] save to:', file_model)

    print()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-m', '--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)

    idx = args.idx
    seed = SEEDS[idx]
    data_name = args.data
    model_name = args.model
    print('seed:', seed)
    print('data_name:', data_name)
    print('model_name:', model_name)
    print('idx', idx)

    pytorch_train_classifier(data_name, model_name, idx)

    # Testing
    # pytorch_train_classifier('banknote', 'dnn', 0)
    # pytorch_train_classifier('breastcancer', 'dnn', 0)
    # pytorch_train_classifier('htru2', 'dnn', 0)
    # pytorch_train_classifier('mnist', 'dnn', 0)
    # pytorch_train_classifier('cifar10', 'resnet', 0)
    # pytorch_train_classifier('cifar10', 'vgg', 0)

# python ./experiments/train_classifier.py -d banknote -m dnn -i 0
