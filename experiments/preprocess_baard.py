import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from defences.util import (acc_on_adv, dataset2tensor, get_correct_examples,
                           get_shape, merge_and_generate_labels)
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import (AddGaussianNoise, predict, predict_numpy,
                               validate)

from experiments.util import load_csv, set_seeds


def baard_preprocess(data, tensor_X):
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


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    if not os.path.exists(args.output_path):
        print('Output folder does not exist. Create:', args.output_path)
        os.mkdir(args.output_path)
        
    print('data:', args.data)
    print('model:', args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Prepare data
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    if args.data == 'mnist':
        dataset_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transforms)
    elif args.data == 'cifar10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transforms)
    else:
        data_path = os.path.join(args.data_path, data_params['data'][args.data]['file_name'])
        print('Read file:', data_path)
        X, y = load_csv(data_path)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_params['data'][args.data]['n_test'],
            random_state=args.random_state)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        dataset_train = TensorDataset(torch.from_numpy(X_train).type(torch.float32), torch.from_numpy(y_train).type(torch.long))
        dataset_test = TensorDataset(torch.from_numpy(X_test).type(torch.float32), torch.from_numpy(y_test).type(torch.long))

    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    shape_train = get_shape(loader_train.dataset)
    shape_test = get_shape(loader_test.dataset)
    print('Train set:', shape_train)
    print('Test set:', shape_test)
    use_prob = True
    print('Using softmax layer:', use_prob)

    # Load model
    if args.data == 'mnist':
        model = BaseModel(use_prob=use_prob).to(device)
        model_name = 'basic'
    elif args.data == 'cifar10':
        model_name = args.pretrained.split('_')[1]
        if model_name == 'resnet':
            model = Resnet(use_prob=use_prob).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=use_prob).to(device)
        else:
            raise NotImplementedError
    else:
        n_features = data_params['data'][args.data]['n_features']
        n_classes = data_params['data'][args.data]['n_classes']
        model = NumericModel(n_features, n_hidden=n_features * 4, n_classes=n_classes, use_prob=use_prob).to(device)
        model_name = 'basic' + str(n_features * 4)

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train * 100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test * 100))

    # Create a subset which only contains recognisable samples.
    # The original train and test sets are no longer needed.
    tensor_train_X, tensor_train_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    dataset_train = TensorDataset(tensor_train_X, tensor_train_y)
    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=False)
    _, acc_perfect = validate(model, loader_train, loss, device)
    print('Accuracy on {} filtered train set: {:.4f}%'.format(len(dataset_train), acc_perfect * 100))

    tensor_test_X, tensor_test_y = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
    dataset_test = TensorDataset(tensor_test_X, tensor_test_y)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)
    _, acc_perfect = validate(model, loader_test, loss, device)
    print('Accuracy on {} filtered test set: {:.4f}%'.format(len(dataset_test), acc_perfect * 100))

    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()

    X_baard = baard_preprocess(args.data, tensor_train_X).cpu().detach().numpy()
    obj = {
        'X_train': X_baard,
        'y_train': y_train
    }
    path_ouput = os.path.join(args.output_path, '{}_{}_baard_train.pt'.format(args.data, args.model, args.model))
    torch.save(obj, path_ouput)
    print('Save to:', path_ouput)
    print()


if __name__ == '__main__':
    main()
