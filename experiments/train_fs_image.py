import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.util import get_shape, dataset2tensor
from models.torch_util import validate
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from defences.feature_squeezing import (GaussianSqueezer, MedianSqueezer,
                                        DepthSqueezer, FeatureSqueezingTorch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--param', type=str, required=True)
    args = parser.parse_args()

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)

    with open(args.param) as param_json:
        param = json.load(param_json)
    param['n_classes'] = 10
    print('Param:', param)

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
        raise ValueError('{} is not supported.'.format(args.data))

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
    else:  # args.data == 'cifar10':
        model_name = args.pretrained.split('_')[1]
        if model_name == 'resnet':
            model = Resnet(use_prob=use_prob).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=use_prob).to(device)
        else:
            raise ValueError('Unknown model: {}'.format(model_name))

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train * 100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test * 100))

    tensor_train_X, tensor_train_y = dataset2tensor(dataset_train)
    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()

    # Train defence
    squeezers = []
    squeezers.append(GaussianSqueezer(x_min=0.0, x_max=1.0, noise_strength=0.025, std=1.0))
    squeezers.append(DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=8))
    if args.data in ['mnist', 'cifar10']:
        squeezers.append(MedianSqueezer(x_min=0.0, x_max=1.0, kernel_size=3))
    print('FS: # of squeezers:', len(squeezers))
    detector = FeatureSqueezingTorch(
        classifier=model,
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        loss=loss,
        batch_size=128,
        x_min=0.0,
        x_max=1.0,
        squeezers=squeezers,
        n_classes=param['n_classes'],
        device=device)
    detector.fit(X_train, y_train, epochs=param['epochs'], verbose=1)

    path_fs = os.path.join(args.output_path, '{}_fs.pt'.format(args.pretrained.split('.')[0]))
    detector.save(path_fs)
    print('Saved fs to:', path_fs)
    print()


if __name__ == '__main__':
    main()
