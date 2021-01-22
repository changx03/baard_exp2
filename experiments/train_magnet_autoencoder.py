# MagNet requires to train autoencoder. Each dataset only need to train the
# autoencoder once.
# Run this before running the MagNet defence.
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
from sklearn.model_selection import train_test_split

# Adding the parent directory.
sys.path.append(os.getcwd())
from models.mnist import BaseModel
from models.cifar10 import Resnet, Vgg
from defences.util import dataset2tensor
from models.torch_util import validate
from defences.magnet import Autoencoder1, Autoencoder2, MagNetDetector

DATA_NAMES = ['mnist', 'cifar10']
DATA = {
    'mnist': {'n_features': (1, 28, 28), 'n_classes': 10},
    'cifar10': {'n_features': (3, 32, 32), 'n_classes': 10}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=DATA_NAMES)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--param', type=str, required=True)
    args = parser.parse_args()

    with open(args.param) as param_json:
        param = json.load(param_json)
    param['n_classes'] = DATA[args.data]['n_classes']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)
    print('Param:', param)
    print('Device: {}'.format(device))

    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    # the autoencoder2 need a larger temperature value for the softmax function.
    use_prob = False
    if args.data == 'mnist':
        dataset_train = datasets.MNIST(
            args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.MNIST(
            args.data_path, train=False, download=True, transform=transforms)

        model = BaseModel(use_prob=use_prob).to(device)
        model_name = 'basic'
    elif args.data == 'cifar10':
        dataset_train = datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.CIFAR10(
            args.data_path, train=False, download=True, transform=transforms)

        model_name = args.pretrained.split('_')[1]
        if model_name == 'resnet':
            model = Resnet(use_prob=use_prob).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=use_prob).to(device)
        else:
            raise ValueError('model_name must be either resnet or vgg.')
    else:
        raise ValueError('This autoencoder does not support other datasets.')

    tensor_X_train, tensor_y_train = dataset2tensor(dataset_train)
    X_train = tensor_X_train.cpu().detach().numpy()
    y_train = tensor_y_train.cpu().detach().numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=5000)

    loader_train = DataLoader(
        dataset_train, batch_size=512, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train*100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test*100))

    if args.data == 'mnist':
        detector1 = MagNetDetector(
            encoder=Autoencoder1(n_channel=DATA[args.data]['n_features'][0]),
            classifier=model,
            lr=param['lr'],
            batch_size=param['batch_size'],
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=1,
            device=device)
        detector1.fit(X_train, y_train, epochs=param['epochs'])

        detector2 = MagNetDetector(
            encoder=Autoencoder2(n_channel=DATA[args.data]['n_features'][0]),
            classifier=model,
            lr=param['lr'],
            batch_size=param['batch_size'],
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=2,
            device=device)
        detector2.fit(X_train, y_train, epochs=param['epochs'])

        detectors = [detector1, detector2]
    elif args.data == 'cifar10':
        autoencoder = Autoencoder2(n_channel=DATA[args.data]['n_features'][0])
        detectors = []
        detector = MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=param['batch_size'],
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=2,
            device=device)
        detector.fit(X_train, y_train, epochs=param['epochs'])
        detectors.append(detector)
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=param['batch_size'],
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='prob',
            temperature=10,
            device=device))
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=param['batch_size'],
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='prob',
            temperature=40,
            device=device))
    else:
        raise ValueError('Unsupported dataset.')
    
    # Train autoencoders
    for ae in detectors:
        mse = ae.score(X_val)
        print('MSE training set: {:.6f}, validation set: {:.6f}'.format(
            ae.history_train_loss[-1] if len(ae.history_train_loss) > 0 else np.inf, 
            mse))

        ae.search_threshold(X_val, fp=param['fp'], update=True)
        print('Threshold:', ae.threshold)

    # Save autoencoders
    for i, ae in enumerate(detectors, start=1):
        encoder_path = os.path.join(
            args.output_path,
            'autoencoder_{}_{}_{}.pt'.format(args.data, model_name, i))
        ae.save(encoder_path)
        print('File is saved to:', encoder_path)

if __name__ == '__main__':
    main()
