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

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.region_based_classifier import RegionBasedClassifier
from defences.util import (get_correct_examples,
                           get_shape, merge_and_generate_labels)
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import predict, validate
from experiments.util import load_csv


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--adv', type=str, required=True, help="Example: 'mnist_basic_apgd_0.3'")
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)
    print('Pretrained samples:', args.adv + '_adv.npy')

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

    # Note: Train set alway shuffle!
    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    shape_train = get_shape(loader_train.dataset)
    shape_test = get_shape(loader_test.dataset)
    print('Train set:', shape_train)
    print('Test set:', shape_test)
    use_prob = True
    print('Using softmax layer:', use_prob)

    n_classes = data_params['data'][args.data]['n_classes']

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
            raise ValueError('Unknown model: {}'.format(model_name))
    else:
        n_features = data_params['data'][args.data]['n_features']
        model = NumericModel(n_features, n_hidden=n_features * 4, n_classes=n_classes, use_prob=use_prob).to(device)
        model_name = 'basic' + str(n_features * 4)

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train * 100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test * 100))

    # Create a subset which only contains recognisable samples.
    # The original train and test sets are no longer needed.
    tensor_train_X, tensor_train_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    dataset_train = TensorDataset(tensor_train_X, tensor_train_y)
    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_train, loss, device)
    print('Accuracy on {} filtered train set: {:.4f}%'.format(len(dataset_train), acc_perfect * 100))

    tensor_test_X, tensor_test_y = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
    dataset_test = TensorDataset(tensor_test_X, tensor_test_y)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_test, loss, device)
    print('Accuracy on {} filtered test set: {:.4f}%'.format(len(dataset_test), acc_perfect * 100))

    # Load pre-trained adversarial examples
    path_benign = os.path.join(args.output_path, args.adv + '_x.npy')
    path_adv = os.path.join(args.output_path, args.adv + '_adv.npy')
    path_y = os.path.join(args.output_path, args.adv + '_y.npy')
    X_benign = np.load(path_benign)
    adv = np.load(path_adv)
    y_true = np.load(path_y)

    dataset = TensorDataset(torch.from_numpy(X_benign), torch.from_numpy(y_true))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} benign samples: {:.4f}%'.format(len(dataset), acc * 100))

    dataset = TensorDataset(torch.from_numpy(adv), torch.from_numpy(y_true))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} adversarial examples: {:.4f}%'.format(len(dataset), acc * 100))

    # Do NOT shuffle the indices, so different defences can use the same test set.
    dataset = TensorDataset(torch.from_numpy(adv))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    pred_adv = predict(model, loader, device).cpu().detach().numpy()

    # Find the thresholds using the 2nd half
    n = len(X_benign) // 2
    # Merge benign samples and adversarial examples into one set.
    # This labels indicate a sample is an adversarial example or not.
    X_val, labels_val = merge_and_generate_labels(adv[n:], X_benign[n:], flatten=False)
    # The predictions for benign samples are exactly same as the true labels.
    pred_val = np.concatenate((pred_adv[n:], y_true[n:]))

    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()

    # Train defence
    time_start = time.time()
    detector = RegionBasedClassifier(
        model=model,
        r=0.2,
        sample_size=1000,
        n_classes=n_classes,
        x_min=0.0,
        x_max=1.0,
        batch_size=512,
        r0=0.0,
        step_size=0.02,
        stop_value=0.4,
        device=device)
    detector.search_thresholds(X_val, pred_val, labels_val, verbose=0)
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
    print()


if __name__ == '__main__':
    main()
