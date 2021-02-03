import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from defences.util import acc_on_adv, get_correct_examples
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.torch_util import predict_numpy, validate

from experiments.util import set_seeds


def get_baard_output(data, model_name, data_path, output_path, file_name, param, batch_size, device):
    file_path = os.path.join(output_path, file_name)
    print('file_path:', file_path)

    obj = torch.load(file_path)
    X = obj['X']
    adv = obj['adv']
    y = obj['y']

    # Load model
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data == 'mnist':
        dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms)
        model = BaseModel(use_prob=False).to(device)
        pretrained = 'mnist_200.pt'
    elif data == 'cifar10':
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms)
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
            pretrained = 'cifar10_resnet_200.pt'
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
            pretrained = 'cifar10_vgg_200.pt'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    pretrained_path = os.path.join(output_path, pretrained)
    model.load_state_dict(torch.load(pretrained_path))
    loss = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} clean samples: {:.4f}%'.format(X.shape[0], acc * 100))
    dataset = TensorDataset(torch.from_numpy(adv), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} adv examples: {:.4f}%'.format(adv.shape[0], acc * 100))

    tensor_train_X, tensor_train_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()

    # Load baard
    baard_train_path = os.path.join('results', '{}_{}_baard_train.pt'.format(data, model_name))
    obj = torch.load(baard_train_path)
    X_baard = obj['X_train']
    # eg: ./results/mnist_basic_apgd2_2.0_adv.npy
    file_root = '{}_{}_apgd2_2.0'.format(data, model_name)
    path_benign = os.path.join(output_path, file_root + '_x.npy')
    path_y = os.path.join(output_path, file_root + '_y.npy')
    X_val = np.load(path_benign)
    y_val = np.load(path_y)
    n = X_val.shape[0] // 2
    X_val = X_val[n:]
    y_val = y_val[n:]

    stages = []
    stages.append(ApplicabilityStage(n_classes=10, quantile=param['q1']))
    stages.append(ReliabilityStage(n_classes=10, k=param['k_re'], quantile=param['q2']))
    stages.append(DecidabilityStage(n_classes=10, k=param['k_de'], quantile=param['q3']))
    print('BAARD: # of stages:', len(stages))

    detector = BAARDOperator(stages=stages)
    detector.stages[0].fit(X_baard, y_train)
    detector.stages[1].fit(X_train, y_train)
    detector.stages[2].fit(X_train, y_train)
    detector.search_thresholds(X_val, y_val, np.zeros_like(y_val))

    pred_X = predict_numpy(model, X, device)
    assert not np.all([pred_X, y])
    baard_label_x = detector.detect(X, y)
    pred_adv = predict_numpy(model, adv, device)
    baard_label_adv = detector.detect(adv, pred_adv)

    acc = acc_on_adv(pred_adv, y, baard_label_adv)
    print('Acc_on_adv:', acc)
    print('FPR:', np.mean(baard_label_x))

    output = {
        'X': X,
        'adv': adv,
        'y': y,
        'baard_label_x': baard_label_x,
        'baard_label_adv': baard_label_adv}
    torch.save(output, file_path)
    print('Save to:', file_path)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    with open(args.param) as param_json:
        param = json.load(param_json)
    print('Param:', param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # eg: cifar10_resnet_baard_train_surrodata_eps2.0_size2000.pt
    name_parser = args.file.split('_')
    data = name_parser[0]
    model_name = name_parser[1]

    get_baard_output(data, model_name, args.data_path, args.output_path, args.file, param, args.batch_size, device)


if __name__ == '__main__':
    main()
