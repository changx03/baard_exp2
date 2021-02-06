import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from defences.util import acc_on_adv, get_correct_examples
from misc.util import set_seeds
from models.torch_util import predict_numpy, validate

from pipeline.run_attack import ATTACKS
from pipeline.train_defence import train_magnet
from pipeline.train_model import train_model

# from pipeline.train_surrogate import get_pretrained_surrogate, train_surrogate

PATH_DATA = 'data'
EPOCHS = 200
DEFENCE = 'magnet'

# TODO: fix this file!


def run_evaluate_magnet(data,
                        model_name,
                        path,
                        seed,
                        json_param,
                        att_name,
                        eps):
    set_seeds(seed)

    # Line attack takes no hyperparameter
    if att_name == 'line':
        eps = [1]
    print('args:', data, model_name, path, seed, json_param, att_name, eps)

    if not os.path.exists(path):
        print('Output folder does not exist. Create:', path)
        os.mkdir(path)

    # Get data
    n_classes = 10
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data == 'mnist':
        dataset_train = datasets.MNIST(PATH_DATA, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(PATH_DATA, train=False, download=True, transform=transform)
    elif data == 'cifar10':
        transform_train = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor()])
        dataset_train = datasets.CIFAR10(PATH_DATA, train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10(PATH_DATA, train=False, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(data))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    file_model = os.path.join(path, '{}_{}_model.pt'.format(data, model_name))
    print('Start training {} model on {}...'.format(model_name, data))
    model = train_model(data, model_name, dataset_train, dataset_test, EPOCHS, device, file_model)

    # Split data
    tensor_X, tensor_y = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
    dataset = TensorDataset(tensor_X, tensor_y)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc_perfect = validate(model, loader, nn.CrossEntropyLoss(), device)
    print('Accuracy on {} filtered test set: {:.2f}%'.format(tensor_y.size(0), acc_perfect * 100))

    print('-------------------------------------------------------------------')
    print('Start training MagNet...')
    # Run preprocessing
    tensor_X, tensor_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    X_train = tensor_X.cpu().detach().numpy()
    y_train = tensor_y.cpu().detach().numpy()

    # We need load the evaluation set first. The clean samples are all the same,
    # which attack is selected does not matter.
    file_data = os.path.join(path, '{}_{}_{}_{}.pt'.format(data, model_name, att_name, int(eps[0] * 1000)))
    obj = torch.load(file_data)
    X = obj['X']
    y = obj['y']
    X_def_val = X[1000:2000]

    with open(json_param) as j:
        param = json.load(j)

    time_start = time.time()
    detector = train_magnet(data, model_name, X_train, y_train, X_def_val, param, device, path, EPOCHS, model=model)
    time_elapsed = time.time() - time_start
    print('Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

    print('-------------------------------------------------------------------')
    print('Start evaluating the robustness of the classifier...')

    eps = np.array(eps, dtype=np.float)
    n_att = eps.shape[0]
    accs_classifier = np.zeros(n_att, dtype=np.float)
    accs_on_adv = np.zeros_like(accs_classifier)
    fprs = np.zeros_like(accs_on_adv)

    pred = predict_numpy(model, X, device)
    print('Acc on clean samples:', np.mean(pred == y))

    for i in range(n_att):
        print('Evaluating {} eps={}'.format(att_name, eps[i]))
        file_data = os.path.join(path, '{}_{}_{}_{}.pt'.format(data, model_name, att_name, round(eps[i] * 1000)))
        obj = torch.load(file_data)
        adv = obj['adv']

        X_def_test = X[:1000]
        y_def_test = y[:1000]
        adv_def_test = adv[:1000]
        pred_adv_def_test = pred[:1000]

        pred = predict_numpy(model, adv_def_test, device)
        acc_base = np.mean(pred == y_def_test)

        X_reformed, labelled_as_adv = detector.detect(adv_def_test, pred_adv_def_test)
        pred_reformed = predict_numpy(model, X_reformed, device)
        acc_def = acc_on_adv(pred_reformed, y_def_test, labelled_as_adv)

        _, labelled_false = detector.detect(X_def_test, y_def_test)
        fpr = np.mean(labelled_false)

        print('acc_model: {:.4f}, acc_on_adv: {:.4f}, fpr: {:.4f}'.format(acc_base, acc_def, fpr))
        accs_classifier[i] = acc_base
        accs_on_adv[i] = acc_def
        fprs[i] = fpr

    results = np.array([eps, accs_classifier, accs_on_adv, fprs]).transpose()
    df = pd.DataFrame(data=results, columns=['eps', 'acc_base', 'acc_on_adv', 'fpr'])
    file_output = os.path.join(path, '{}_{}_{}_{}.csv'.format(data, model_name, DEFENCE, att_name))
    df.to_csv(file_output, index=False)
    print('Saved results to:', file_output)

    print('DONE!')
    print('-------------------------------------------------------------------\n')


if __name__ == '__main__':
    path_cur = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_cur, 'seeds.json')) as j:
        json_obj = json.load(j)
        seeds = json_obj['seeds']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('--attack', type=str, default='apgd2', choices=ATTACKS)
    parser.add_argument('--eps', type=float, default=[2.0], nargs='+')
    path_json_magnet = os.path.join('params', 'magnet_param.json')
    parser.add_argument('--json', type=str, default=path_json_magnet)
    parser.add_argument('--idx', type=int, default=0, choices=list(range(len(seeds))))
    args = parser.parse_args()
    print(args)

    idx = args.idx
    run_evaluate_magnet(
        data=args.data,
        model_name=args.model,
        path='result_{}'.format(str(idx)),
        seed=seeds[idx],
        json_param=args.json,
        att_name=args.attack,
        eps=args.eps)
