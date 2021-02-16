"""
Runing grid search for BAARD's hyperparameters on PyTorch models.
"""
import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from defences.preprocess_baard import (preprocess_baard_img,
                                       preprocess_baard_numpy)
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import predict_numpy, validate
from sklearn.metrics import roc_curve
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import get_correct_examples, load_csv, set_seeds, acc_on_advx

from experiments import ATTACKS, get_advx_untargeted, get_output_path

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 192
N_SAMPLES = 2000
DEF_NAME = 'baard'
PARAM_GRID = ParameterGrid({
    'fpr1': [0., 0.001, 0.01, 0.02],
    'fpr2': [0., 0.001, 0.01, 0.02],
    'fpr3': [0., 0.001, 0.01, 0.02]
})


def get_advx(data_name, model_name, path_results, model, X, y, att, eps, device):
    path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv_val.npy'.format(data_name, model_name, att, str(float(eps))))
    if os.path.exists(path_adv):
        print('[ATTACK] Find:', path_adv)
        adv = np.load(path_adv)
    else:
        print('[ATTACK] Start generating {} {} eps={} adversarial examples...'.format(X.shape[0], att, eps))
        start = time.time()
        adv = get_advx_untargeted(model, data_name, att, eps=eps, device=device, X=X, y=y, batch_size=BATCH_SIZE)
        time_elapsed = time.time() - start
        print('[ATTACK] Time spend on generating {} advx: {}'.format(len(adv), str(datetime.timedelta(seconds=time_elapsed))))
        np.save(path_adv, adv)
        print('[ATTACK] Save to', path_adv)
    return adv


def baard_grid_search(data_name, model_name):
    idx = 0  # Only searching in the 1st run.
    seed = SEEDS[0]
    set_seeds(seed)

    # Step 1 Load data
    # Create folders
    path_results = get_output_path(idx, data_name, model_name)
    if not os.path.exists(path_results):
        print('Output folder does not exist. Create:', path_results)
        path = Path(os.path.join(path_results, 'data'))
        print('Create folder:', path)
        path.mkdir(parents=True, exist_ok=True)
        path = Path(os.path.join(path_results, 'results'))
        path.mkdir(parents=True, exist_ok=True)
        print('Create folder:', path)

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
        print('Read file:', data_path)
        X, y = load_csv(data_path)
        scalar = MinMaxScaler().fit(X, y)
        X = scalar.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=seed)
        dataset_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        dataset_test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    ############################################################################
    # Step 2: Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[CLASSIFIER] Device: {}'.format(device))

    file_model = os.path.join(path_results, 'data', '{}_{}_model.pt'.format(data_name, model_name))
    if not os.path.exists(file_model):
        raise FileNotFoundError('Cannot find pretrained model: {}'.format(file_model))
    if data_name == 'mnist':
        model = BaseModel(use_prob=False).to(device)
    elif data_name == 'cifar10':
        if model_name == 'resnet':
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
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # loss = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(file_model, map_location=device))

    ############################################################################
    # Step 3: Filter data
    path_X_train = os.path.join(path_results, 'data', '{}_{}_X_train.npy'.format(data_name, model_name))
    if os.path.exists(path_X_train):
        print('[DATA] Found existing data:', path_X_train)
        X_train = np.load(path_X_train)
        y_train = np.load(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)))
        X_test = np.load(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)))
        y_test = np.load(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)))
    else:
        tensor_X_train, tensor_y_train = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
        tensor_X_test, tensor_y_test = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
        X_train = tensor_X_train.cpu().detach().numpy()
        y_train = tensor_y_train.cpu().detach().numpy()
        X_test = tensor_X_test.cpu().detach().numpy()
        y_test = tensor_y_test.cpu().detach().numpy()
        np.save(path_X_train, X_train)
        np.save(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)), X_test)
        np.save(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)), y_train)
        np.save(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)), y_test)
        print('[DATA] Save to:', path_X_train)

    dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    _, acc_perfect = validate(model, loader, nn.CrossEntropyLoss(), device)
    print('[DATA] Accuracy on {} filtered test set: {:.2f}%'.format(y_test.shape[0], acc_perfect * 100))

    # Split rules:
    # 1. Benchmark_defence_test: 1000 (def_test)
    # 2. Benchmark_defence_val:  1000 (def_val)
    idx_shuffle = np.random.permutation(X_test.shape[0])[:N_SAMPLES]
    X_test = X_test[idx_shuffle]
    y_test = y_test[idx_shuffle]
    # How many examples do we have?
    if len(X_test) > N_SAMPLES:
        n = N_SAMPLES
    else:
        n = len(X_test)
    n = n // 2
    print('[DATA] n:', n)
    # X_att = X_test[:n]
    # y_att = y_test[:n]
    X_val = X_test[n:]
    y_val = y_test[n:]

    ############################################################################
    # Step 4: Load attack
    # Combine APGD L-inf and APGD L2
    start = time.time()
    adv1 = get_advx(data_name, model_name, path_results, model, X_val, y_val, 'apgd', 0.3, device)
    adv2 = get_advx(data_name, model_name, path_results, model, X_val, y_val, 'apgd2', 2.0, device)
    time_elapsed = time.time() - start
    print('[ATTACK] Time spend on training advx: {}'.format(str(datetime.timedelta(seconds=time_elapsed))))

    adv = np.concatenate((adv1, adv2))
    y_adv = np.concatenate((y_val, y_val))
    pred_adv = predict_numpy(model, adv, device)
    ############################################################################
    # Step 5: Run BAARD in grid search
    # Initialize BAARD
    file_baard_train = os.path.join(path_results, 'data', '{}_{}_baard_s1_train_data.pt'.format(data_name, model_name))
    if os.path.exists(file_baard_train):
        print('[DEFENCE] Found existing BAARD preprocess data:', file_baard_train)
        obj = torch.load(file_baard_train)
        X_train_s1 = obj['X_s1']
        X_train = obj['X']
        y_train = obj['y']
    else:
        if data_name in ['mnist', 'cifar10']:
            X_train_s1 = preprocess_baard_img(data_name, torch.from_numpy(X_train))
            X_train_s1 = X_train_s1.cpu().detach().numpy()
        else:
            X_train_s1 = preprocess_baard_numpy(X_train)

        obj = {
            'X_s1': X_train_s1,
            'X': X_train,
            'y': y_train
        }
        torch.save(obj, file_baard_train)
        print('[DEFENCE] Save to:', file_baard_train)
    assert X_train_s1.shape == X_train.shape
    print('[DEFENCE] X_train_s1', X_train_s1.shape)
    n_classes = len(np.unique(y_train))
    print('[DEFENCE] n_classes:', n_classes)

    k_re = 10
    k_de = 100 if data_name in ['mnist', 'cifar10'] else 30

    fpr1s = np.zeros(len(list(PARAM_GRID)), dtype=np.float)
    fpr2s = np.zeros(len(list(PARAM_GRID)), dtype=np.float)
    fpr3s = np.zeros(len(list(PARAM_GRID)), dtype=np.float)
    accuracies = np.zeros(len(list(PARAM_GRID)), dtype=np.float)
    fprs = np.ones(len(list(PARAM_GRID)), dtype=np.float)
    start = time.time()
    for i, param in enumerate(tqdm(PARAM_GRID)):
        print('[BAARD] param:', param)
        fpr1s[i] = param['fpr1']
        fpr2s[i] = param['fpr2']
        fpr3s[i] = param['fpr3']
        stages = []
        stages.append(ApplicabilityStage(n_classes=n_classes, fpr=fpr1s[i], verbose=False))
        stages.append(ReliabilityStage(n_classes=n_classes, k=k_re, fpr=fpr2s[i], verbose=False))
        stages.append(DecidabilityStage(n_classes=n_classes, k=k_de, fpr=fpr3s[i], verbose=False))
        detector = BAARDOperator(stages=stages)
        detector.fit(X_train, y_train, X_train_s1)
        detector.search_thresholds(X_val, y_val)

        labelled_as_adv = detector.detect(adv, pred_adv)
        acc = acc_on_advx(pred_adv, y_adv, labelled_as_adv)
        labelled_benign_as_adv = detector.detect(X_val, y_val)
        fpr = np.mean(labelled_benign_as_adv)

        accuracies[i] = acc
        fprs[i] = fpr
    time_elapsed = time.time() - start
    print('[BAARD] Time spend on grid search: {}'.format(str(datetime.timedelta(seconds=time_elapsed))))
    ############################################################################
    # Step 6: Save results
    data = {
        'k_re': np.repeat(k_re, len(accuracies)),
        'k_de': np.repeat(k_de, len(accuracies)),
        'fpr1': fpr1s,
        'fpr2': fpr2s,
        'fpr3': fpr3s,
        'acc_on_adv': accuracies,
        'fpr': fprs
    }
    df = pd.DataFrame(data)
    path_csv = os.path.join(path_results, 'results', '{}_{}_baard_grid_search.csv'.format(data_name, model_name))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data', type=str, required=True)
    # parser.add_argument('-m', '--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    # args = parser.parse_args()
    # print('args:', args)

    # seed = SEEDS[0]
    # data_name = args.data
    # model_name = args.model
    # print('seed:', seed)
    # print('data:', data_name)
    # print('model_name:', model_name)

    # baard_grid_search(data_name, model_name)

    # Testing
    # baard_grid_search('banknote', 'dnn')
    baard_grid_search('breastcancer', 'dnn')
    # baard_grid_search('htru2', 'dnn')
    # baard_grid_search('mnist', 'dnn')
    # baard_grid_search('cifar10', 'resnet')
    # baard_grid_search('cifar10', 'vgg')
