"""
Find best K value while holding FPR as a constant
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
BATCH_SIZE = 128
N_SAMPLES = 2000
DEF_NAME = 'baard'
FPR_MNIST = 0.01
FPR_CIFAR10 = 0.01

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

def find_k(data_name, model_name, epsilons, idx=0):
    seed = SEEDS[idx]
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
    X_att = X_test[:n]
    y_att = y_test[:n]
    X_val = X_test[n:]
    y_val = y_test[n:]

    ############################################################################
    # Step 4: Load attack
    X = X_val.copy()
    y = y_val.copy()
    for e in epsilons:
        adv = get_advx(data_name, model_name, path_results, model, X_val, y_val, 'apgd', e, device)
        X = np.append(X, adv, axis=0)
        y = np.append(y, y_val)
    
    print('X', X.shape)
    print('y', y.shape)
    # all samples in X are advx except the first n clean samples.
    target = np.ones(X.shape[0])
    target[:X_val.shape[0]] = 0

    if data_name == 'mnist':
        fpr = FPR_MNIST
    elif data_name == 'cifar10':
        fpr = FPR_CIFAR10
    else:
        raise NotImplementedError()

    pred = predict_numpy(model, X, device)

    ############################################################################
    # Step 5: Run BAARD Stage 2
    # ks = [1, 2, ..., 50]
    ks = np.array(list(range(0, 30, 2)))
    ks[0] = 1
    accs = np.zeros_like(ks, dtype=np.float)
    fprs = np.zeros_like(ks, dtype=np.float)
    print('Stage 2:')
    for i, k in enumerate(ks):
        s2 = ReliabilityStage(10, k, fpr, verbose=False)
        s2.fit(X_train, y_train)
        # Use a different set
        s2.search_thresholds(X_att, y_att, np.zeros_like(y_val))

        labelled_as_adv = s2.predict(X, pred)
        acc = acc_on_advx(pred[1000:], y[1000:], labelled_as_adv[1000:])
        fpr = np.mean(labelled_as_adv[:1000])
        print('k:{}, acc:{}, fpr:{}'.format(k, acc, fpr))

        accs[i] = acc
        fprs[i] = fpr
    
    ############################################################################
    # Step 6: Save results for Stage 2
    data = {
        'k': ks,
        'acc': accs,
        'fpr': fprs,
    }
    df = pd.DataFrame(data)
    path_csv = os.path.join(path_results, 'results', '{}_{}_k_s2.csv'.format(data_name, model_name))
    df.to_csv(path_csv)
    print('Save to:', path_csv)

    ############################################################################
    # Step 7: Run BAARD Stage 3
    # ks = [10, 20, ..., 200]
    ks = np.array(list(range(10, 200, 10)))
    accs = np.zeros_like(ks, dtype=np.float)
    fprs = np.zeros_like(ks, dtype=np.float)
    print('Stage 3:')
    for i, k in enumerate(ks):
        s3 = DecidabilityStage(10, k, fpr, verbose=False)
        s3.fit(X_train, y_train)
        # Use a different set
        s3.search_thresholds(X_att, y_att, np.zeros_like(y_val))

        labelled_as_adv = s3.predict(X, pred)
        acc = acc_on_advx(pred[1000:], y[1000:], labelled_as_adv[1000:])
        fpr = np.mean(labelled_as_adv[:1000])
        print('k:{}, acc:{}, fpr:{}'.format(k, acc, fpr))

        accs[i] = acc
        fprs[i] = fpr
    
    ############################################################################
    # Step 8: Save results for Stage 3
    data = {
        'k': ks,
        'acc': accs,
        'fpr': fprs,
    }
    df = pd.DataFrame(data)
    path_csv = os.path.join(path_results, 'results', '{}_{}_k_s3.csv'.format(data_name, model_name))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('-e', '--eps', type=float, required=True, nargs='+')
    parser.add_argument('-i', '--idx', type=int, required=True)
    args = parser.parse_args()
    print('args:', args)

    data_name = args.data
    model_name = args.model
    epsilons = args.eps
    idx = args.idx
    print('data:', data_name)
    print('model_name:', model_name)
    print('epsilons:', epsilons)
    print('idx:', idx)

    find_k(data_name, model_name, epsilons, idx)

    # find_k('mnist', 'dnn', [0.063, 0.3, 1.0, 2.0], 0)
    # find_k('cifar10', 'resnet', [0.031, 0.3, 1.0, 2.0], 0)


# python3 ./experiments/find_k.py -d mnist -m dnn -e 0.063 0.3 1.0 2.0 -i 0
# python3 ./experiments/find_k.py -d cifar10 -m resnet -e 0.031 0.3 1.0 2.0 -i 0
