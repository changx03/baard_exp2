"""
Generating CSV files for ROC AUC plots.
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
from defences.baard import (ApplicabilityStage, DecidabilityStage,
                            ReliabilityStage)
from defences.preprocess_baard import (preprocess_baard_img,
                                       preprocess_baard_numpy)
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import predict_numpy, validate
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import get_correct_examples, load_csv, set_seeds

from experiments import ATTACKS, get_advx_untargeted, get_output_path

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 128
N_SAMPLES = 2000
DEF_NAME = 'baard'
FPR_LOOKUP_LIST = np.concatenate((np.arange(0., 0.02, step=0.002), np.arange(0.02, 0.2, step=0.01), np.arange(0.2, 1., step=0.1)))


def pytorch_baard_search_param(data_name, model_name, att, eps):
    idx = 0  # Only searching in the 1st run.
    seed = SEEDS[0]
    set_seeds(seed)

    path_results = get_output_path(idx, data_name, model_name)
    if not os.path.exists(path_results):
        print('Output folder does not exist. Create:', path_results)
        path = Path(os.path.join(path_results, 'data'))
        print('Create folder:', path)
        path.mkdir(parents=True, exist_ok=True)
        path = Path(os.path.join(path_results, 'results'))
        path.mkdir(parents=True, exist_ok=True)
        print('Create folder:', path)

    # Step 1: Load data
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data_name == 'mnist':
        dataset_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)
    elif data_name == 'cifar10':
        dataset_train = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform)
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
    # Only working on validation set
    path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv_val.npy'.format(data_name, model_name, att, str(float(eps))))
    if os.path.exists(path_adv):
        print('[ATTACK] Find:', path_adv)
        adv = np.load(path_adv)
    else:
        print('[ATTACK] Start generating {} {} eps={} adversarial examples...'.format(X_val.shape[0], att, eps))
        start = time.time()
        adv = get_advx_untargeted(model, data_name, att, eps=eps, device=device, X=X_val, y=y_val, batch_size=BATCH_SIZE)
        time_elapsed = time.time() - start
        print('[ATTACK] Time spend on generating {} advx: {}'.format(len(adv), str(datetime.timedelta(seconds=time_elapsed))))
        np.save(path_adv, adv)
        print('[ATTACK] Save to', path_adv)

    pred_adv = predict_numpy(model, adv, device)
    acc_naked = np.mean(pred_adv == y_val)
    print('[ATTACK] Acc without def:', acc_naked)

    ############################################################################
    # Step 5: Train S1
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

    s1 = ApplicabilityStage(n_classes=n_classes, fpr=0., verbose=False)
    s1.fit(X_train_s1, y_train)
    param1 = FPR_LOOKUP_LIST
    tpr1 = []
    fpr1 = []
    print('[Stage 1] Start training Stage 1...')
    for i in tqdm(param1, desc='Stage 1'):
        s1.fpr = i
        s1.search_thresholds(X_val, y_val, np.zeros_like(y_val))
        output_adv = s1.predict(adv, pred_adv)
        output_X = s1.predict(X_val, y_val)
        tpr = np.mean(output_adv)
        fpr = np.mean(output_X)
        tpr1.append(tpr)
        fpr1.append(fpr)
        if tpr == 1.0:
            break
    param1 = np.append(param1[:len(tpr1)], [1.])
    tpr1 = np.append(tpr1, [1.])
    fpr1 = np.append(fpr1, [1.])
    data = {
        'param1': param1,
        'tpr': np.array(tpr1),
        'fpr': np.array(fpr1)
    }
    df = pd.DataFrame(data)
    path_s1 = os.path.join(path_results, 'results', '{}_{}_{}_{}_baard_s1_roc.csv'.format(data_name, model_name, att, str(float(eps))))
    df.to_csv(path_s1)
    print('[Stage 1] Save to:', path_s1)

    ############################################################################
    # Step 6: Train S2
    # Combine benign and adversarial examples
    X_combined = np.concatenate((adv, X_val))
    y_combined = np.concatenate((pred_adv, y_val))
    labels = np.concatenate((np.ones_like(pred_adv), np.zeros_like(y_val)))

    k_re = 10
    s2 = ReliabilityStage(n_classes=n_classes, k=k_re, fpr=0., verbose=False)
    s2.fit(X_train, y_train)
    print('[Stage 2] Start training Stage 2...')
    outputs_s2 = s2.predict_proba(X_combined, y_combined)
    fpr2, tpr2, threshold2 = roc_curve(labels, outputs_s2)
    data = {
        'threshold': threshold2,
        'tpr': tpr2,
        'fpr': fpr2
    }
    df = pd.DataFrame(data)
    path_s2 = os.path.join(path_results, 'results', '{}_{}_{}_{}_baard_s2_roc.csv'.format(data_name, model_name, att, str(float(eps))))
    df.to_csv(path_s2)
    print('[Stage 2] Save to:', path_s2)

    ############################################################################
    # Step 7: Train S3
    k_de = 100 if data_name in ['mnist', 'cifar10'] else 30
    s3 = DecidabilityStage(n_classes=n_classes, k=k_de, fpr=0., verbose=False)
    s3.fit(X_train, y_train)
    print('[Stage 2] Start training Stage 3...')
    outputs_s3 = s3.predict_proba(X_combined, y_combined)
    fpr3, tpr3, threshold3 = roc_curve(labels, outputs_s3)
    data = {
        'threshold': threshold3,
        'tpr': tpr3,
        'fpr': fpr3
    }
    df = pd.DataFrame(data)
    path_s3 = os.path.join(path_results, 'results', '{}_{}_{}_{}_baard_s3_roc.csv'.format(data_name, model_name, att, str(float(eps))))
    df.to_csv(path_s3)
    print('[Stage 3] Save to:', path_s3)

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-m', '--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('-a', '--attack', type=str, default='fgsm', choices=ATTACKS)
    parser.add_argument('-e', '--eps', type=float, default=0.3, required=True)
    args = parser.parse_args()
    print('args:', args)

    seed = SEEDS[0]
    data = args.data
    model_name = args.model
    att = args.attack
    eps = args.eps
    print('seed:', seed)
    print('data:', data)
    print('model_name:', model_name)
    print('attack:', att)
    print('eps:', eps)
    pytorch_baard_search_param(data, model_name, att, eps)

    # Testing
    # pytorch_baard_search_param('mnist', 'dnn', 'apgd', 0.3)
    # pytorch_baard_search_param('mnist', 'dnn', 'apgd2', 2.)
    # pytorch_baard_search_param('cifar10', 'resnet', 'apgd', 0.3)
    # pytorch_baard_search_param('cifar10', 'resnet', 'apgd2', 2.)
    # pytorch_baard_search_param('banknote', 'dnn', 'apgd', 0.3)
    # pytorch_baard_search_param('banknote', 'dnn', 'apgd2', 2.)
    # pytorch_baard_search_param('breastcancer', 'dnn', 'apgd', 0.3)
    # pytorch_baard_search_param('breastcancer', 'dnn', 'apgd2', 2.)
    # pytorch_baard_search_param('htru2', 'dnn', 'apgd', 0.3)
    # pytorch_baard_search_param('htru2', 'dnn', 'apgd2', 2.)
