import os
import sys

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

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
from models.torch_util import predict_numpy, validate
from torch.utils.data import DataLoader, TensorDataset
from utils import acc_on_advx, get_correct_examples, set_seeds

from experiments import (ATTACKS, get_advx_untargeted, get_baard,
                         get_output_path, train_model)

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 128
N_SAMPLES = 2000
DEF_NAME = 'baard'


def pytorch_attack_against_baard(data_name, model_name, att, epsilons, idx, baard_param, fresh_att=False, fresh_def=True):
    seed = SEEDS[idx]
    set_seeds(seed)

    if att == 'boundary':
        epsilons = [0.]

    path_results = get_output_path(idx, data_name, model_name)
    if not os.path.exists(path_results):
        print('Output folder does not exist. Create:', path_results)
        path = Path(os.path.join(path_results, 'data'))
        print('Create folder:', path)
        path.mkdir(parents=True, exist_ok=True)
        path = Path(os.path.join(path_results, 'results'))
        path.mkdir(parents=True, exist_ok=True)
        print('Create folder:', path)

    # Prepare data
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
        raise ValueError('Unknown dataset: {}'.format(data_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Get classifier
    file_model = os.path.join(path_results, 'data', '{}_{}_model.pt'.format(data_name, model_name))
    print('[CLASSIFIER] Start training {} model on {}...'.format(model_name, data_name))
    start = time.time()
    model = train_model(data_name, model_name, dataset_train, dataset_test, device, file_model)
    time_elapsed = time.time() - start
    print('[CLASSIFIER] Time spend on training classifier: {}'.format(str(datetime.timedelta(seconds=time_elapsed))))

    # Split data
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
    n = N_SAMPLES // 2
    print('[DATA] n:', n)
    X_att = X_test[:n]
    y_att = y_test[:n]
    X_val = X_test[n:]
    y_val = y_test[n:]

    detector = get_baard(
        data_name=data_name,
        model_name=model_name,
        idx=idx,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        baard_param=baard_param,
        restart=fresh_def)

    accuracies_no_def = []
    acc_on_advs = []
    fprs = []
    for e in epsilons:
        try:
            path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv.npy'.format(data_name, model_name, att, str(float(e))))
            if os.path.exists(path_adv) and not fresh_att:
                print('[ATTACK] Find:', path_adv)
                adv = np.load(path_adv)
            else:
                print('[ATTACK] Start generating {} {} eps={} adversarial examples...'.format(X_att.shape[0], att, e))
                start = time.time()
                adv = get_advx_untargeted(model, data_name, att, eps=e, device=device, X=X_att, y=y_att, batch_size=BATCH_SIZE)
                time_elapsed = time.time() - start
                print('[ATTACK] Time spend on generating {} advx: {}'.format(len(adv), str(datetime.timedelta(seconds=time_elapsed))))
                np.save(path_adv, adv)
                print('[ATTACK] Save to', path_adv)

            pred_adv = predict_numpy(model, adv, device)
            acc_naked = np.mean(pred_adv == y_att)
            print('[ATTACK] Acc without def:', acc_naked)

            # Preform defence
            print('[DEFENCE] Start running BAARD...')
            start = time.time()
            labelled_as_adv = detector.detect(adv, y_att)
            labelled_benign_as_adv = detector.detect(X_att, y_att)
            time_elapsed = time.time() - start
            print('[DEFENCE] Time spend:', str(datetime.timedelta(seconds=time_elapsed)))

            acc = acc_on_advx(pred_adv, y_att, labelled_as_adv)
            fpr = np.mean(labelled_benign_as_adv)

            print('[DEFENCE] acc_on_adv:', acc)
            print('[DEFENCE] fpr:', fpr)
        except Exception as e:
            print('[ERROR]', e)
            acc_naked = np.nan
            acc = np.nan
            fpr = np.nan
        finally:
            accuracies_no_def.append(acc_naked)
            acc_on_advs.append(acc)
            fprs.append(fpr)

    data = {
        'data': np.repeat(data_name, len(epsilons)),
        'model': np.repeat(model_name, len(epsilons)),
        'attack': np.repeat(att, len(epsilons)),
        'adv_param': np.array(epsilons),
        'acc_no_def': np.array(accuracies_no_def),
        'acc_on_adv': np.array(acc_on_advs),
        'fpr': np.array(fprs)
    }
    df = pd.DataFrame(data)
    path_csv = os.path.join(path_results, 'results', '{}_{}_{}_{}_{}.csv'.format(data_name, model_name, att, DEF_NAME, len(detector.stages)))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-m', '--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('-i', '--idx', type=int, default=0, choices=list(range(len(SEEDS))))
    parser.add_argument('-a', '--attack', type=str, default='fgsm', choices=ATTACKS)
    parser.add_argument('-e', '--eps', type=float, default=[0.3], nargs='+')
    parser.add_argument('-p', '--param', type=str, required=True)
    args = parser.parse_args()
    print('args:', args)

    idx = args.idx
    data = args.data
    model_name = args.model
    att = args.attack
    epsilons = args.eps
    seed = SEEDS[args.idx]
    print('data:', data)
    print('model_name:', model_name)
    print('attack:', att)
    print('epsilons:', epsilons)
    print('seed:', seed)
    pytorch_attack_against_baard(data, model_name, att, epsilons, idx)

# # Testing
# if __name__ == '__main__':
#     pytorch_attack_against_baard('mnist', 'dnn', 'fgsm', [0.3], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'apgd', [0.3], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'apgd2', [2.0], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'bim', [0.3], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'boundary', [0.], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'cw2', [0.], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'cwinf', [10.], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'deepfool', [1e-6], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
#     pytorch_attack_against_baard('mnist', 'dnn', 'line', [1.], 10, './params/baard_mnist_3.json', fresh_att=False, fresh_def=True)
