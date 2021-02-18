"""
Evaluate Region-based Classifier against adversarial attacks on image datasets. 
"""
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
from defences.region_based_classifier import RegionBasedClassifier
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.torch_util import predict_numpy, validate
from torch.utils.data import DataLoader, TensorDataset
from utils import acc_on_advx, get_correct_examples, set_seeds

from experiments import (ATTACKS, get_advx_untargeted, get_output_path,
                         pytorch_train_classifier)

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 192
EPOCHS = 200
N_SAMPLES = 2000
DEF_NAME = 'rc'


def pytorch_attack_against_rc_img(data_name, model_name, att, epsilons, idx):
    print('Runing pytorch_attack_against_rc_img.py')
    seed = SEEDS[idx]
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

    # Step 1 Load data
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data_name == 'mnist':
        dataset_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)
    elif data_name == 'cifar10':
        dataset_train = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(DATA_PATH, train=False, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(data_name))

    ############################################################################
    # Step 2: Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[CLASSIFIER] Device: {}'.format(device))

    file_model = os.path.join(path_results, 'data', '{}_{}_model.pt'.format(data_name, model_name))
    if not os.path.exists(file_model):
        pytorch_train_classifier(data_name, model_name, idx)

    if data_name == 'mnist':
        model = BaseModel(use_prob=False).to(device)
    elif data_name == 'cifar10':
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
        else:
            raise NotImplementedError
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

        # Shuffle indices before saving
        idx_shuffle = np.random.permutation(X_test.shape[0])
        X_test = X_test[idx_shuffle]
        y_test = y_test[idx_shuffle]
        
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
    n = N_SAMPLES // 2
    print('[DATA] n:', n)
    X_att = X_test[:n]
    y_att = y_test[:n]
    X_val = X_test[n:]
    y_val = y_test[n:]

    ############################################################################
    # Step 4: Load detector
    detector = RegionBasedClassifier(
        model=model,
        r=0.2,
        sample_size=1000,
        n_classes=10,
        x_min=0.0,
        x_max=1.0,
        r0=0.0,
        step_size=0.02,
        stop_value=0.4,
        device=device)

    path_best_r = os.path.join(path_results, 'results', '{}_{}_rc_r.json'.format(data_name, model_name))
    if os.path.exists(path_best_r):
        with open(path_best_r) as j:
            obj_r = json.load(j)
        r_best = obj_r['r']
        print('[DETECTOR] Find:', path_best_r)
    else:
        print('[DETECTOR] Cannot found:', path_best_r)
        print('[DETECTOR] Start searching r_best...')
        time_start = time.time()
        r_best = detector.search_thresholds(X_val, y_val, np.zeros_like(y_val), verbose=0)
        time_elapsed = time.time() - time_start
        print('[DETECTOR] Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
        with open(path_best_r, 'w') as f:
            json.dump({'r': r_best}, f)
            print('Save to:', path_best_r)
    print('[DETECTOR] r_best:', r_best)
    detector.r = r_best

    ############################################################################
    # Step 5: Generate attack and preform defence
    if att == 'boundary':
        epsilons = [0.]

    # Adversarial examples come from the same benign set
    pred_benign = detector.detect(X_att, y_att)
    fpr = np.mean(pred_benign != y_att)

    accuracies_no_def = []
    acc_on_advs = []
    for e in epsilons:
        try:
            path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv.npy'.format(data_name, model_name, att, str(float(e))))
            if os.path.exists(path_adv):
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
            print('[DEFENCE] Start running RC...')
            start = time.time()
            pred_adv = detector.detect(adv, pred_adv)
            time_elapsed = time.time() - start
            print('[DEFENCE] Time spend:', str(datetime.timedelta(seconds=time_elapsed)))

            labels = np.zeros_like(pred_adv)
            acc = acc_on_advx(pred_adv, y_att, labels)

            print('[DEFENCE] acc_on_adv:', acc)
            print('[DEFENCE] fpr:', fpr)
        except Exception as e:
            print('[ERROR]', e)
            acc_naked = np.nan
            acc = np.nan
        finally:
            accuracies_no_def.append(acc_naked)
            acc_on_advs.append(acc)

    data = {
        'data': np.repeat(data_name, len(epsilons)),
        'model': np.repeat(model_name, len(epsilons)),
        'attack': np.repeat(att, len(epsilons)),
        'adv_param': np.array(epsilons),
        'acc_no_def': np.array(accuracies_no_def),
        'acc_on_adv': np.array(acc_on_advs),
        'fpr': np.repeat(fpr, len(epsilons))}
    df = pd.DataFrame(data)
    path_csv = os.path.join(path_results, 'results', '{}_{}_{}_{}.csv'.format(data_name, model_name, att, DEF_NAME))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-m', '--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('-i', '--idx', type=int, default=0, choices=list(range(len(SEEDS))))
    parser.add_argument('-a', '--att', type=str, default='fgsm', choices=ATTACKS)
    parser.add_argument('-e', '--eps', type=float, default=[0.3], nargs='+')
    args = parser.parse_args()
    print('args:', args)

    idx = args.idx
    data = args.data
    model_name = args.model
    att = args.att
    epsilons = args.eps
    seed = SEEDS[args.idx]
    print('data:', data)
    print('model_name:', model_name)
    print('attack:', att)
    print('epsilons:', epsilons)
    print('seed:', seed)
    pytorch_attack_against_rc_img(data, model_name, att, epsilons, idx)

    # Testing
    # pytorch_attack_against_rc_img('mnist', 'dnn', 'apgd', [0.3], 0)
    # pytorch_attack_against_rc_img('mnist', 'dnn', 'apgd2', [2.0], 0)

# python ./experiments/pytorch_attack_against_rc_img.py -d mnist -i 0 -a apgd -e 0.3 1.0
