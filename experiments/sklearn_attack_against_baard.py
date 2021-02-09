import os
import sys
from operator import mod

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
from art.attacks.evasion import (BasicIterativeMethod, BoundaryAttack,
                                 DecisionTreeAttack, FastGradientMethod)
from art.estimators.classification import SklearnClassifier
from defences import (ApplicabilityStage, BAARDOperator, DecidabilityStage,
                      ReliabilityStage)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from utils import acc_on_advx, load_csv, set_seeds

ATTACKS = ['bim', 'fgsm', 'boundary', 'tree']
DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 128
N_SAMPLES = 2000
DEF_NAME = 'rc'


def get_output_path(i, data, model_name):
    return os.path.join('results', 'result_' + str(i), '{}_{}'.format(
        data, model_name))


def get_attack(att_name, classifier, eps=None):
    if att_name == 'bim':
        eps_step = eps / 10.0
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=100,
            targeted=False,
            batch_size=BATCH_SIZE)
    elif att_name == 'boundary':
        attack = BoundaryAttack(
            estimator=classifier,
            max_iter=1000,
            targeted=False)
    elif att_name == 'fgsm':
        attack = FastGradientMethod(
            estimator=classifier,
            eps=eps,
            batch_size=BATCH_SIZE)
    elif att_name == 'tree':
        attack = DecisionTreeAttack(
            classifier=classifier)
    else:
        raise NotImplementedError
    return attack


def get_correct_examples_sklearn(estimator, X, y):
    pred = estimator.predict(X)
    idx = np.where(pred == y)[0]
    return X[idx], y[idx]


def preprocess_baard(X, std=1., eps=0.025):
    X_noisy = X + eps * np.random.normal(loc=0, scale=std, size=X.size())
    return X_noisy


def get_defence(data_name, model_name, idx, model, X_train, y_train, X_val, y_val, baard_param):
    # Prepare training data
    path_results = get_output_path(idx, data_name, model_name)

    file_baard_train = os.path.join(path_results, '{}_{}_baard_s1_train_data.pt'.format(
        data, model_name))
    if os.path.exists(file_baard_train):
        print('Found existing BAARD preprocess data:', file_baard_train)
        obj = torch.load(file_baard_train)
        X_train_s1 = obj['X_s1']
        X_train = obj['X']
        y_train = obj['y']
    else:
        X_train_s1 = preprocess_baard(X_train)
        obj = {
            'X_s1': X_train_s1,
            'X': X_train,
            'y': y_train
        }
        torch.save(obj, file_baard_train)
        print('Save to:', file_baard_train)
    assert X_train_s1.shape == X_train.shape
    print('X_train_s1', X_train_s1.shape)
    n_classes = len(np.unique(y_train))
    print('n_classes:', n_classes)

    # Load each stage
    with open(baard_param) as j:
        baard_param = json.load(j)
    print('Param:', baard_param)
    sequence = baard_param['sequence']
    stages = []
    if sequence[0]:
        s1 = ApplicabilityStage(n_classes=n_classes, quantile=baard_param['q1'])
        s1.fit(X_train_s1, y_train)
        stages.append(s1)
    if sequence[1]:
        s2 = ReliabilityStage(n_classes=n_classes, k=baard_param['k_re'], quantile=baard_param['q2'])
        s2.fit(X_train, y_train)
        stages.append(s2)
    if sequence[2]:
        s3 = DecidabilityStage(n_classes=n_classes, k=baard_param['k_de'], quantile=baard_param['q3'])
        s3.fit(X_train, y_train)
        stages.append(s3)
    print('BAARD stages:', len(stages))
    detector = BAARDOperator(stages=stages)

    # Set thresholds
    file_baard_threshold = os.path.join(path_results, '{}_{}_baard_threshold.pt'.format(
        data, model_name))
    if os.path.exists(file_baard_threshold):
        print('Found existing BAARD thresholds:', file_baard_threshold)
        detector.load(file_baard_threshold)
    else:
        # Search thresholds
        detector.search_thresholds(X_val, y_val, np.zeros_like(y_val))
        detector.save(file_baard_threshold)
    return detector


def sklearn_attack_against_baard(data_name, model_name, att, epsilons, idx, baard_param):
    seed = SEEDS[idx]

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
    data_path = os.path.join(DATA_PATH, METADATA['data'][data_name]['file_name'])
    print('Read file: {}'.format(data_path))
    X, y = load_csv(data_path)
    n_classes = len(np.unique(y))

    # Apply scaling
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    path_results = get_output_path(idx, data_name, model_name)
    path_X_train = os.path.join(path_results, 'data', '{}_{}_X_train.npy'.format(
        data_name, model_name))
    if os.path.exists(path_X_train):
        print('Found existing data:', path_X_train)
        X_train = np.load(path_X_train)
        X_test = np.load(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(
            data_name, model_name)))
        y_train = np.load(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format
                                       (data_name, model_name)))
        y_test = np.load(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(
            data_name, model_name)))
    else:
        print('Cannot found:', path_X_train)
        n_test = METADATA['data'][data_name]['n_test']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=seed)
        np.save(path_X_train, X_train)
        np.save(os.path.join(
            path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)), X_test)
        np.save(os.path.join(
            path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)), y_train)
        np.save(os.path.join(
            path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)), y_test)
        print('Save to:', path_X_train)

    # Train model
    if model_name == 'svm':
        model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=seed)
    elif model_name == 'tree':
        model = ExtraTreeClassifier(random_state=seed)
    else:
        raise NotImplementedError
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print(('Train Acc: {:.4f}, ' + 'Test Acc: {:.4f}').format(acc_train, acc_test))

    # Get perfect subset
    X_test, y_test = get_correct_examples_sklearn(model, X_test, y_test)

    # How many examples do we have?
    if len(X_test) > N_SAMPLES:
        n = N_SAMPLES
    else:
        n = len(X_test)
    # X_benign = X_test[:n]
    # y_true = y_test[:n]
    n = n // 2
    print('n:', n)
    X_att = X_test[:n]
    y_att = y_test[:n]
    X_val = X_test[n:]
    y_val = y_test[n:]

    # Train defence
    print('Start training BAARD...')
    X_train, y_train = get_correct_examples_sklearn(model, X_train, y_train)
    print('Correct train set:', X_train.shape, y_train.shape)
    detector = get_defence(
        data_name=data_name,
        model_name=mod,
        idx=idx,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        baard_param=baard_param
    )

    # Override epsilon
    if att == 'boundary' or att == 'tree':
        epsilons = [0]

    classifier = SklearnClassifier(model=model, clip_values=(0.0, 1.0))
    accuracies_no_def = []
    acc_on_advs = []
    fprs = []
    for e in epsilons:
        # Load/Create adversarial examples
        attack = get_attack(att, classifier, e)
        path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv.npy'.format(
            data_name, model_name, att, str(float(e))))
        if os.path.exists(path_adv):
            print('Find:', path_adv)
            adv = np.load(path_adv)
        else:
            adv = attack.generate(X_att)
            np.save(path_adv, adv)
            print('Save to', path_adv)

        acc_naked = model.score(adv, y_att)
        print('Acc without def:', acc_naked)
        accuracies_no_def.append(acc_naked)

        # Preform defence
        labelled_as_adv = detector.detect(adv)
        print('labelled_as_adv:', labelled_as_adv)
        pred_adv = model.predict(X_att)
        acc = acc_on_advx(pred_adv, y_att, labelled_as_adv)
        acc_on_advs.append(acc)
        print('acc_on_advx:', acc)

        labelled_as_adv = detector.detect(X_att)
        fpr = np.mean(labelled_as_adv == y_att)
        fprs.append(fpr)
        print('fpr:', fpr)
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
    path_csv = os.path.join(path_results, 'results', '{}_{}_{}_{}_{}.csv'.format(
        data_name, model_name, att, DEF_NAME, len(detector.stages)))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-m', '--model', type=str, default='svm', choices=['svm', 'tree'])
    parser.add_argument('-i', '--idx', type=int, default=0, choices=list(range(len(SEEDS))))
    parser.add_argument('-a', '--attack', type=str, default='fgsm', choices=ATTACKS)
    parser.add_argument('-e', '--eps', type=float, default=[0.3], nargs='+')
    parser.add_argument('-p', '--param', type=str, required=True, default=os.pah.join('params', 'baard_tune_3s.json'))
    args = parser.parse_args()
    print('args:', args)

    idx = args.idx
    set_seeds(SEEDS[idx])

    data = args.data
    model_name = args.model
    att = args.attack
    epsilons = args.eps
    param = args.param
    print('data:', data)
    print('model_name:', model_name)
    print('attack:', att)
    print('epsilons:', epsilons)
    print('seed:', SEEDS[idx])
    print('param:', param)
    sklearn_attack_against_baard(data, model_name, att, epsilons, idx, param)

# python3 ./run_exp/sklearn_attack_against_baard.py -d banknote -m svm -i 0 -a fgsm -e 0.1 0.2 -p ./params/baard_tune_3s.json
# python3 ./run_exp/sklearn_attack_against_baard.py -d banknote -m tree -i 0 -a tree -p ./params/baard_tune_3s.json
