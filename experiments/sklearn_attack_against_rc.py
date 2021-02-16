"""
Evaluate Region-based Classifier against adversarial attacks on numeric datasets with SVM and Tree classifiers. 
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
from art.attacks.evasion import (BasicIterativeMethod, BoundaryAttack,
                                 DecisionTreeAttack, FastGradientMethod)
from art.estimators.classification import SklearnClassifier
from defences.region_based_classifier import SklearnRegionBasedClassifier
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
    return os.path.join('results', 'result_' + str(i), '{}_{}'.format(data, model_name))


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
            targeted=False,
            verbose=False)
    elif att_name == 'fgsm':
        attack = FastGradientMethod(
            estimator=classifier,
            eps=eps,
            batch_size=BATCH_SIZE)
    elif att_name == 'tree':
        attack = DecisionTreeAttack(
            classifier=classifier,
            verbose=False)
    else:
        raise NotImplementedError
    return attack


def sklearn_attack_against_rc(data_name, model_name, att, epsilons, idx, fresh_att=False, fresh_def=True):
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
    data_path = os.path.join(DATA_PATH, METADATA['data'][data_name]['file_name'])
    print('Read file: {}'.format(data_path))
    X, y = load_csv(data_path)
    n_classes = len(np.unique(y))

    # Apply scaling
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    path_results = get_output_path(idx, data_name, model_name)
    path_X_train = os.path.join(path_results, 'data', '{}_{}_X_train.npy'.format(data_name, model_name))
    if os.path.exists(path_X_train):
        print('Found existing data:', path_X_train)
        X_train = np.load(path_X_train)
        X_test = np.load(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)))
        y_train = np.load(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)))
        y_test = np.load(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)))
    else:
        print('Cannot found:', path_X_train)
        n_test = METADATA['data'][data_name]['n_test']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=seed)
        np.save(path_X_train, X_train)
        np.save(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)), X_test)
        np.save(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)), y_train)
        np.save(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)), y_test)
        print('Save to:', path_X_train)

    ############################################################################
    # Step 2: Train model
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

    ############################################################################
    # Step 3: Filter data
    # Get perfect subset
    pred_test = model.predict(X_test)
    idx_correct = np.where(pred_test == y_test)[0]
    X_test = X_test[idx_correct]
    y_test = y_test[idx_correct]

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

    ############################################################################
    # Step 4: Load detector
    detector = SklearnRegionBasedClassifier(
        model=model,
        r=0.2,
        sample_size=1000,
        n_classes=n_classes,
        x_min=0.0,
        x_max=1.0,
        r0=0.0,
        step_size=0.02,
        stop_value=0.4)

    path_best_r = os.path.join(path_results, 'results', '{}_{}.json'.format(data_name, model_name))
    if os.path.exists(path_best_r) and not fresh_def:
        with open(path_best_r) as j:
            obj_r = json.load(j)
        r_best = obj_r['r']
        print('Find:', path_best_r)
    else:
        print('Cannot found:', path_best_r)
        print('Start searching r_best...')
        time_start = time.time()
        r_best = detector.search_thresholds(X_val, model.predict(X_val), np.zeros_like(y_val), verbose=0)
        time_elapsed = time.time() - time_start
        print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
        with open(path_best_r, 'w') as f:
            json.dump({'r': r_best}, f)
            print('Save to:', path_best_r)
    print('r_best:', r_best)
    detector.r = r_best

    ############################################################################
    # Step 5: Generate attack and preform defence
    # Override epsilon
    if att == 'boundary' or att == 'tree':
        epsilons = [0]

    classifier = SklearnClassifier(model=model, clip_values=(0.0, 1.0))
    accuracies_no_def = []
    acc_on_advs = []
    fprs = []
    for e in epsilons:
        try:
            # Load/Create adversarial examples
            attack = get_attack(att, classifier, e)
            path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv.npy'.format(data_name, model_name, att, str(float(e))))
            if os.path.exists(path_adv) and not fresh_att:
                print('Find:', path_adv)
                adv = np.load(path_adv)
            else:
                adv = attack.generate(X_att)
                np.save(path_adv, adv)
                print('Save to', path_adv)

            acc_naked = model.score(adv, y_att)
            print('Acc without def:', acc_naked)

            # Preform defence
            pred_adv = detector.detect(adv)
            print('pred_adv:', pred_adv.shape)
            res_test = np.zeros_like(pred_adv)
            acc = acc_on_advx(pred_adv, y_att, res_test)
            print('acc_on_advx:', acc)

            pred_benign = detector.detect(X_att)
            fpr = np.mean(pred_benign != y_att)
            print('fpr:', fpr)
        except Exception as e:
            print(e)
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
    path_csv = os.path.join(path_results, 'results', '{}_{}_{}_{}.csv'.format(data_name, model_name, att, DEF_NAME))
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
    sklearn_attack_against_rc(data, model_name, att, epsilons, idx)

    # Testing
    # sklearn_attack_against_rc('banknote', 'svm', 'fgsm', ['0.2'], 10)
