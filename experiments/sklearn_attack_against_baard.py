"""
Evaluate BAARD against adversarial attacks on numeric datasets with SVM and Tree classifiers. 
"""
import os
import sys

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from art.attacks.evasion import (BasicIterativeMethod, BoundaryAttack,
                                 DecisionTreeAttack, FastGradientMethod)
from art.estimators.classification import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from utils import (acc_on_advx, get_correct_examples_sklearn, load_csv,
                   set_seeds)

from experiments import get_baard, get_output_path

ATTACKS = ['bim', 'fgsm', 'boundary', 'tree']
DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 128
N_SAMPLES = 2000
DEF_NAME = 'baard'


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


def sklearn_attack_against_baard(data_name, model_name, att, epsilons, idx, baard_param, fresh_att=False, fresh_def=True):
    seed = SEEDS[idx]
    set_seeds(seed)

    path_results = get_output_path(idx, data_name, model_name)
    if not os.path.exists(path_results):
        print('[DATA] Output folder does not exist. Create:', path_results)
        path = Path(os.path.join(path_results, 'data'))
        print('[DATA] Create folder:', path)
        path.mkdir(parents=True, exist_ok=True)
        path = Path(os.path.join(path_results, 'results'))
        path.mkdir(parents=True, exist_ok=True)
        print('[DATA] Create folder:', path)

    # Step 1 Load data
    data_path = os.path.join(DATA_PATH, METADATA['data'][data_name]['file_name'])
    print('[DATA] Read file: {}'.format(data_path))
    n_test = METADATA['data'][data_name]['n_test']
    X, y = load_csv(data_path)
    n_classes = len(np.unique(y))

    # Apply scaling
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=seed)

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
    print(('[CLASSIFIER] Train Acc: {:.4f}, ' + 'Test Acc: {:.4f}').format(acc_train, acc_test))

    ############################################################################
    # Step 3: Filter data
    # Get perfect subset
    path_X_train = os.path.join(path_results, 'data', '{}_{}_X_train.npy'.format(data_name, model_name))
    if os.path.exists(path_X_train):
        print('[DATA] Found existing data:', path_X_train)
        X_train = np.load(path_X_train)
        y_train = np.load(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)))
        X_test = np.load(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)))
        y_test = np.load(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)))
    else:
        X_train, y_train = get_correct_examples_sklearn(model, X_train, y_train)
        X_test, y_test = get_correct_examples_sklearn(model, X_test, y_test)
        np.save(path_X_train, X_train)
        np.save(os.path.join(path_results, 'data', '{}_{}_X_test.npy'.format(data_name, model_name)), X_test)
        np.save(os.path.join(path_results, 'data', '{}_{}_y_tain.npy'.format(data_name, model_name)), y_train)
        np.save(os.path.join(path_results, 'data', '{}_{}_y_test.npy'.format(data_name, model_name)), y_test)
        print('[DATA] Save to:', path_X_train)

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
    # Step 4: Load detector
    print('[DEFENCE] Start training BAARD...')
    print('[DEFENCE] Correct train set:', X_train.shape, y_train.shape)
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
                print('[ATTACK] Find:', path_adv)
                adv = np.load(path_adv)
            else:
                adv = attack.generate(X_att)
                np.save(path_adv, adv)
                print('[ATTACK] Save to', path_adv)

            acc_naked = model.score(adv, y_att)
            print('[ATTACK] Acc without def:', acc_naked)

            # Preform defence
            pred_adv = model.predict(adv)
            labelled_as_adv = detector.detect(adv, pred_adv)
            acc = acc_on_advx(pred_adv, y_att, labelled_as_adv)
            print('[DEFENCE] acc_on_adv:', acc)

            # NOTE: clean samples are the same set. Do not repeat.
            if len(fprs) != 0:
                fpr = fprs[0]
            else:
                labelled_benign_as_adv = detector.detect(X_att, y_att)
                fpr = np.mean(labelled_benign_as_adv)
            print('[DEFENCE] fpr:', fpr)
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
    path_csv = os.path.join(path_results, 'results', '{}_{}_{}_{}_{}.csv'.format(data_name, model_name, att, DEF_NAME, len(detector.stages)))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-m', '--model', type=str, default='svm', choices=['svm', 'tree'])
    parser.add_argument('-i', '--idx', type=int, default=0, choices=list(range(len(SEEDS))))
    parser.add_argument('-a', '--att', type=str, default='fgsm', choices=ATTACKS)
    parser.add_argument('-e', '--eps', type=float, default=[0.3], nargs='+')
    parser.add_argument('-p', '--param', type=str, required=True, default=os.pah.join('params', 'baard_num_3.json'))
    args = parser.parse_args()
    print('args:', args)

    idx = args.idx
    data = args.data
    model_name = args.model
    att = args.att
    epsilons = args.eps
    param = args.param
    print('data:', data)
    print('model_name:', model_name)
    print('attack:', att)
    print('epsilons:', epsilons)
    print('seed:', SEEDS[idx])
    print('param:', param)
    sklearn_attack_against_baard(data, model_name, att, epsilons, idx, param)

    # Testing
    # sklearn_attack_against_baard('banknote', 'svm', 'fgsm', [0.2], 0, './params/baard_num_3.json', fresh_att=False, fresh_def=True)
