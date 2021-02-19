"""
Evaluate BAARD against adversarial attacks on numeric datasets with SVM and Tree
classifiers. 
"""
import datetime
import os
import sys
import time

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import argparse
import json

import numpy as np
import pandas as pd
from art.attacks.evasion import (BasicIterativeMethod, BoundaryAttack,
                                 DecisionTreeAttack, FastGradientMethod)
from art.estimators.classification import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from utils import (acc_on_advx, get_correct_examples_sklearn, load_csv, mkdir,
                   set_seeds)

from experiments import get_baard, get_output_path

ATTACKS = ['bim', 'fgsm', 'boundary', 'tree']
DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 192
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
            targeted=False)
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


def sklearn_attack_against_baard(data_name, model_name, att, epsilons, idx, baard_param=None):
    print('Runing sklearn_attack_against_baard.py')
    seed = SEEDS[idx]
    set_seeds(seed)

    path_results = get_output_path(idx, data_name, model_name)
    mkdir(os.path.join(path_results, 'data'))
    mkdir(os.path.join(path_results, 'results'))

    # Step 1 Load data
    data_path = os.path.join(DATA_PATH, METADATA['data'][data_name]['file_name'])
    print('[DATA] Read file: {}'.format(data_path))
    n_test = METADATA['data'][data_name]['n_test']
    X, y = load_csv(data_path)
    # n_classes = len(np.unique(y))

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
    X_val = X_test[n: n * 2]
    y_val = y_test[n: n * 2]

    ############################################################################
    # Step 4: Load detector
    print('[DEFENCE] Start training BAARD...')
    if baard_param is None:
        baard_param = os.path.join('params', 'baard_{}.json'.format(data_name))
    if not os.path.exists(baard_param):
        raise FileNotFoundError("Cannot find BAARD's config file: {}".format(baard_param))

    detector = get_baard(
        data_name=data_name,
        model_name=model_name,
        idx=idx,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        baard_param=baard_param)

    ############################################################################
    # Step 5: Generate attack and preform defence
    # Override epsilon
    if att == 'boundary' or att == 'tree':
        epsilons = [0]

    # Adversarial examples come from the same benign set
    labelled_fp_1, labelled_fp_2, labelled_fp_3 = detector.detect(X_att, y_att, per_stage=True)
    fpr_1 = np.mean(labelled_fp_1)
    fpr_2 = np.mean(labelled_fp_2)
    fpr_3 = np.mean(labelled_fp_3)

    classifier = SklearnClassifier(model=model, clip_values=(0.0, 1.0))
    accuracies_no_def = []
    acc_on_advs_1 = []
    acc_on_advs_2 = []
    acc_on_advs_3 = []
    for e in epsilons:
        try:
            # Load/Create adversarial examples
            attack = get_attack(att, classifier, e)
            path_adv = os.path.join(path_results, 'data', '{}_{}_{}_{}_adv.npy'.format(data_name, model_name, att, str(float(e))))
            if os.path.exists(path_adv):
                print('[ATTACK] Find:', path_adv)
                adv = np.load(path_adv)
            else:
                adv = attack.generate(X_att)
                np.save(path_adv, adv)
                print('[ATTACK] Save to', path_adv)

            pred_adv = model.predict(adv)
            acc_naked = model.score(adv, y_att)
            print('[ATTACK] Acc without def:', acc_naked)

            # Preform defence
            print('[DEFENCE] Start running BAARD...')
            start = time.time()
            labelled_as_adv_1, labelled_as_adv_2, labelled_as_adv_3 = detector.detect(adv, pred_adv, per_stage=True)
            time_elapsed = time.time() - start
            print('[DEFENCE] Time spend:', str(datetime.timedelta(seconds=time_elapsed)))

            acc_1 = acc_on_advx(pred_adv, y_att, labelled_as_adv_1)
            acc_2 = acc_on_advx(pred_adv, y_att, labelled_as_adv_2)
            acc_3 = acc_on_advx(pred_adv, y_att, labelled_as_adv_3)

            print('[DEFENCE] acc_on_adv (Stage 3):', acc_3)
            print('[DEFENCE] fpr (Stage 3):', fpr_3)
        except Exception as e:
            print(e)
            acc_naked = np.nan
            acc_1 = acc_2 = acc_3 = np.nan
        finally:
            accuracies_no_def.append(acc_naked)
            acc_on_advs_1.append(acc_1)
            acc_on_advs_2.append(acc_2)
            acc_on_advs_3.append(acc_3)

    data = {
        'data': np.repeat(data_name, len(epsilons)),
        'model': np.repeat(model_name, len(epsilons)),
        'attack': np.repeat(att, len(epsilons)),
        'adv_param': np.array(epsilons),
        'acc_no_def': np.array(accuracies_no_def),
        'acc_on_adv_1': np.array(acc_on_advs_1),
        'fpr_1': np.repeat(fpr_1, len(epsilons)),
        'acc_on_adv_2': np.array(acc_on_advs_2),
        'fpr_2': np.repeat(fpr_2, len(epsilons)),
        'acc_on_adv_3': np.array(acc_on_advs_3),
        'fpr_3': np.repeat(fpr_3, len(epsilons))}
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
    parser.add_argument('-a', '--att', type=str, default='fgsm', choices=ATTACKS)
    parser.add_argument('-e', '--eps', type=float, default=[0.3], nargs='+')
    parser.add_argument('-p', '--param', type=str)
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
    # sklearn_attack_against_baard('banknote', 'svm', 'fgsm', [0.3, 1.0], 0)

# python ./experiments/sklearn_attack_against_baard.py -d banknote -m svm -i 0 -a fgsm -e 0.3 1.0
