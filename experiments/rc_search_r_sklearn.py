import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.region_based_classifier import SklearnRegionBasedClassifier

from experiments.util import load_csv, set_seeds


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'tree'])
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--adv', type=str, default='fgsm_0.2')
    parser.add_argument('--param', type=str, default=None)
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()

    set_seeds(args.random_state)
    
    if args.param == None:
        args.param = os.path.join('params', 'rc_param_{}.json'.format(args.data))

    adv_root = '{}_{}_{}'.format(args.data, args.model, args.adv)

    print('Dataset:', args.data)
    print('Model:', args.model)
    print('Pretrained samples:', adv_root + '_adv.npy')

    with open(args.param) as param_json:
        param = json.load(param_json)
    param['n_classes'] = data_params['data'][args.data]['n_classes']
    print('Param:', param)

    # Prepare data
    data_path = os.path.join(args.data_path, data_params['data'][args.data]['file_name'])
    print('Read file: {}'.format(data_path))
    X, y = load_csv(data_path)

    # Apply scaling
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    n_test = data_params['data'][args.data]['n_test']
    random_state = args.random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state)

    # Train model
    if args.model == 'svm':
        model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=random_state)
    elif args.model == 'tree':
        model = ExtraTreeClassifier()
    else:
        raise NotImplementedError
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print(('Train Acc: {:.4f}, ' + 'Test Acc: {:.4f}').format(acc_train, acc_test))

    # Load adversarial examples
    path_x = os.path.join(args.output_path, '{}_x.npy'.format(adv_root))
    path_y = os.path.join(args.output_path, '{}_y.npy'.format(adv_root))
    path_adv = os.path.join(args.output_path, '{}_adv.npy'.format(adv_root))

    X_benign = np.load(path_x)
    y_true = np.load(path_y)
    adv = np.load(path_adv)
    print(X_benign.shape, y_true.shape, adv.shape)
    print('Acc on clean:', model.score(X_benign, y_true))
    print('Acc on adv:', model.score(adv, y_true))

    n = len(X_benign) // 2
    X_val = X_benign[n:]
    y_val = y_true[n:]

    # Train defence
    time_start = time.time()
    detector = SklearnRegionBasedClassifier(
        model=model,
        r=0.2,
        sample_size=1000,
        n_classes=param['n_classes'],
        x_min=0.0,
        x_max=1.0,
        r0=0.0,
        step_size=0.02,
        stop_value=0.4)
    r_best = detector.search_thresholds(X_val, model.predict(X_val), np.zeros_like(y_val), verbose=0)
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))

    param = {
        "r": r_best,
        "sample_size": 1000,
        "batch_size": 512,
        "r0": 0,
        "step_size": 0.02,
        "stop_value": 0.40
    }
    path_json = os.path.join('params', 'rc_param_{}_{}.json'.format(args.data, args.model))
    with open(path_json, 'w') as f:
        json.dump(param, f)
    print('Save to:', path_json)
    print()


if __name__ == '__main__':
    main()
