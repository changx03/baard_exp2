import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from defences.region_based_classifier import SklearnRegionBasedClassifier
from defences.util import acc_on_adv, merge_and_generate_labels
from experiments.util import load_csv, set_seeds


def baard_preprocess(X, eps=0.02, mean=0., std=1., x_min=0.0, x_max=1.0):
    """Preprocess training data"""
    noise = eps * np.random.normal(mean, scale=std, size=X.shape)
    return np.clip(X + noise, x_min, x_max)


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=data_params['datasets'])
    parser.add_argument('--model', type=str, required=True, choices=['svm', 'tree'])
    parser.add_argument('--adv', type=str, required=True, help="Example: 'apgd_0.3'")
    parser.add_argument('--defence', type=str, required=True, choices=['baard', 'rc'])
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--save', type=int, default=1, choices=[0, 1])
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    args.adv = '{}_{}_{}'.format(args.data, args.model, args.adv)
    print(args)

    set_seeds(args.random_state)
    
    print('Dataset:', args.data)
    print('Model:', args.model)
    print('Pretrained samples:', args.adv + '_adv.npy')
    print('Defence:', args.defence)

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
    else:
        raise NotImplementedError
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print(('Train Acc: {:.4f}, ' + 'Test Acc: {:.4f}').format(acc_train, acc_test))

    # Load adversarial examples
    path_x = os.path.join(args.output_path, '{}_x.npy'.format(args.adv))
    path_y = os.path.join(args.output_path, '{}_y.npy'.format(args.adv))
    path_adv = os.path.join(args.output_path, '{}_adv.npy'.format(args.adv))

    X_benign = np.load(path_x)
    y_true = np.load(path_y)
    adv = np.load(path_adv)
    print('Acc on clean:', model.score(X_benign, y_true))
    print('Acc on adv:', model.score(adv, y_true))

    n = len(X_benign) // 2
    X_val = X_benign[n:]
    y_val = y_true[n:]
    adv_val = adv[n:]

    X_test = X_benign[:n]
    y_test = y_true[:n]
    adv_test = adv[:n]

    # Train defence
    time_start = time.time()
    if args.defence == 'baard':
        sequence = param['sequence']
        stages = []
        if sequence[0]:
            stages.append(ApplicabilityStage(n_classes=param['n_classes'], quantile=param['q1']))
        if sequence[1]:
            stages.append(ReliabilityStage(n_classes=param['n_classes'], k=param['k_re'], quantile=param['q2']))
        if sequence[2]:
            stages.append(DecidabilityStage(n_classes=param['n_classes'], k=param['k_de'], quantile=param['q3']))
        print('BAARD: # of stages:', len(stages))
        detector = BAARDOperator(stages=stages)

        # Run preprocessing
        baard_train_path = os.path.join('results', '{}_{}_baard_train.pt'.format(args.data, args.model))
        obj = torch.load(baard_train_path)
        X_baard = obj['X_train']
        y_train = obj['y_train']
        detector.stages[0].fit(X_baard, y_train)
        detector.stages[1].fit(X_train, y_train)
        if len(detector.stages) == 3:
            detector.stages[2].fit(X_train, y_train)
        detector.search_thresholds(X_val, model.predict(X_val), np.zeros_like(y_val))
    elif args.defence == 'rc':
        detector = SklearnRegionBasedClassifier(
            model=model,
            r=param['r'],
            sample_size=1000,
            n_classes=param['n_classes'],
            x_min=0.0,
            x_max=1.0,
            r0=0.0,
            step_size=0.02,
            stop_value=0.4)
    else:
        raise NotImplementedError
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))

    # Test defence
    time_start = time.time()
    X_test, labels_test = merge_and_generate_labels(adv_test, X_test, flatten=False)
    pred_adv = model.predict(adv_test)
    pred_test = np.concatenate((pred_adv, y_test))
    y_test = np.concatenate((y_test, y_test))

    if args.defence == 'baard':
        res_test = detector.detect(X_test, pred_test)
    elif args.defence == 'rc':
        pred_test = detector.detect(X_test, pred_test)
        res_test = np.zeros_like(pred_test)
    else:
        raise NotImplementedError

    acc = acc_on_adv(pred_test[:n], y_test[:n], res_test[:n])
    if args.defence == 'rc':
        fpr = np.mean(pred_test[n:] != y_test[n:])
    else:
        fpr = np.mean(res_test[n:])
    print('Acc_on_adv:', acc)
    print('FPR:', fpr)
    time_elapsed = time.time() - time_start
    print('Total test time:', str(datetime.timedelta(seconds=time_elapsed)))

    # Save results
    suffix = '_' + args.suffix if args.suffix is not None else ''

    obj = {
        'X_test': X_test,
        'y_test': y_test,
        'labels_test': labels_test,
        'res_test': pred_test if args.defence == 'rc' else res_test,
        'param': param}

    if args.save:
        path_result = os.path.join(args.output_path, '{}_{}{}.pt'.format(args.adv, args.defence, suffix))
        torch.save(obj, path_result)
        print('Saved to:', path_result)
    else:
        print('No file is save!')
    print()


if __name__ == '__main__':
    main()
