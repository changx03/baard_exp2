import os
import sys

sys.path.append(os.getcwd())

import json

import numpy as np
import torch
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from defences.preprocess_baard import preprocess_baard_numpy, preprocess_baard_img
from experiments.get_output_path import get_output_path


def check_json_param(param1, param2):
    return (param1['k_re'] != param2['k_re'] or \
        param1['k_de'] != param2['k_de'] or \
        param1['fpr1'] != param2['fpr1'] or \
        param1['fpr2'] != param2['fpr2'] or \
        param1['fpr3'] != param2['fpr3'])


def get_baard(data_name, model_name, idx, X_train, y_train, X_val, y_val, baard_param, restart):
    path_results = get_output_path(idx, data_name, model_name)

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

    # Load each stage
    with open(baard_param) as j:
        baard_param = json.load(j)
        sequence = baard_param['sequence']
        path_param_backup = os.path.join(path_results, 'results', '{}_{}_baard_{}.json'.format(data_name, model_name, np.sum(sequence)))
        if os.path.exists(path_param_backup):
            k = open(path_param_backup)
            param_saved = json.load(k)
            if check_json_param(baard_param, param_saved):
                print('[DEFENCE] Do not match existing BAARD JSON params. Delete and save new JSON file.')
                os.remove(path_param_backup)
                json.dump(baard_param, open(path_param_backup, 'w'))
                restart = True
                print('[DEFENCE] Save to:', path_param_backup)
            else:
                print('[DEFENCE] Found existing BAARD params:', path_param_backup)
        else:
            json.dump(baard_param, open(path_param_backup, 'w'))
            print('[DEFENCE] Save to:', path_param_backup)
    print('[DEFENCE] Param:', baard_param)

    stages = []
    if sequence[0]:
        s1 = ApplicabilityStage(n_classes=n_classes, fpr=baard_param['fpr1'], verbose=False)
        s1.fit(X_train_s1, y_train)
        stages.append(s1)
    if sequence[1]:
        s2 = ReliabilityStage(n_classes=n_classes, k=baard_param['k_re'], fpr=baard_param['fpr2'], verbose=False)
        s2.fit(X_train, y_train)
        stages.append(s2)
    if sequence[2]:
        s3 = DecidabilityStage(n_classes=n_classes, k=baard_param['k_de'], fpr=baard_param['fpr3'], verbose=False)
        s3.fit(X_train, y_train)
        stages.append(s3)
    print('[DEFENCE] BAARD stages:', len(stages))
    detector = BAARDOperator(stages=stages)

    # Set thresholds
    file_baard_threshold = os.path.join(path_results, 'data', '{}_{}_baard_threshold_{}.pt'.format(data_name, model_name, len(stages)))
    if os.path.exists(file_baard_threshold) and not restart:
        print('[DEFENCE] Found existing BAARD thresholds:', file_baard_threshold)
        detector.load(file_baard_threshold)
    else:
        # Search thresholds
        detector.search_thresholds(X_val, y_val, np.zeros_like(y_val))
        detector.save(file_baard_threshold)
        print('[DEFENCE] Save to:', file_baard_threshold)
    return detector
