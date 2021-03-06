import os
import sys
import random

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from defences.util import acc_on_adv
from models.torch_util import predict_numpy


def load_csv(file_path):
    """Load a pre-processed CSV file."""
    df = pd.read_csv(file_path, sep=',')
    y = df['Class'].to_numpy().astype(np.long)
    X = df.drop(['Class'], axis=1).to_numpy().astype(np.float32)
    return X, y


def get_dataframe(df, model, data, nmodel, nattack, ndefence, device, result_path):
    path_result = os.path.join(result_path, '{}_{}_{}_{}.pt'.format(data, nmodel, nattack, ndefence))
    result = torch.load(path_result)
    X_test = result['X_test']
    y_test = result['y_test']
    labels_test = result['labels_test']
    res_test = result['res_test']

    idx_tp = np.where(labels_test == 1)[0]

    detected_as_adv = res_test[idx_tp]
    y_true = y_test[idx_tp]
    if ndefence == 'magnet':
        X_reformed = result['X_reformed']
        y_pred = predict_numpy(model, X_reformed[idx_tp], device=device)
    elif ndefence == 'rc':
        y_pred = result['res_test'][idx_tp]
        detected_as_adv = np.zeros_like(y_pred)
    else:
        y_pred = predict_numpy(model, X_test[idx_tp], device=device)

    score = acc_on_adv(y_pred, y_true, detected_as_adv)

    # Compute FPR
    idx_fp = np.where(labels_test == 0)[0]
    if ndefence == 'rc':
        fpr = np.mean(res_test[idx_fp] != y_test[idx_fp])
    else:
        fpr = np.mean(res_test[idx_fp])

    att_str = nattack.split('_')
    df = df.append({
        'Attack': att_str[0],
        'Adv_param': att_str[1],
        'Defence': ndefence,
        'FPR': fpr * 100,
        'Acc_on_adv': score * 100,
    }, ignore_index=True)
    return df


def get_dataframe_sklearn(df, model, data, nmodel, nattack, ndefence, result_path):
    path_result = os.path.join(result_path, '{}_{}_{}_{}.pt'.format(data, nmodel, nattack, ndefence))
    result = torch.load(path_result)
    X_test = result['X_test']
    y_test = result['y_test']
    labels_test = result['labels_test']
    res_test = result['res_test']

    idx_tp = np.where(labels_test == 1)[0]

    detected_as_adv = res_test[idx_tp]
    y_true = y_test[idx_tp]
    if ndefence == 'rc':
        y_pred = result['res_test'][idx_tp]
        detected_as_adv = np.zeros_like(y_pred)
    else:
        y_pred = model.predict(X_test[idx_tp])

    score = acc_on_adv(y_pred, y_true, detected_as_adv)

    # Compute FPR
    idx_fp = np.where(labels_test == 0)[0]
    if ndefence == 'rc':
        fpr = np.mean(res_test[idx_fp] != y_test[idx_fp])
    else:
        fpr = np.mean(res_test[idx_fp])

    att_str = nattack.split('_')
    df = df.append({
        'Attack': att_str[0],
        'Adv_param': att_str[1],
        'Defence': ndefence,
        'FPR': fpr * 100,
        'Acc_on_adv': score * 100,
    }, ignore_index=True)
    return df


def set_seeds(random_state):
    random_state = int(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
