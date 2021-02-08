import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
from utils.acc_on_advx import acc_on_advx
from models.torch_util import predict_numpy


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

    score = acc_on_advx(y_pred, y_true, detected_as_adv)

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

    score = acc_on_advx(y_pred, y_true, detected_as_adv)

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
