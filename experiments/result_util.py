import os

import numpy as np
import torch

from defences.util import score
from models.torch_util import predict_numpy

def get_dataframe(df, model, data, nmodel, nattack, ndefence, device):
    path_result = os.path.join('..', 'results', '{}_{}_{}_{}.pt'.format(data, nmodel, nattack, ndefence))
    result = torch.load(path_result)
    X_test = result['X_test']
    y_test = result['y_test']
    labels_test = result['labels_test']
    res_test = result['res_test']

    idx_tp = np.where(labels_test == 1)[0]
    pred = predict_numpy(model, X_test[idx_tp], device=device)
    acc_adv = np.sum(pred == y_test[idx_tp]) / len(idx_tp)

    # Compute accuracy on adv
    if ndefence in ['baard_2stage', 'baard_3stage', 'fs', 'lid']:
        score_ = np.sum(res_test[idx_tp]) / len(idx_tp)
    elif ndefence == 'rc':
        score_ = np.sum(res_test[idx_tp] == y_test[idx_tp]) / len(idx_tp)
    elif ndefence == 'magnet':
        X_reformed = result['X_reformed']
        pred_reformed = predict_numpy(model, X_reformed[idx_tp], device=device)
        score_ = score(res_test[idx_tp], y_test[idx_tp], pred_reformed, np.ones(len(idx_tp)))
    else:
        raise ValueError

    # Compute FPR
    idx_fp = np.where(labels_test == 0)[0]
    fpr = np.sum(res_test[idx_fp]) / len(idx_fp)
    if ndefence == 'rc':
        fpr = np.sum(res_test[idx_fp] != y_test[idx_fp]) / len(idx_fp)

    att_str = nattack.split('_')
    df = df.append({
        'Attack': att_str[0], 
        'Epsilon': att_str[1], 
        'Without Defence': acc_adv * 100,
        'Defence': ndefence,
        'False Positive Rate': fpr * 100,
        'Score': score_ * 100,
    }, ignore_index=True)
    return df