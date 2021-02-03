import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from experiments.util import get_dataframe

VERSION = 5
OUTPUT_PATH = 'csv'
DATASETS = ['banknote', 'breastcancer', 'htru2', 'cifar10', 'cifar10', 'mnist']
MODEL_NAMES = ['basic16', 'basic120', 'basic32', 'resnet', 'vgg', 'basic']
ATTACKS_NUM = [
    'apgd_0.05', 'apgd_0.2', 'apgd_0.4', 'apgd_1.0',
    'apgd2_0.4', 'apgd2_1.0', 'apgd2_2.0', 'apgd2_3.0',
    'boundary_0.3',
    'cw2_0.0', 'cw2_5.0', 'cw2_10.0',
    'deepfool_1e-06',
    'fgsm_0.05', 'fgsm_0.2', 'fgsm_0.4', 'fgsm_1.0']
ATTACKS_CIFAR10 = [
    'apgd_0.031', 'apgd_0.3', 'apgd_0.6', 'apgd_1.0', 'apgd_1.5',
    'apgd2_1.5', 'apgd2_2.0', 'apgd2_3.0', 'apgd2_5.0',
    'cw2_0.0', 'cw2_5.0', 'cw2_10.0',
    'deepfool_1e-06',
    'fgsm_0.031', 'fgsm_0.3', 'fgsm_0.6', 'fgsm_1.0', 'fgsm_1.5',
    'line_0.0', 'line_0.5', 'line_1.0',
    'watermark_0.3', 'watermark_0.6']
ATTACKS_MNIST = [
    'apgd_0.063', 'apgd_0.3', 'apgd_0.6', 'apgd_1.0', 'apgd_1.5',
    'apgd2_1.5', 'apgd2_2.0', 'apgd2_3.0', 'apgd2_5.0',
    'boundary_0.3',
    'cw2_0.0', 'cw2_5.0', 'cw2_10.0',
    'deepfool_1e-06',
    'fgsm_0.063', 'fgsm_0.3', 'fgsm_0.6', 'fgsm_1.0', 'fgsm_1.5',
    'line_0.0', 'line_0.5', 'line_1.0',
    'watermark_0.3', 'watermark_0.6']
DEFENCES_NUM = ['baard_2stage', 'baard_3stage', 'lid', 'rc']
DEFENCES_IMG = ['fs', 'magnet']
COLUMNS = ['Attack', 'Adv_param', 'Defence', 'FPR', 'Acc_on_adv']


def get_model(idx, device):
    model_file = DATASETS[idx] + '_400.pt'
    if idx == 0:
        model = NumericModel(n_features=4, n_hidden=4 * 4, n_classes=2, use_prob=True)
    elif idx == 1:
        model = NumericModel(n_features=30, n_hidden=30 * 4, n_classes=2, use_prob=True)
    elif idx == 2:
        model = NumericModel(n_features=8, n_hidden=8 * 4, n_classes=2, use_prob=True)
    elif idx == 3:
        model = Resnet(use_prob=True)
        model_file = '{}_{}_200.pt'.format(DATASETS[idx], MODEL_NAMES[idx])
    elif idx == 4:
        model = Vgg(use_prob=True)
        model_file = '{}_{}_200.pt'.format(DATASETS[idx], MODEL_NAMES[idx])
    elif idx == 5:
        model = BaseModel(use_prob=True)
        model_file = '{}_200.pt'.format(DATASETS[idx])
    else:
        raise NotImplementedError
    path_model = os.path.join('results', model_file)
    model.load_state_dict(torch.load(path_model))
    model = model.to(device)
    return model


def read_results(idx, data, device):
    model = get_model(idx, device)
    df = pd.DataFrame(columns=COLUMNS)
    if data not in ['mnist', 'cifar10']:
        attack_names = ATTACKS_NUM
        defence_names = DEFENCES_NUM
    else:
        defence_names = DEFENCES_NUM + DEFENCES_IMG
        if data == 'mnist':
            attack_names = ATTACKS_MNIST
        else:
            attack_names = ATTACKS_CIFAR10
    model_name = MODEL_NAMES[idx]
    for attack in attack_names:
        for defence in defence_names:
            try:
                df = get_dataframe(df, model, data, model_name, attack, defence, device)
            except FileNotFoundError as err:
                print(err)
                continue

    # These attacks have no hyperparameter
    df.loc[(df['Attack'] == 'boundary'), 'Adv_param'] = np.nan

    output_file = os.path.join(OUTPUT_PATH, '{}_{}_{}.csv'.format(data, model_name, VERSION))
    df.to_csv(output_file)
    print('Save to:', output_file)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    for i, d in enumerate(DATASETS):
        read_results(i, d, device)
