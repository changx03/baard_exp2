import os

import pandas as pd
import numpy as np

DATASETS = ['banknote', 'breastcancer', 'htru2', 'cifar10', 'cifar10', 'mnist']
MODEL_NAMES = ['basic16', 'basic120', 'basic32', 'resnet', 'vgg', 'basic']
VERSION = 5


def get_csv_path(idx):
    return os.path.join('csv', '{}_{}_{}.csv'.format(DATASETS[idx], MODEL_NAMES[idx], VERSION))


def save_excel(idx):
    path_csv = get_csv_path(idx)
    df = pd.read_csv(path_csv, sep=',')
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(
        columns={'Epsilon': 'Adv_param', 'Score': 'Acc_on_adv'})
    table = df.pivot(
        index=['Attack', 'Adv_param'],
        columns=['Defence'],
        values=['Acc_on_adv', 'FPR'])
    table = table.replace(-100, np.nan)
    with pd.ExcelWriter(os.path.join('tables', '{}_{}_{}.xlsx'.format(DATASETS[idx], MODEL_NAMES[idx], VERSION))) as writer:
        table.to_excel(writer, sheet_name=DATASETS[idx])


if __name__ == '__main__':
    for idx in range(len(DATASETS)):
        save_excel(idx)
