import os

import pandas as pd
import numpy as np

DATASETS = [
    'banknote',
    'breastcancer',
    'cifar10_resnet',
    'cifar10_vgg',
    'htru2',
    'mnist',
]
VERSION = 3


def get_csv_path(dataset):
    return os.path.join('csv', '{}_{}.csv'.format(dataset, VERSION))


def save_excel(dataset):
    path_csv = get_csv_path(dataset)
    df = pd.read_csv(path_csv, sep=',')
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(
        columns={'Epsilon': 'Adv_param', 'Score': 'Acc_on_adv'})
    table = df.pivot(
        index=['Attack', 'Adv_param'],
        columns=['Defence'],
        values=['Acc_on_adv', 'FPR'])
    table = table.replace(-100, np.nan)
    with pd.ExcelWriter(os.path.join('tables', '{}_{}.xlsx'.format(dataset, VERSION))) as writer:
        table.to_excel(writer, sheet_name=dataset)


if __name__ == '__main__':
    for dataset in DATASETS:
        save_excel(dataset)
