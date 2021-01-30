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

def get_csv_path(dataset):
    return os.path.join('csv', dataset+'_2.csv')

def save_excel(dataset):
    path_csv = get_csv_path(dataset)
    df = pd.read_csv(path_csv, sep=',')
    df = df.drop(columns=['Unnamed: 0'])
    table = df.pivot(index=['Attack', 'Epsilon', 'Without Defence'], columns=['Defence'], values=['Score', 'False Positive Rate'])
    table = table.replace(-100, np.nan)
    with pd.ExcelWriter(os.path.join('tables', dataset+'_2.xlsx')) as writer:
        table.to_excel(writer, sheet_name=dataset)


if __name__ == '__main__':
    for dataset in DATASETS:
        save_excel(dataset)
