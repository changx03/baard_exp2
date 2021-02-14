import os

import pandas as pd
import numpy as np

DATASETS = ['banknote', 'breastcancer', 'htru2']
MODEL_NAMES = ['svm', 'tree']
VERSION = 2


def save_excel(data, model_name):
    path_csv = os.path.join('csv', '{}_{}_{}.csv'.format(data, model_name, VERSION))
    df = pd.read_csv(path_csv, sep=',')
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(
        columns={'Epsilon': 'Adv_param', 'Score': 'Acc_on_adv'})
    table = df.pivot(
        index=['Attack', 'Adv_param'],
        columns=['Defence'],
        values=['Acc_on_adv', 'FPR'])
    table = table.replace(-100, np.nan)
    excel_file_name = '{}_{}_{}.xlsx'.format(data, model_name, VERSION)
    excel_file_path = os.path.join('tables', excel_file_name)
    with pd.ExcelWriter(excel_file_path) as writer:
        table.to_excel(writer, sheet_name=data)
        print('Save to:', excel_file_path)


if __name__ == '__main__':
    for data in DATASETS:
        for model_name in MODEL_NAMES:
            save_excel(data, model_name)
