import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from experiments.util import load_csv, get_dataframe_sklearn

VERSION = 2
OUTPUT_PATH = 'csv'
DATASETS = ['banknote', 'breastcancer', 'htru2']
MODELS = ['svm']
ATTACKS_NUM = [
    "bim_0.05", "bim_0.2", "bim_0.4", "bim_1.0", "boundary_0.3", "fgsm_0.05", "fgsm_0.2", "fgsm_0.4", "fgsm_1.0"
]
DEFENCES_NUM = ['baard_2stage', 'baard_3stage', 'rc']
COLUMNS = ['Attack', 'Adv_param', 'Defence', 'FPR', 'Acc_on_adv']
DATA_PATH = 'data'
RANDOM_STATE = 1234
RESULT_PATH = 'results'


def read_results(data, model_name):
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    # Prepare data
    data_path = os.path.join(DATA_PATH, data_params['data'][data]['file_name'])
    print('Read file: {}'.format(data_path))
    X, y = load_csv(data_path)

    # Apply scaling
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    n_test = data_params['data'][data]['n_test']
    random_state = RANDOM_STATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state)

    model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print(('Train Acc: {:.4f}, ' + 'Test Acc: {:.4f}').format(acc_train, acc_test))

    df = pd.DataFrame(columns=COLUMNS)
    for attack in ATTACKS_NUM:
        for defence in DEFENCES_NUM:
            try:
                df = get_dataframe_sklearn(df, model, data, model_name, attack, defence, RESULT_PATH)
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

    for data in DATASETS:
        for model in MODELS:
            read_results(data, model)
