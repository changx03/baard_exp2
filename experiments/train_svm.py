import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
# # Adding the parent directory.

DATA_NAMES = ['banknote', 'htru2', 'segment', 'texture']
DATA = {
    'banknote': {'file_name': 'banknote_preprocessed.csv', 'n_features': 4, 'n_test': 400, 'n_classes': 2},
    'htru2': {'file_name': 'htru2_preprocessed.csv', 'n_features': 8, 'n_test': 4000, 'n_classes': 2},
    'segment': {'file_name': 'segment_preprocessed.csv', 'n_features': 18, 'n_test': 400, 'n_classes': 7},
    'texture': {'file_name': 'texture_preprocessed.csv', 'n_features': 40, 'n_test': 600, 'n_classes': 11},
}
RANDOM_STATE = int(2**12)


def load_csv(file_path):
    """Load a pre-processed CSV file."""
    df = pd.read_csv(file_path, sep=',')
    y = df['Class'].to_numpy().astype(np.long)
    X = df.drop(['Class'], axis=1).to_numpy().astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=DATA_NAMES)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--kernel', type=str, default='linear')
    parser.add_argument('--regularisation', type=int, default=1.0)
    parser.add_argument('--gamma', type=str, default="scale")
    args = parser.parse_args()

    # Prepare data
    data_path = os.path.join(args.data_path, DATA[args.data]['file_name'])
    print('Read file: {}'.format(data_path))
    X, y = load_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DATA[args.data]['n_test'], random_state=RANDOM_STATE)

    # Apply scaling
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    since = time.time()
    model = SVC(kernel="linear",C=1.0,gamma="scale")
    model.fit(X_train, y_train)

    print(('Train Acc: {:.4f}%, '+'Test Acc: {:.4f}%').format(model.score(X_train, y_train),model.score(X_test, y_test)))

    time_elapsed = time.time() - since
    print('Total run time: {:.0f}m {:.1f}s'.format(time_elapsed // 60,time_elapsed % 60))

if __name__ == '__main__':
    main()