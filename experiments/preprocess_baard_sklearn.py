import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier

# Adding the parent directory.
sys.path.append(os.getcwd())
from experiments.util import load_csv, set_seeds


def baard_preprocess(X, eps=0.02, mean=0., std=1., x_min=0.0, x_max=1.0):
    """Preprocess training data"""
    noise = eps * np.random.normal(mean, scale=std, size=X.shape)
    return np.clip(X + noise, x_min, x_max)


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'tree'])
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=str, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    print('data:', args.data)
    print('model:', args.model)

    # Prepare data
    data_path = os.path.join(args.data_path, data_params['data'][args.data]['file_name'])
    print('Read file: {}'.format(data_path))
    X, y = load_csv(data_path)

    # Apply scaling
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    n_test = data_params['data'][args.data]['n_test']
    random_state = args.random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state)

    # Train model
    if args.model == 'svm':
        model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=random_state)
    elif args.model == 'tree':
        model = ExtraTreeClassifier(random_state=random_state)
    else:
        raise NotImplementedError
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print(('Train Acc: {:.4f}, ' + 'Test Acc: {:.4f}').format(acc_train, acc_test))

    # Get perfect subset
    pred_test = model.predict(X_test)
    idx_correct = np.where(pred_test == y_test)[0]
    X_test = X_test[idx_correct]
    y_test = y_test[idx_correct]

    X_baard = baard_preprocess(X_train)
    obj = {
        'X_train': X_baard,
        'y_train': y_train
    }
    path_ouput = os.path.join(args.output_path, '{}_{}_baard_train.pt'.format(args.data, args.model, args.model))
    torch.save(obj, path_ouput)
    print('Save to:', path_ouput)
    print()


if __name__ == '__main__':
    main()
