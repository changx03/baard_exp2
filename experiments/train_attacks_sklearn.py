import os
import sys
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
print("sys.path ", sys.path)

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from art.attacks.evasion import (BasicIterativeMethod, BoundaryAttack,
                                 FastGradientMethod, DecisionTreeAttack)
from art.estimators.classification import SklearnClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# # Adding the parent directory.
sys.path.append(os.getcwd())
from experiments.util import set_seeds

ATTACKS = ['bim', 'fgsm', 'boundary', 'tree']

def load_csv(file_path):
    """Load a pre-processed CSV file."""
    df = pd.read_csv(file_path, sep=',')
    y = df['Class'].to_numpy().astype(np.long)
    X = df.drop(['Class'], axis=1).to_numpy().astype(np.float32)
    return X, y


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=data_params['datasets'])
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'tree'])
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--attack', type=str, required=True, choices=ATTACKS)
    parser.add_argument('--eps', type=float, default=0.3)
    # NOTE: In CW_L2 attack, eps is the upper bound of c.
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    if not os.path.exists(args.output_path):
        print('Output folder does not exist. Create:', args.output_path)
        os.mkdir(args.output_path)
        
    print('Dataset:', args.data)
    print('Running attack:', args.attack)

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

    classifier = SklearnClassifier(model=model, clip_values=(0.0, 1.0))

    if args.attack == 'bim':
        eps_step = args.eps / 10.0
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=args.eps,
            eps_step=eps_step,
            max_iter=100,
            targeted=False,
            batch_size=args.batch_size)
    elif args.attack == 'boundary':
        attack = BoundaryAttack(
            estimator=classifier,
            max_iter=1000,
            sample_size=20,
            targeted=False)
    elif args.attack == 'fgsm':
        attack = FastGradientMethod(
            estimator=classifier,
            eps=args.eps,
            batch_size=args.batch_size)
    elif args.attack == 'tree':
        attack = DecisionTreeAttack(
            classifier=classifier)
    else:
        raise NotImplementedError

    # How many examples do we have?
    if len(X_test) > args.n_samples:
        n = args.n_samples
    else:
        n = len(X_test)

    X_benign = X_test[:n]
    y_true = y_test[:n]
    adv = attack.generate(X_benign)
    acc = model.score(adv, y_true)
    print('Acc on adv:', acc)

    output_file = '{}_{}_{}_{}'.format(args.data, args.model, args.attack, str(args.eps))
    path_x = os.path.join(args.output_path, '{}_x.npy'.format(output_file))
    path_y = os.path.join(args.output_path, '{}_y.npy'.format(output_file))
    path_adv = os.path.join(args.output_path, '{}_adv.npy'.format(output_file))
    np.save(path_x, X_benign)
    np.save(path_y, y_true)
    np.save(path_adv, adv)

    print('Saved to:', path_adv)
    print()


if __name__ == '__main__':
    main()
