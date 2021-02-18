import argparse
import os
import sys
import time
import datetime

sys.path.append(os.getcwd())

from experiments.sklearn_attack_against_baard import sklearn_attack_against_baard
from experiments.sklearn_attack_against_magnet import sklearn_attack_against_magnet
from experiments.sklearn_attack_against_rc import sklearn_attack_against_rc


def run_small_batch(data, model, att, eps, i):
    sklearn_attack_against_baard(data, model, att, eps, i)
    sklearn_attack_against_magnet(data, model, att, eps, i)
    sklearn_attack_against_rc(data, model, att, eps, i)


def run_svm(i, data):
    model = 'svm'

    eps = [0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'bim']
    for att in attacks:
        run_small_batch(data, model, att, eps, i)

    eps = [0.]
    att = 'boundary'
    run_small_batch(data, model, att, eps, i)


def run_tree(i, data):
    model = 'tree'

    eps = [0., 5., 10.]
    att = 'tree'
    run_small_batch(data, model, att, eps, i)

    eps = [0.]
    att = 'boundary'
    run_small_batch(data, model, att, eps, i)


def run(i):
    for d in ['banknote', 'breastcancer', 'htru2']:
        run_svm(i, d)
        run_tree(i, d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)
    print('Running full experiment on numeric datasets (sklearn)...')

    start = time.time()
    run(args.idx)
    time_elapsed = time.time() - start
    print('Total time spend on numeric datasets (sklearn):', str(datetime.timedelta(seconds=time_elapsed)))
