import argparse
import os
import sys
import time
import datetime

sys.path.append(os.getcwd())

from experiments.pytorch_attack_against_baard_num import pytorch_attack_against_baard_num
from experiments.pytorch_attack_against_lid_num import pytorch_attack_against_lid_num
from experiments.pytorch_attack_against_magnet_num import pytorch_attack_against_magnet_num
from experiments.pytorch_attack_against_rc_num import pytorch_attack_against_rc_num


def run_small_batch(data, att, eps, i):
    pytorch_attack_against_baard_num(data, att, eps, i)
    pytorch_attack_against_lid_num(data, att, eps, i)
    pytorch_attack_against_magnet_num(data, att, eps, i)
    pytorch_attack_against_rc_num(data, att, eps, i)


def run_per_dataset(i, data):
    eps = [0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        run_small_batch(data, att, eps, i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    run_small_batch(data, att, eps, i)

    eps = [0., 5., 10.]
    att = 'cw2'
    run_small_batch(data, att, eps, i)

    # eps = [0.]
    # att = 'boundary'
    # run_small_batch(data, att, eps, i)


def run(i):
    for d in ['banknote', 'breastcancer', 'htru2']:
        run_per_dataset(i, d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)
    print('Running full experiment on numeric datasets...')

    start = time.time()
    run(args.idx)
    time_elapsed = time.time() - start
    print('Total time spend on numeric datasets:', str(datetime.timedelta(seconds=time_elapsed)))
