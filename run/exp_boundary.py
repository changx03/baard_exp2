import argparse
import os
import sys
import time
import datetime

sys.path.append(os.getcwd())

from experiments.pytorch_attack_against_baard_img import pytorch_attack_against_baard_img
from experiments.pytorch_attack_against_baard_num import pytorch_attack_against_baard_num
from experiments.pytorch_attack_against_fs_img import pytorch_attack_against_fs_img
from experiments.pytorch_attack_against_lid_img import pytorch_attack_against_lid_img
from experiments.pytorch_attack_against_lid_num import pytorch_attack_against_lid_num
from experiments.pytorch_attack_against_magnet_img import pytorch_attack_against_magnet_img
from experiments.pytorch_attack_against_magnet_num import pytorch_attack_against_magnet_num
from experiments.pytorch_attack_against_rc_img import pytorch_attack_against_rc_img
from experiments.pytorch_attack_against_rc_num import pytorch_attack_against_rc_num


def run_small_batch_num(data, att, eps, i):
    pytorch_attack_against_baard_num(data, att, eps, i)
    pytorch_attack_against_lid_num(data, att, eps, i)
    pytorch_attack_against_magnet_num(data, att, eps, i)
    pytorch_attack_against_rc_num(data, att, eps, i)


def run_small_batch_mnist(att, eps, i, data='mnist', model='dnn'):
    pytorch_attack_against_baard_img(data, model, att, eps, i)
    pytorch_attack_against_fs_img(data, model, att, eps, i)
    pytorch_attack_against_lid_img(data, model, att, eps, i)
    pytorch_attack_against_magnet_img(data, model, att, eps, i)
    pytorch_attack_against_rc_img(data, model, att, eps, i)


def run(i):
    for d in ['banknote', 'breastcancer', 'htru2']:
        run_small_batch_num(d, 'boundary', [0.], i)

    run_small_batch_mnist('boundary', [0.], i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)
    print('Running full experiment on boundary attack...')

    start = time.time()
    run(args.idx)
    time_elapsed = time.time() - start
    print('Total time spend on boundary attack:', str(datetime.timedelta(seconds=time_elapsed)))
