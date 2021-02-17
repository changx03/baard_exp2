import argparse
import os
import sys
import time
import datetime

sys.path.append(os.getcwd())

from experiments.pytorch_attack_against_baard_img import pytorch_attack_against_baard_img
from experiments.pytorch_attack_against_fs_img import pytorch_attack_against_fs_img
from experiments.pytorch_attack_against_lid_img import pytorch_attack_against_lid_img
from experiments.pytorch_attack_against_magnet_img import pytorch_attack_against_magnet_img
from experiments.pytorch_attack_against_rc_img import pytorch_attack_against_rc_img


def run_small_batch(att, eps, i, data='cifar10', model='vgg'):
    pytorch_attack_against_baard_img(data, model, att, eps, i)
    pytorch_attack_against_fs_img(data, model, att, eps, i)
    pytorch_attack_against_lid_img(data, model, att, eps, i)
    pytorch_attack_against_magnet_img(data, model, att, eps, i)
    pytorch_attack_against_rc_img(data, model, att, eps, i)


def run(i):
    eps = [0.031, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        run_small_batch(att, eps, i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    run_small_batch(att, eps, i)

    eps = [0.]
    att = 'deepfool'
    run_small_batch(att, eps, i)

    eps = [0., 5., 10.]
    att = 'cw2'
    run_small_batch(att, eps, i)

    eps = [1.]
    att = 'line'
    run_small_batch(att, eps, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)
    print('Running full experiment on MNIST...')

    start = time.time()
    run(args.idx)
    time_elapsed = time.time() - start
    print('Total time spend on MNIST:', str(datetime.timedelta(seconds=time_elapsed)))
