import argparse
import os
import sys
import json
import time
import datetime

sys.path.append(os.getcwd())

from pipeline.generate_adv import run_generate_adv
from pipeline.full_pipeline_baard import run_full_pipeline_baard
from pipeline.full_pipeline_magnet import run_full_pipeline_magnet
from pipeline.evaluate_baard import run_evaluate_baard
from pipeline.evaluate_magnet import run_evaluate_magnet

with open(os.path.join('pipeline', 'seeds.json')) as j:
    json_obj = json.load(j)
    seeds = json_obj['seeds']


def run_baard(data, model, att, eps, idx):
    time_start = time.time()
    print('[Exp {}] Evaluating BAARD on {} {} {} attacks...'.format(idx, data, model, att))
    path = 'result_' + str(idx)
    json_mnist = os.path.join('params', 'baard_{}_3.json'.format(data))
    run_generate_adv(data, model, path, seeds[idx], att, eps=eps)
    run_evaluate_baard(data, model, path=path, seed=seeds[idx], json_param=json_mnist, att_name=att, eps=eps)
    time_elapsed = time.time() - time_start
    print('[Exp {}] Time spend on BAARD {} {} {}:'.format(idx, data, model, att), str(datetime.timedelta(seconds=time_elapsed)))
    print()


def run_magnet(data, model, att, eps, idx):
    time_start = time.time()
    print('[Exp {}] Evaluating MagNet on {} {} {} attacks...'.format(idx, data, model, att))
    path = 'result_' + str(idx)
    json_param = os.path.join('params', 'magnet_param.json')
    run_full_pipeline_magnet(data, model, path, seeds[idx], json_param, att, eps[2])
    run_evaluate_magnet(data, model, path=path, seed=seeds[idx], json_param=json_param, att_name=att, eps=eps)
    time_elapsed = time.time() - time_start
    print('[Exp {}] Time spend on MagNet {} {} {}:'.format(idx, data, model, att), str(datetime.timedelta(seconds=time_elapsed)))
    print()


def run(i):
    eps = [0.063, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    run_full_pipeline_baard('mnist', 'dnn', 'result_' + str(i), seeds[i], os.path.join('params', 'baard_mnist_3.json'), 'apgd', eps[2])
    run_full_pipeline_magnet('mnist', 'dnn', 'result_' + str(i), seeds[i], os.path.join('params', 'magnet_param.json'), 'apgd', eps[2])
    run_baard('mnist', 'dnn', 'apgd', eps, i)
    run_magnet('mnist', 'dnn', 'apgd', eps, i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    run_baard('mnist', 'dnn', 'apgd2', eps, i)
    run_magnet('mnist', 'dnn', 'apgd2', eps, i)

    eps = [0.031, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    run_full_pipeline_baard('cifar10', 'resnet', 'result_' + str(i), seeds[i], os.path.join('params', 'baard_cifar10_3.json'), 'apgd', eps[2])
    run_full_pipeline_magnet('cifar10', 'resnet', 'result_' + str(i), seeds[i], os.path.join('params', 'magnet_param.json'), 'apgd', eps[2])
    run_baard('cifar10', 'resnet', 'apgd', eps, i)
    run_magnet('cifar10', 'resnet', 'apgd', eps, i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    run_baard('cifar10', 'resnet', 'apgd2', eps, i)
    run_magnet('cifar10', 'resnet', 'apgd2', eps, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, required=True, choices=list(range(len(seeds))))
    args = parser.parse_args()
    print(args)

    run(args.idx)
