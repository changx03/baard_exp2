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

def run():
    for i in range(0, 5):
        run_full_pipeline_baard('mnist', 'dnn', 'result_' + str(i), seeds[i], os.path.join('params', 'baard_mnist_3.json'), 'apgd', 0.3)
        run_full_pipeline_baard('cifar10', 'resnet', 'result_' + str(i), seeds[i], os.path.join('params', 'baard_cifar10_3.json'), 'apgd', 0.3)


if __name__ == '__main__':
    run()
