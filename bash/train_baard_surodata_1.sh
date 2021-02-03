#!/bin/bash

python3 ./experiments/train_baard_adv.py --data mnist --model basic --eps 2.0 --test 0
python3 ./experiments/train_baard_adv.py --data cifar10 --model resnet --eps 2.0 --test 0
python3 ./experiments/train_baard_adv.py --data cifar10 --model vgg --eps 2.0 --test 0
python3 ./experiments/train_baard_adv.py --data mnist --model basic --eps 2.0 --test 1 --n_samples 1000
python3 ./experiments/train_baard_adv.py --data cifar10 --model resnet --eps 2.0 --test 1 --n_samples 1000
python3 ./experiments/train_baard_adv.py --data cifar10 --model vgg --eps 2.0 --test 1 --n_samples 1000
