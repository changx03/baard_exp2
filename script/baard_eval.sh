#!/bin/bash

python3 ./pipeline/evaluate_baard.py --data mnist --model dnn --attack apgd --eps 0.063 0.3 0.6 1.0 1.5 --idx 0
python3 ./pipeline/evaluate_baard.py --data cifar10 --model resnet --attack apgd --eps 0.031 0.3 0.6 1.0 1.5 --idx 0
