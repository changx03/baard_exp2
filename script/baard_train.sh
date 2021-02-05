#!/bin/bash

python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd --eps 0.3 --idx 0  --json "./params/baard_mnist_3.json"
python3 ./pipeline/full_pipeline_baard.py --data cifar10 --model resnet --attack apgd --eps 0.3 --idx 0  --json "./params/baard_cifar10_3.json"
