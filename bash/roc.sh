#!/bin/bash

python3 ./experiments/baard_roc.py -d mnist -m dnn -a apgd -e 0.3
python3 ./experiments/baard_roc.py -d mnist -m dnn -a apgd2 -e 2.0
python3 ./experiments/baard_roc.py -d cifar10 -m resnet -a apgd -e 0.3
python3 ./experiments/baard_roc.py -d cifar10 -m resnet -a apgd2 -e 2.0
