#!/bin/bash

python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd --eps 0.3 --idx 0
python3 ./pipeline/full_pipeline_baard.py --data cifar10 --model resnet --attack apgd --eps 0.3 --idx 0
