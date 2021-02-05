#!/bin/bash

python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --run 5
python3 ./pipeline/full_pipeline_baard.py --data cifar10 --model resnet --attack apgd2 --eps 2.0 --run 5
