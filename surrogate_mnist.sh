#!/bin/bash

# python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --json "./params/baard_mnist_3.json" --idx 0
python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --json "./params/baard_mnist_3.json" --idx 1
python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --json "./params/baard_mnist_3.json" --idx 2
python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --json "./params/baard_mnist_3.json" --idx 3
python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --json "./params/baard_mnist_3.json" --idx 4
