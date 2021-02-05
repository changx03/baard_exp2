#!/bin/bash

python3 ./pipeline/generate_adv.py --data mnist --model dnn --attack apgd --eps 0.063 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0
python3 ./pipeline/full_pipeline_magnet.py --data mnist --model dnn --attack apgd --eps 0.063 --idx 0
python3 ./pipeline/evaluate_magnet.py --data mnist --model dnn --attack apgd --eps 0.063 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0

python3 ./pipeline/generate_adv.py --data mnist --model dnn --attack apgd2 --eps 1.0 1.5 2.0 3.0 4.0 5.0 --idx 0
python3 ./pipeline/full_pipeline_magnet.py --data mnist --model dnn --attack apgd2 --eps 2.0 --idx 0
python3 ./pipeline/evaluate_magnet.py --data mnist --model dnn --attack apgd2 --eps 1.0 1.5 2.0 3.0 4.0 5.0 --idx 0

python3 ./pipeline/generate_adv.py --data mnist --model dnn --attack fgsm --eps 0.063 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0
python3 ./pipeline/full_pipeline_magnet.py --data mnist --model dnn --attack fgsm --eps 0.063 --idx 0
python3 ./pipeline/evaluate_magnet.py --data mnist --model dnn --attack fgsm --eps 0.063 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0

python3 ./pipeline/generate_adv.py --data cifar10 --model resnet --attack apgd --eps 0.031 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0
python3 ./pipeline/full_pipeline_magnet.py --data cifar10 --model resnet --attack apgd --eps 0.031 --idx 0
python3 ./pipeline/evaluate_magnet.py --data cifar10 --model resnet --attack apgd --eps 0.031 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0

python3 ./pipeline/generate_adv.py --data cifar10 --model resnet --attack apgd2 --eps 1.0 1.5 2.0 3.0 4.0 5.0 --idx 0
python3 ./pipeline/full_pipeline_magnet.py --data cifar10 --model resnet --attack apgd2 --eps 2.0 --idx 0
python3 ./pipeline/evaluate_magnet.py --data cifar10 --model resnet --attack apgd2 --eps 1.0 1.5 2.0 3.0 4.0 5.0 --idx 0

python3 ./pipeline/generate_adv.py --data cifar10 --model resnet --attack fgsm --eps 0.031 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0
python3 ./pipeline/full_pipeline_magnet.py --data cifar10 --model resnet --attack fgsm --eps 0.031 --idx 0
python3 ./pipeline/evaluate_magnet.py --data cifar10 --model resnet --attack fgsm --eps 0.031 0.1 0.2 0.3 0.6 1.0 1.5 2.0 --idx 0

