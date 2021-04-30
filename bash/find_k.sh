#!/bin/bash

INDICES=(0 1 2 3 4)

for I in "${INDICES[@]}"; do
    python3 ./experiments/find_k.py -d mnist -m dnn -e 0.063 0.3 1.0 2.0 -i $I
    python3 ./experiments/find_k.py -d cifar10 -m resnet -e 0.031 0.3 1.0 2.0 -i $I
done
