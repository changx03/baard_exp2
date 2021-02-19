#!/bin/bash

INDICES=(0 1 2 3 4)

for I in "${INDICES[@]}"; do
    python ./run/exp_mnist.py -i $I > ./log/exp_mnist_$I.out
done