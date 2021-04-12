#!/bin/bash

for IDX in {1..4}; do
    echo "Index=$IDX"
    python3 ./run/exp_num_nn.py -i $IDX
    python3 ./run/exp_mnist.py -i $IDX
    python3 ./run/exp_cifar10_resnet.py -i $IDX
    python3 ./run/exp_cifar10_vgg.py -i $IDX
done
