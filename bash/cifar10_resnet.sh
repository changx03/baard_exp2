#!/bin/bash

INDICES=(0 1 2 3 4)

for I in "${INDICES[@]}"; do
    nohup python ./run/exp_cifar10_resnet.py -i $I > ./log/exp_cifar10_resnet_$I.out &
done