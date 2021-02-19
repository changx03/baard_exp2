#!/bin/bash

INDICES=(0 1 2 3 4)

for I in "${INDICES[@]}"; do
    nohup python ./run/exp_num_sklearn.py -i $I > ./log/exp_num_sklearn_$I.out &
done