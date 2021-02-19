#!/bin/bash

INDICES=(0 1 2 3 4)

for I in "${INDICES[@]}"; do
    nohup python ./run/exp_boundary_num.py -i $I > ./log/exp_boundary_num_$I.out &
done