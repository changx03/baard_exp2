#!/bin/bash

for IDX in {1..4}; do
    echo "Index=$IDX"
    python3 ./experiments/whitebox_baard.py -d mnist -i $IDX -e 1.0 2.0 3.0 5.0 8.0
    python3 ./experiments/whitebox_baard.py -d cifar10 -i $IDX -e 0.05 0.1 0.5 1.0 2.0
done
