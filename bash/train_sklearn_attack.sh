#!/bin/bash

DATASETS=("banknote" "breastcancer" "htru2")

# EPSILONS=(0.05 0.2 0.4 1.0)
# ATTACKS=("bim" "fgsm")
# MOD="svm"

# for DATA in "${DATASETS[@]}"; do
#     for ATT in "${ATTACKS[@]}"; do
#         for EPS in "${EPSILONS[@]}"; do
#             python3 ./experiments/train_attacks_sklearn.py --data $DATA --model $MOD --attack $ATT --eps $EPS
#         done    
#     done
# done

# for DATA in "${DATASETS[@]}"; do
#     python3 ./experiments/train_attacks_sklearn.py --data $DATA --model $MOD --attack boundary
# done
    

MOD="tree"
ATTACKS=("boundary" "tree")
for DATA in "${DATASETS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        python3 ./experiments/train_attacks_sklearn.py --data $DATA --model $MOD --attack $ATT
    done
done
