#!/bin/bash
# chmod +x ./bash/train_svm_attack.sh
# ./bash/train_svm_attack.sh | tee -a ./log/train_svm_attack.log

DATASETS=("banknote" "breastcancer" "htru2")
EPSILONS=(0.05 0.2 0.4 1.0)
ATTACKS=("bim" "fgsm")

for DATA in "${DATASETS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        for EPS in "${EPSILONS[@]}"; do
            python3 ./experiments/train_attacks_svm.py --data $DATA --attack $ATT --eps $EPS
        done    
    done
done

for DATA in "${DATASETS[@]}"; do
    python3 ./experiments/train_attacks_svm.py --data $DATA --attack boundary
done
    