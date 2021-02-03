#!/bin/bash
 
DATASETS=("banknote" "breastcancer" "htru2")

ATTACKS=("bim_0.05" "bim_0.2" "bim_0.4" "bim_1.0" "boundary_0.3" "fgsm_0.05" "fgsm_0.2" "fgsm_0.4" "fgsm_1.0")
MOD="svm"

for DATA in "${DATASETS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence rc --param "./params/rc_param_"$DATA"_"$MOD".json"
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence baard --param "./params/baard_param_2s.json" --suffix 2stage
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence baard --param "./params/baard_param_3s.json" --suffix 3stage
    done
done


ATTACKS=("boundary_0.3" "tree_0.3")
MOD="tree"

for DATA in "${DATASETS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence rc --param "./params/rc_param_"$DATA"_"$MOD".json"
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence baard --param "./params/baard_param_2s.json" --suffix 2stage
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence baard --param "./params/baard_param_3s.json" --suffix 3stage
    done
done
