#!/bin/bash
# chmod +x ./bash/train_sklearn_defence.sh
# ./bash/train_sklearn_defence.sh | tee -a ./log/train_sklearn_defence.log
 
ATTACKS=("bim_0.05" "bim_0.2" "bim_0.4" "bim_1.0" "boundary_0.3" "fgsm_0.05" "fgsm_0.2" "fgsm_0.4" "fgsm_1.0")
DATASETS=("banknote" "breastcancer" "htru2")
MOD="svm"
# DEF="rc"
# for DATA in "${DATASETS[@]}"; do
#     for ATT in "${ATTACKS[@]}"; do
#         python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence $DEF --param "./params/"$DEF"_param_"$DATA"_"$MOD".json"
#     done
# done


DEF="baard"
for DATA in "${DATASETS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence $DEF --param "./params/"$DEF"_param_2s.json" --suffix 2stage
        python3 ./experiments/train_defences_sklearn.py --data $DATA --model $MOD --adv $ATT --defence $DEF --param "./params/"$DEF"_param_3s.json" --suffix 3stage
    done
done
