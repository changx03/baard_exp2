#!/bin/bash
# chmod +x ./bash/train_extra_num.sh
# ./bash/train_extra_num.sh | tee ./log/train_extra_num.log

runExp ()
{
    DATA=$1
    MODEL_FILE="${DATA}_400.pt"
    MODEL_NAME=$2
    echo "[data=${DATA}, model_file=${MODEL_FILE}, model_name=${MODEL_NAME}]"
    for ATTACK in apgd1 apgd2
    do
        for EPSILON in 2.0 3.0
        do
            echo "Running attack ${ATTACK}_${EPSILON}..."
            python3 ./experiments/train_attacks.py --data $DATA --pretrained $MODEL_FILE --attack $ATTACK --eps $EPSILON
            echo "Running defence ${ATTACK}_${EPSILON}..."
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence baard --param ./params/baard_param_2s.json --suffix 2stage --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence baard --param ./params/baard_param_3s.json --suffix 3stage --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence lid --param ./params/lid_param.json --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence rc --param ./params/rc_param.json --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
        done
    done
}

runExp banknote basic16
runExp breastcancer basic120
runExp htru2 basic32
