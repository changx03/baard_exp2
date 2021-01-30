#!/bin/bash
# chmod +x ./bash/train_extra_img.sh
# ./bash/train_extra_img.sh | tee ./log/train_extra_img.log

runExp ()
{
    DATA=$1
    MODEL_NAME=$2
    MODEL_FILE=$3
    echo "[data=${DATA}, model_file=${MODEL_FILE}, model_name=${MODEL_NAME}]"
    for ATTACK in apgd1 apgd2
    do
        for EPSILON in 2.0 3.0 5.0
        do
            echo "Running attack ${ATTACK}_${EPSILON}..."
            python3 ./experiments/train_attacks.py --data $DATA --pretrained $MODEL_FILE --attack $ATTACK --eps $EPSILON
            echo "Running defence ${ATTACK}_${EPSILON}..."
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence lid --param ./params/lid_param.json --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence magnet --param ./params/magnet_param.json --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
            python3 ./experiments/train_defences.py --data $DATA --pretrained $MODEL_FILE --defence rc --param ./params/rc_param.json --adv "${DATA}_${MODEL_NAME}_${ATTACK}_${EPSILON}"
        done
    done
    echo "---------------------------------------------------------------------"
}

runExp mnist basic mnist_200.pt
runExp cifar10 resnet cifar10_resnet_200.pt
runExp cifar10 vgg cifar10_vgg_200.pt
