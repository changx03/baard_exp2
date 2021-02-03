#!/bin/bash
# chmod +x ./bash/train_3_defence_magnet_cifar10.sh
# ./bash/train_3_defence_magnet_cifar10.sh | tee ./log/train_3_defence_magnet_cifar10.log

ATTACKS=("apgd_0.031" "apgd_0.3" "apgd_0.6" "apgd_1.0" "apgd_1.5" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "cw2_0.0" "cw2_5.0" "cw2_10.0" "deepfool_1e-06" "fgsm_0.031" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "watermark_0.3" "watermark_0.6")
MODELS=("resnet" "vgg")
DATA="cifar10"
DEF="magnet"
for MOD in "${MODELS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        echo "Testing $DATA $MOD $ATT..."
        python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_"$MOD"_200.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
        echo
    done
done
