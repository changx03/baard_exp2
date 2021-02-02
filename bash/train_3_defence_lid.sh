#!/bin/bash
# chmod +x ./bash/train_3_defence_baard.sh
# ./bash/train_3_defence_baard.sh | tee ./log/train_3_defence_baard.log

ATTACKS=("apgd_0.05" "apgd_0.2" "apgd_0.4" "apgd_1.0" "apgd2_0.4" "apgd2_1.0" "apgd2_2.0" "apgd2_3.0" "boundary_0.3" "cw2_0.0" "cw2_5.0" "cw2_10.0" "deepfool_1e-06" "fgsm_0.05" "fgsm_0.2" "fgsm_0.4" "fgsm_1.0")

DATA="banknote"
MOD="basic16"
DEF="lid"

for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
done


DATA="breastcancer"
MOD="basic120"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
done


DATA="htru2"
MOD="basic32"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
done


ATTACKS=("apgd_0.063" "apgd_0.3" "apgd_0.6" "apgd_1.0" "apgd_1.5" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "cw2_0.0" "cw2_5.0" "cw2_10.0" "deepfool_1e-06" "fgsm_0.063" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "watermark_0.3" "watermark_0.6")
MOD="basic"
DATA="mnist"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_200.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
done


ATTACKS=("apgd_0.031" "apgd_0.3" "apgd_0.6" "apgd_1.0" "apgd_1.5" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "cw2_0.0" "cw2_5.0" "cw2_10.0" "deepfool_1e-06" "fgsm_0.031" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "watermark_0.3" "watermark_0.6")
MODELS=("resnet" "vgg")
DATA="cifar10"
for MOD in "${MODELS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_"$MOD"_200.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
    done
done
