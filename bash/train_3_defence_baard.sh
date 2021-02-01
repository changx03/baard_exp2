#!/bin/bash
# chmod +x ./bash/train_3_defence_baard.sh
# ./bash/train_3_defence_baard.sh | tee ./log/train_3_defence_baard.log

ATTACKS=("apgd_0.05" "apgd_0.2" "apgd_0.4" "apgd1_0.2" "apgd1_0.4" "apgd_1.0" "apgd1_1.0" "apgd1_2.0" "apgd1_3.0" "apgd2_0.2" "apgd2_0.4" "apgd2_1.0" "apgd2_2.0" "apgd2_3.0" "boundary_0.3" "cw2_0.0" "cw2_10.0" "cw2_5.0" "cwinf_0.0" "cwinf_5.0" "deepfool_1e-06" "fgsm_0.05" "fgsm_0.2" "fgsm_0.4" "fgsm_1.0")

DATA="banknote"
MOD="basic16"
DEF="baard"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_2s.json" --suffix 2stage --adv $DATA"_"$MOD"_"$ATT
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_3s.json" --suffix 3stage --adv $DATA"_"$MOD"_"$ATT
done


DATA="breastcancer"
MOD="basic120"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_2s.json" --suffix 2stage --adv $DATA"_"$MOD"_"$ATT
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_3s.json" --suffix 3stage --adv $DATA"_"$MOD"_"$ATT
done


DATA="htru2"
MOD="basic32"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_2s.json" --suffix 2stage --adv $DATA"_"$MOD"_"$ATT
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_3s.json" --suffix 3stage --adv $DATA"_"$MOD"_"$ATT
done


ATTACKS=("apgd_0.063" "apgd_0.3" "apgd1_0.3" "apgd_1.0" "apgd1_1.0" "apgd1_1.5" "apgd1_2.0" "apgd1_3.0" "apgd1_5.0" "apgd_1.5" "apgd2_0.063" "apgd2_0.3" "apgd2_0.6" "apgd2_1.0" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "boundary_0.3" "cw2_0.0" "cw2_10.0" "cw2_5.0" "cwinf_0.0" "cwinf_5.0" "deepfool_1e-06" "fgsm_0.063" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "shadow_0.3" "watermark_0.1" "watermark_0.3" "watermark_0.6" "watermark_1.0")
MOD="basic"
DATA="mnist"
for ATT in "${ATTACKS[@]}"; do
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_2.json" --suffix 2stage --adv $DATA"_"$MOD"_"$ATT
    python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_400.pt" --defence $DEF --param "./params/"$DEF"_param_3.json" --suffix 3stage --adv $DATA"_"$MOD"_"$ATT
done


ATTACKS=("apgd_0.031" "apgd_0.3" "apgd_0.6" "apgd1_0.3" "apgd1_0.6" "apgd_1.0" "apgd1_1.0" "apgd1_1.5" "apgd1_2.0" "apgd1_3.0" "apgd1_5.0" "apgd_1.5" "apgd2_0.3" "apgd2_0.6" "apgd2_1.0" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "cw2_0.0" "cw2_10.0" "cw2_5.0" "cwinf_0.0" "cwinf_5.0" "deepfool_1e-06" "fgsm_0.031" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "watermark_0.1" "watermark_0.3" "watermark_0.6" "watermark_1.0")
MODELS=("resnet" "vgg")
DATA="cifar10"
for MOD in "${MODELS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_"$MOD"_200.pt" --defence $DEF --param "./params/"$DEF"_param_2.json" --suffix 2stage --adv $DATA"_"$MOD"_"$ATT
        python3 ./experiments/train_defences.py --data $DATA --pretrained $DATA"_"$MOD"_200.pt" --defence $DEF --param "./params/"$DEF"_param_3.json" --suffix 3stage --adv $DATA"_"$MOD"_"$ATT
    done
done

# Example
# python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence baard --param ./params/baard_param_3s.json --suffix 3stage --adv banknote_basic16_apgd_0.2
# python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence baard --param ./params/baard_param_3s.json --suffix 3stage --adv breastcancer_basic120_apgd_0.2
# python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_cw2_0.0
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_cw2_0.0
