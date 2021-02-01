#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_image.sh
# ./bash/train_3_defence_fs_image.sh | tee ./log/train_3_defence_fs_image.log

# This file trains the squeezers
echo "Training squeezers for MNIST..."
python3 ./experiments/train_fs_image.py --data mnist --pretrained mnist_200.pt --param ./params/fs_param.json
echo "\nTraining squeezers for CIFAR10 resnet..."
python3 ./experiments/train_fs_image.py --data cifar10 --pretrained cifar10_resnet_200.pt --param ./params/fs_param.json
echo "\nTraining squeezers for CIFAR10 vgg..."
python3 ./experiments/train_fs_image.py --data cifar10 --pretrained cifar10_vgg_200.pt --param ./params/fs_param.json

# mnist
ATTACKS=("apgd_0.063" "apgd_0.3" "apgd1_0.3" "apgd_1.0" "apgd1_1.0" "apgd1_1.5" "apgd1_2.0" "apgd1_3.0" "apgd1_5.0" "apgd_1.5" "apgd2_0.063" "apgd2_0.3" "apgd2_0.6" "apgd2_1.0" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "boundary_0.3" "cw2_0.0" "cw2_10.0" "cw2_5.0" "cwinf_0.0" "cwinf_5.0" "deepfool_1e-06" "fgsm_0.063" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "shadow_0.3" "watermark_0.1" "watermark_0.3" "watermark_0.6" "watermark_1.0")
MOD="basic"
DATA="mnist"
DEF="fs"
for ATT in "${ATTACKS[@]}"; do
    echo "\nTesting $DATA $MOD $ATT..."
    python3 ./experiments/train_defences_old.py --data $DATA --pretrained $DATA"_200.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
done

# cifar10
ATTACKS=("apgd_0.031" "apgd_0.3" "apgd_0.6" "apgd1_0.3" "apgd1_0.6" "apgd_1.0" "apgd1_1.0" "apgd1_1.5" "apgd1_2.0" "apgd1_3.0" "apgd1_5.0" "apgd_1.5" "apgd2_0.3" "apgd2_0.6" "apgd2_1.0" "apgd2_1.5" "apgd2_2.0" "apgd2_3.0" "apgd2_5.0" "cw2_0.0" "cw2_10.0" "cw2_5.0" "cwinf_0.0" "cwinf_5.0" "deepfool_1e-06" "fgsm_0.031" "fgsm_0.3" "fgsm_0.6" "fgsm_1.0" "fgsm_1.5" "line_0.0" "line_0.5" "line_1.0" "watermark_0.1" "watermark_0.3" "watermark_0.6" "watermark_1.0")
MODELS=("resnet" "vgg")
DATA="cifar10"
DEF="fs"
for MOD in "${MODELS[@]}"; do
    for ATT in "${ATTACKS[@]}"; do
        echo "\nTesting $DATA $MOD $ATT..."
        python3 ./experiments/train_defences_old.py --data $DATA --pretrained $DATA"_"$MOD"_200.pt" --defence $DEF --param "./params/"$DEF"_param.json" --adv $DATA"_"$MOD"_"$ATT
    done
done
