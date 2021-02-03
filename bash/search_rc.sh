#!/bin/bash

python3 ./experiments/rc_search_r.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.2
python3 ./experiments/rc_search_r.py --data breastcancer --pretrained breastcancer_400.pt --adv breastcancer_basic120_cw2_0.0
python3 ./experiments/rc_search_r.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_0.0
python3 ./experiments/rc_search_r.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_0.0
python3 ./experiments/rc_search_r.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_0.0
python3 ./experiments/rc_search_r.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_0.0

MODELS=("svm" "tree")
DATASETS=("banknote" "breastcancer" "htru2")

for MOD in "${MODELS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        python3 ./experiments/rc_search_r_sklearn.py --data $DATA --model $MOD --adv "boundary_0.3"
    done
done
