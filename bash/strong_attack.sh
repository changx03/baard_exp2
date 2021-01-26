#!/bin/bash
# chmod +x ./bash/strong_attack.sh
# ./bash/strong_attack.sh | tee ./log/strong_attack.log

# Attack
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 1.5
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd1 --eps 1.5
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd2 --eps 1.5
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 1.5

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 1.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd1 --eps 1.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd2 --eps 1.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 1.5

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 1.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd1 --eps 1.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd2 --eps 1.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 1.5

# BAARD
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv mnist_basic_apgd_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_apgd_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv mnist_basic_apgd1_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_apgd1_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv mnist_basic_apgd2_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_apgd2_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv mnist_basic_fgsm_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_resnet_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_resnet_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_resnet_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_resnet_fgsm_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_vgg_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_vgg_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_vgg_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_vgg_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_vgg_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_vgg_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_2.json --suffix 2stage --adv cifar10_vgg_fgsm_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_vgg_fgsm_1.5

# FS
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_apgd_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_apgd1_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_apgd2_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_vgg_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_vgg_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_vgg_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_vgg_fgsm_1.5

# LID
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_apgd_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_apgd1_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_apgd2_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_vgg_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_vgg_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_vgg_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_vgg_fgsm_1.5

# MagNet
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_apgd_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_apgd1_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_apgd2_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_vgg_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_vgg_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_vgg_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_vgg_fgsm_1.5

# RC
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd1_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd2_1.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_fgsm_1.5

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd1_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd2_1.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_fgsm_1.5
