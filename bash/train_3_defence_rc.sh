#!/bin/bash
# chmod +x ./bash/train_3_defence_rc.sh
# ./bash/train_3_defence_rc.sh | tee -a ./log/train_3_defence_rc.log

# banknote
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.05 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_cw2_0.01 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_cw2_100.0 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_deepfool_0.001 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_deepfool_1e-06 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_fgsm_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_fgsm_0.05 --defence rc --param rc_param.json

# cifar10
#resnet
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.031 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_0.01 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_100.0 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_0.001 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_1e-06 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.031 --defence rc --param rc_param.json

# vgg
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.031 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_0.01 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_100.0 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_0.001 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_1e-06 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.031 --defence rc --param rc_param.json

# htru2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_apgd_0.2 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_apgd_0.05 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_0.01 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_100.0 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_deepfool_0.001 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_deepfool_1e-06 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_fgsm_0.2 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_fgsm_0.05 --defence rc --param rc_param_htru2.json
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_boundary_0.3 --defence rc --param rc_param_htru2.json

# mnist
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.3 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.063 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_boundary_0.3 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_0.01 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_100.0 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_0.001 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_1e-06 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.3 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.063 --defence rc --param rc_param.json

# segment
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.05 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_0.01 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_100.0 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_0.001 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_1e-06 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.05 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_boundary_0.3 --defence rc --param rc_param.json

# texture
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_apgd_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_apgd_0.05 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_cw2_0.01 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_cw2_100.0 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_deepfool_0.001 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_deepfool_1e-06 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_fgsm_0.2 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_fgsm_0.05 --defence rc --param rc_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_boundary_0.3 --defence rc --param rc_param.json
