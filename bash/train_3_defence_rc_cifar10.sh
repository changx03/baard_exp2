#!/bin/bash
# chmod +x ./bash/train_3_defence_rc_cifar10.sh
# ./bash/train_3_defence_rc_cifar10.sh | tee ./log/train_3_defence_rc_cifar10.log

# cifar10
# resnet
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd1_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd1_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd1_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd_1.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd1_1.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd2_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd2_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd2_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_apgd2_1.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_cw2_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_cw2_10.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_cw2_5.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_cwinf_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_cwinf_10.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_cwinf_5.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_deepfool_1e-06
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_fgsm_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_fgsm_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_fgsm_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_fgsm_1.0

# vgg
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd1_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd1_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd1_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd_1.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd1_1.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd2_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd2_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd2_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_apgd2_1.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_cw2_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_cw2_10.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_cw2_5.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_cwinf_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_cwinf_10.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_cwinf_5.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_deepfool_1e-06
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_fgsm_0.031
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_fgsm_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_fgsm_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_vgg_fgsm_1.0
