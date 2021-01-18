#!/bin/bash
# chmod +x ./bash/train_3_defence_lid_cifar10.sh
# ./bash/train_3_defence_lid_cifar10.sh | tee -a ./log/train_3_defence_lid_cifar10.log

# cifar10
#resnet
# The resnet model does not support sequential out of the box!
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.2 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.031 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_0.01 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_100.0 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_0.001 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_1e-06 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.2 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.031 --defence lid --param lid_param.json

# vgg
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.2 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.031 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_0.01 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_100.0 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_0.001 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_1e-06 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.2 --defence lid --param lid_param.json
# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.031 --defence lid --param lid_param.json
