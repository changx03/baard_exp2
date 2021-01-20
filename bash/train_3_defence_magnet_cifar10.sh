#!/bin/bash
# chmod +x ./bash/train_3_defence_magnet_cifar10.sh
# ./bash/train_3_defence_magnet_cifar10.sh | tee -a ./log/train_3_defence_magnet_cifar10.log

# cifar10
# resnet
python3 ./experiments/train_magnet_autoencoder.py --data cifar10 --pretrained cifar10_resnet_200.pt --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.2 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.031 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_0.01 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_100.0 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_0.001 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_1e-06 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.2 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.031 --defence magnet --param magnet_param.json

# vgg
python3 ./experiments/train_magnet_autoencoder.py --data cifar10 --pretrained cifar10_vgg_200.pt --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.2 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.031 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_0.01 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_100.0 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_0.001 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_1e-06 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.2 --defence magnet --param magnet_param.json
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.031 --defence magnet --param magnet_param.json
