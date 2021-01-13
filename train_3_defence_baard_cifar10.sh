#!/bin/bash
# chmod +x train_3_defence_baard_cifar10.sh
# ./train_3_defence_baard_cifar10.sh | tee -a defence_baard_cifar.log

# cifar10
#resnet
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.2 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.2 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.031 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_apgd_0.031 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_0.01 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_0.01 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_100.0 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_cw2_100.0 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_0.001 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_0.001 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_1e-06 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_deepfool_1e-06 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.2 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.2 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.031 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --adv cifar10_resnet_fgsm_0.031 --defence baard --param baard_param_3.json --suffix 3tages

# vgg
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.2 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.2 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.031 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_apgd_0.031 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_0.01 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_0.01 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_100.0 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_cw2_100.0 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_0.001 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_0.001 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_1e-06 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_deepfool_1e-06 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.2 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.2 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.031 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_vgg_200.pt --adv cifar10_vgg_fgsm_0.031 --defence baard --param baard_param_3.json --suffix 3tages
