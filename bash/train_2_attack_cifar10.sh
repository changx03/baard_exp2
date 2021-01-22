#!/bin/bash
# chmod +x ./bash/train_2_attack_cifar10.sh
# ./bash/train_2_attack_cifar10.sh | tee -a ./log/train_2_attack_cifar10.log

# cifar10 resnet
# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd1 --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd1 --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd1 --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd1 --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd2 --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd2 --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd2 --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd2 --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cw2 --eps 100

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cwinf --eps 0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cwinf --eps 10
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cwinf --eps 100

# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack deepfool --eps 1e-6

# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack shadow --batch_size 400


# cifar10 vgg
# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd1 --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd1 --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd1 --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd1 --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd2 --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd2 --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd2 --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd2 --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cw2 --eps 100

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cwinf --eps 0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cwinf --eps 10
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cwinf --eps 100

# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack deepfool --eps 1e-6

# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 1.0

python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack shadow --batch_size 400