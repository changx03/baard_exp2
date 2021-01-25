#!/bin/bash
# chmod +x ./bash/line_attack.sh
# ./bash/line_attack.sh | tee -a ./log/line_attack.log

# MNIST
# Attack
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack line --eps 0.0
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack line --eps 0.5
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack line --eps 1.0

# Defence
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_line_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_line_0.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_line_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_line_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_line_0.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_line_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_line_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_line_0.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_line_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_line_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_line_0.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_line_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_line_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_line_0.5
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_line_1.0


# CIFAR10
# Attack
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack line --eps 0.0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack line --eps 0.5
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack line --eps 1.0

# Defence
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_line_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_line_0.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_line_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_line_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_line_0.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_line_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_line_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_line_0.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_line_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_line_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_line_0.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_line_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_line_0.0
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_line_0.5
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_line_1.0
