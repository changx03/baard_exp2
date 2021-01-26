#!/bin/bash
# chmod +x ./bash/watermark.sh
# ./bash/watermark.sh | tee -a ./log/watermark.log

# python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack watermark --eps 0.1
# python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack watermark --eps 0.3
# python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack watermark --eps 0.6
# python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack watermark --eps 1.0

# python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_watermark_0.1
# python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_watermark_0.3
# python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_watermark_0.6
# python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv mnist_basic_watermark_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_watermark_0.1
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_watermark_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_watermark_0.6
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence fs --param ./params/fs_param.json --adv mnist_basic_watermark_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_watermark_0.1
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_watermark_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_watermark_0.6
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence lid --param ./params/lid_param.json --adv mnist_basic_watermark_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_watermark_0.1
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_watermark_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_watermark_0.6
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence magnet --param ./params/magnet_param.json --adv mnist_basic_watermark_1.0

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_watermark_0.1
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_watermark_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_watermark_0.6
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_watermark_1.0


# CIFAR10
# Attack
# python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack watermark --eps 0.1
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack watermark --eps 0.3
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack watermark --eps 0.6
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack watermark --eps 1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_watermark_0.1
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_watermark_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_watermark_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence baard --param ./params/baard_param_3.json --suffix 3stage --adv cifar10_resnet_watermark_1.0

# python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_watermark_0.1
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_watermark_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_watermark_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence fs --param ./params/fs_param.json --adv cifar10_resnet_watermark_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_watermark_0.1
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_watermark_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_watermark_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence lid --param ./params/lid_param.json --adv cifar10_resnet_watermark_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_watermark_0.1
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_watermark_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_watermark_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence magnet --param ./params/magnet_param.json --adv cifar10_resnet_watermark_1.0

python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_watermark_0.1
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_watermark_0.3
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_watermark_0.6
python3 ./experiments/train_defences.py --data cifar10 --pretrained cifar10_resnet_200.pt --defence rc --param ./params/rc_param.json --adv cifar10_resnet_watermark_1.0
