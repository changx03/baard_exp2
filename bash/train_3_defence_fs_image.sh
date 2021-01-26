#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_image.sh
# ./bash/train_3_defence_fs_image.sh | tee ./log/train_3_defence_fs_image.log

# This file trains the squeezers
python3 ./experiments/train_fs_image.py --data mnist --pretrained mnist_200.pt --param ./params/fs_param.json
python3 ./experiments/train_fs_image.py --data cifar10 --pretrained cifar10_resnet_200.pt --param ./params/fs_param.json
python3 ./experiments/train_fs_image.py --data cifar10 --pretrained cifar10_vgg_200.pt --param ./params/fs_param.json