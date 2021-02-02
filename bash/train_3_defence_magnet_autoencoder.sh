#!/bin/bash
# chmod +x ./bash/train_3_defence_magnet_autoencoder.sh
# ./bash/train_3_defence_magnet_autoencoder.sh | tee ./log/train_3_defence_magnet_autoencoder.log

python3 ./experiments/train_magnet_autoencoder.py --data mnist --pretrained mnist_200.pt --param ./params/magnet_param.json
python3 ./experiments/train_magnet_autoencoder.py --data cifar10 --pretrained cifar10_resnet_200.pt --param ./params/magnet_param.json
python3 ./experiments/train_magnet_autoencoder.py --data cifar10 --pretrained cifar10_vgg_200.pt --param ./params/magnet_param.json
