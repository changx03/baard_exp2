#!/bin/bash
# chmod +x ./bash/train_3_defence_magnet_mnist.sh
# ./bash/train_3_defence_magnet_mnist.sh | tee -a ./log/train_3_defence_magnet_mnist.log

# mnist
python3 ./experiments/train_magnet_autoencoder.py --data mnist --pretrained mnist_200.pt --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.3 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.3 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.063 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_boundary_0.3 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_0.01 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_100.0 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_0.001 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_1e-06 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.3 --defence magnet --param ./params/magnet_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.063 --defence magnet --param ./params/magnet_param.json
