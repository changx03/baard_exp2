#!/bin/bash
# chmod +x ./bash/train_3_defence_lid_mnist.sh
# ./bash/train_3_defence_lid_mnist.sh | tee -a ./log/train_3_defence_lid_mnist.log

# mnist
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.3 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.063 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_boundary_0.3 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_0.01 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_100.0 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_0.001 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_1e-06 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.3 --defence lid --param ./params/lid_param.json
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.063 --defence lid --param ./params/lid_param.json
