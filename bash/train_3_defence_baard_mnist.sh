#!/bin/bash
# chmod +x ./bash/train_3_defence_baard_mnist.sh
# ./bash/train_3_defence_baard_mnist.sh | tee -a ./log/train_3_defence_baard_mnist.log

# mnist
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.3 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.3 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.063 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_apgd_0.063 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_boundary_0.3 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_boundary_0.3 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_0.01 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_0.01 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_100.0 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_cw2_100.0 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_0.001 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_0.001 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_1e-06 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_deepfool_1e-06 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.3 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.3 --defence baard --param baard_param_3.json --suffix 3tages

python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.063 --defence baard --param baard_param_2.json --suffix 2tages
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --adv mnist_basic_fgsm_0.063 --defence baard --param baard_param_3.json --suffix 3tages
