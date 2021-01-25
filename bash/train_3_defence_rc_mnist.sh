#!/bin/bash
# chmod +x ./bash/train_3_defence_rc_mnist.sh
# ./bash/train_3_defence_rc_mnist.sh | tee ./log/train_3_defence_rc_mnist.log

# mnist
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd_0.063
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd1_0.063
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd1_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd_1.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd1_1.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd2_0.063
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd2_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd2_0.6
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_apgd2_1.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_bim_0.063
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_bim_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_bim_1.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_boundary_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_cw2_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_cw2_10.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_cw2_5.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_cwinf_0.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_cwinf_10.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_cwinf_5.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_deepfool_1e-06
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_fgsm_0.063
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_fgsm_0.3
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_fgsm_0.6
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_fgsm_1.0
python3 ./experiments/train_defences.py --data mnist --pretrained mnist_200.pt --defence rc --param ./params/rc_param.json --adv mnist_basic_shadow_0.3
