#!/bin/bash
# chmod +x ./bash/train_3_defence_lid_htru2.sh
# ./bash/train_3_defence_lid_htru2.sh | tee ./log/train_3_defence_lid_htru2.log

# htru2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd_0.05
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd_0.2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd_0.4
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd1_0.05
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd1_0.2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd1_0.4
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd_1.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd1_1.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd2_0.05
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd2_0.2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd2_0.4
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_apgd2_1.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_boundary_0.3
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_cw2_0.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_cw2_10.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_cw2_5.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_cwinf_0.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_cwinf_10.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_cwinf_5.0
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_deepfool_1e-06
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_fgsm_0.05
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_fgsm_0.2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_fgsm_0.4
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --defence lid --param ./params/lid_param.json --adv htru2_basic32_fgsm_1.0