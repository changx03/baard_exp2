#!/bin/bash
# chmod +x ./bash/train_3_defence_baard_htru2.sh
# ./bash/train_3_defence_baard_htru2.sh | tee -a ./log/train_3_defence_baard_htru2.log

# htru2
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_apgd_0.2 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_apgd_0.2 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_apgd_0.05 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_apgd_0.05 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_0.01 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_0.01 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_100.0 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_cw2_100.0 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_deepfool_0.001 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_deepfool_0.001 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_deepfool_1e-06 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_deepfool_1e-06 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_fgsm_0.2 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_fgsm_0.2 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_fgsm_0.05 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_fgsm_0.05 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_boundary_0.3 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data htru2 --pretrained htru2_400.pt --adv htru2_basic32_boundary_0.3 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
