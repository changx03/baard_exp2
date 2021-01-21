#!/bin/bash
# chmod +x ./bash/train_3_defence_baard_segment.sh
# ./bash/train_3_defence_baard_segment.sh | tee -a ./log/train_3_defence_baard_segment.log

# segment
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.2 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.2 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.05 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.05 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_0.01 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_0.01 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_100.0 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_100.0 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_0.001 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_0.001 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_1e-06 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_1e-06 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.2 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.2 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.05 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.05 --defence baard --param ./params/baard_param_3s.json --suffix 3stage
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_boundary_0.3 --defence baard --param ./params/baard_param_2s.json --suffix 2stage
