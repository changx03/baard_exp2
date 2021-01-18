#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_segment.sh
# ./bash/train_3_defence_fs_segment.sh | tee -a ./log/train_3_defence_fs_segment.log

# segment
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.2 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_apgd_0.05 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_0.01 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_cw2_100.0 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_0.001 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_deepfool_1e-06 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.2 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_fgsm_0.05 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data segment --pretrained segment_400.pt --adv segment_basic72_boundary_0.3 --defence fs --param fs_param.json
