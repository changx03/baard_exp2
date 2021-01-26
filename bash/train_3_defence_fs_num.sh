#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_num.sh
# ./bash/train_3_defence_fs_num.sh | tee ./log/train_3_defence_fs_num.log

# This file trains the squeezers
python3 ./experiments/train_fs_num.py --data banknote --pretrained banknote_400.pt --param ./params/fs_param_num.json
python3 ./experiments/train_fs_num.py --data breastcancer --pretrained breastcancer_400.pt --param ./params/fs_param_num.json
python3 ./experiments/train_fs_num.py --data htru2 --pretrained htru2_400.pt --param ./params/fs_param_num.json
