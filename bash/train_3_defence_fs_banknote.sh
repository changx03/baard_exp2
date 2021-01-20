#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_banknote.sh
# ./bash/train_3_defence_fs_banknote.sh | tee -a ./log/train_3_defence_fs_banknote.log

# banknote
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.2 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.05 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_cw2_0.01 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_cw2_100.0 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_deepfool_0.001 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_deepfool_1e-06 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_fgsm_0.2 --defence fs --param ./params/fs_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_fgsm_0.05 --defence fs --param ./params/fs_param.json
