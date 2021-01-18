#!/bin/bash
# chmod +x ./bash/train_3_defence_lid_banknote.sh
# ./bash/train_3_defence_lid_banknote.sh | tee -a ./log/train_3_defence_lid_banknote.log

# banknote
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.2 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_apgd_0.05 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_cw2_0.01 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_cw2_100.0 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_deepfool_0.001 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_deepfool_1e-06 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_fgsm_0.2 --defence lid --param lid_param.json
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --adv banknote_basic16_fgsm_0.05 --defence lid --param lid_param.json
