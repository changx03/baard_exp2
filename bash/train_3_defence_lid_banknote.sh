#!/bin/bash
# chmod +x ./bash/train_3_defence_lid_banknote.sh
# ./bash/train_3_defence_lid_banknote.sh | tee ./log/train_3_defence_lid_banknote.log

# banknote
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd_0.05
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd_0.2
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd_0.4
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd1_0.05
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd1_0.2
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd1_0.4
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd_1.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd1_1.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd2_0.05
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd2_0.2
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd2_0.4
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_apgd2_1.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_boundary_0.3
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_cw2_0.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_cw2_10.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_cw2_5.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_cwinf_0.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_cwinf_10.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_cwinf_5.0
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_deepfool_1e-06
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_fgsm_0.05
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_fgsm_0.2
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_fgsm_0.4
python3 ./experiments/train_defences.py --data banknote --pretrained banknote_400.pt --defence lid --param ./params/lid_param.json --adv banknote_basic16_fgsm_1.0