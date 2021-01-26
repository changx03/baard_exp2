#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_breastcancer.sh
# ./bash/train_3_defence_fs_breastcancer.sh | tee ./log/train_3_defence_fs_breastcancer.log

# breastcancer
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd_0.05
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd_0.2
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd_0.4
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd_1.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd1_0.05
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd1_0.2
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd1_0.4
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd1_1.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd2_0.05
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd2_0.2
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd2_0.4
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_apgd2_1.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_boundary_0.3
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_cw2_0.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_cw2_10.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_cw2_5.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_cwinf_0.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_cwinf_10.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_cwinf_5.0
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_deepfool_1e-06
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_fgsm_0.05
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_fgsm_0.2
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_fgsm_0.4
python3 ./experiments/train_defences.py --data breastcancer --pretrained breastcancer_400.pt --defence fs --param ./params/fs_param_num.json --adv breastcancer_basic120_fgsm_1.0