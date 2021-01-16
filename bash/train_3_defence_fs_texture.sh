#!/bin/bash
# chmod +x ./bash/train_3_defence_fs_texture.sh
# ./bash/train_3_defence_fs_texture.sh | tee -a ./log/train_3_defence_fs_texture.log

# texture
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_apgd_0.2 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_apgd_0.05 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_cw2_0.01 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_cw2_100.0 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_deepfool_0.001 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_deepfool_1e-06 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_fgsm_0.2 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_fgsm_0.05 --defence fs --param fs_param.json
python3 ./experiments/train_defences.py --data texture --pretrained texture_400.pt --adv texture_basic160_boundary_0.3--defence fs --param fs_param.json
