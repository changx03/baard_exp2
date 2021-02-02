#!/bin/bash
# chmod +x ./bash/train_2_attack_num.sh
# ./bash/train_2_attack_num.sh | tee -a ./log/train_2_attack_num.log

DATASETS=("banknote" "breastcancer" "htru2")

for MOD in "${MODELS[@]}"; do
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd --eps 0.05
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd --eps 0.2
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd --eps 0.4
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd --eps 1.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd2 --eps 0.4
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd2 --eps 1.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd2 --eps 2.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack apgd2 --eps 3.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack boundary --eps 0.3
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack cw2 --eps 0.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack cw2 --eps 5.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack cw2 --eps 10.0
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack deepfool --eps 1e-06
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack fgsm --eps 0.05
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack fgsm --eps 0.2
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack fgsm --eps 0.4
    python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_400.pt" --attack fgsm --eps 1.0
done
