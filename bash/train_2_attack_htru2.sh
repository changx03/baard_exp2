#!/bin/bash
# chmod +x ./bash/train_2_attack_htru2.sh
# ./bash/train_2_attack_htru2.sh | tee -a ./log/train_2_attack_htru2.log

# htru2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd --eps 0.05
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd --eps 0.2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd --eps 0.4
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd --eps 1.0

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd1 --eps 0.05
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd1 --eps 0.2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd1 --eps 0.4
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd1 --eps 1.0

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd2 --eps 0.05
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd2 --eps 0.2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd2 --eps 0.4
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd2 --eps 1.0

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cw2 --eps 100

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cwinf --eps 0
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cwinf --eps 10
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cwinf --eps 100

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack deepfool --eps 1e-6

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack fgsm --eps 0.05
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack fgsm --eps 0.4
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack fgsm --eps 1.0

python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack shadow --batch_size 400
