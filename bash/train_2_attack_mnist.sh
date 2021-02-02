#!/bin/bash
# chmod +x ./bash/train_2_attack_mnist.sh
# ./bash/train_2_attack_mnist.sh | tee -a ./log/train_2_attack_mnist.log

# mnist
# python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 0.063

DATA="mnist"

python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd --eps 0.063
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd --eps 0.3
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd --eps 0.6
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd --eps 1.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd --eps 1.5
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd2 --eps 1.5
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd2 --eps 2.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd2 --eps 3.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack apgd2 --eps 5.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack cw2 --eps 0.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack cw2 --eps 5.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack cw2 --eps 10.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack deepfool --eps 1e-06
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack fgsm --eps 0.063
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack fgsm --eps 0.3
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack fgsm --eps 0.6
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack fgsm --eps 1.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack fgsm --eps 1.5
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack line --eps 0.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack line --eps 0.5
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack line --eps 1.0
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack watermark --eps 0.3
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack watermark --eps 0.6
python3 ./experiments/train_attacks.py --data $DATA --pretrained $DATA"_200.pt" --attack watermark --eps 1.0
