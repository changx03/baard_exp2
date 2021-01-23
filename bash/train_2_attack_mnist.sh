#!/bin/bash
# chmod +x ./bash/train_2_attack_mnist.sh
# ./bash/train_2_attack_mnist.sh | tee -a ./log/train_2_attack_mnist.log

# mnist
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 0.063
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 0.3
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 0.6
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 1.0

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd1 --eps 0.063
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd1 --eps 0.3
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd1 --eps 0.6
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd1 --eps 1.0

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd2 --eps 0.063
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd2 --eps 0.3
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd2 --eps 0.6
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd2 --eps 1.0

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cw2 --eps 5
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cw2 --eps 10

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cwinf --eps 0
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cwinf --eps 5
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cwinf --eps 10

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack deepfool --eps 1e-6

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 0.063
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 0.3
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 0.6
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 1.0

# python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack shadow --batch_size 400
