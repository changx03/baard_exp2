
#!/bin/bash
# chmod +x ./train_2_attack_boundary.sh

python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack boundary
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack boundary
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack boundary
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack boundary
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack boundary
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack boundary
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack boundary