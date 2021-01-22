#!/bin/bash
# chmod +x ./bash/train_2_attack.sh
# ./bash/train_2_attack.sh | tee -a ./log/train_attack.log

# mnist
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 0.063
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack fgsm --eps 0.3
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 0.063 --batch_size 64
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack apgd --eps 0.3 --batch_size 64
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data mnist --pretrained mnist_200.pt --attack deepfool --eps 1e-3 --batch_size 64

# cifar10 resnet
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 0.031 --batch_size 64
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack apgd --eps 0.2 --batch_size 64
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_resnet_200.pt --attack deepfool --eps 1e-3 --batch_size 64

# cifar10 vgg
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 0.031
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 0.031 --batch_size 64
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack apgd --eps 0.2 --batch_size 64
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data cifar10 --pretrained cifar10_vgg_200.pt --attack deepfool --eps 1e-3 --batch_size 64

# banknote
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack fgsm --eps 0.05
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack apgd --eps 0.05 --batch_size 64
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack apgd --eps 0.2 --batch_size 64
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data banknote --pretrained banknote_400.pt --attack deepfool --eps 1e-3 --batch_size 64

# htru2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack fgsm --eps 0.05
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd --eps 0.05 --batch_size 64
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack apgd --eps 0.2 --batch_size 64
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data htru2 --pretrained htru2_400.pt --attack deepfool --eps 1e-3 --batch_size 64

# segment
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack fgsm --eps 0.05
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack apgd --eps 0.05 --batch_size 64
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack apgd --eps 0.2 --batch_size 64
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data segment --pretrained segment_400.pt --attack deepfool --eps 1e-3 --batch_size 64

# texture
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack fgsm --eps 0.05
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack fgsm --eps 0.2
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack apgd --eps 0.05 --batch_size 64
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack apgd --eps 0.2 --batch_size 64
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack cw2 --eps 0
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack cw2 --eps 10
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack cw2 --eps 100
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack deepfool --eps 1e-6 --batch_size 64
python3 ./experiments/train_attacks.py --data texture --pretrained texture_400.pt --attack deepfool --eps 1e-3 --batch_size 64
