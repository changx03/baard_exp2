#!/bin/bash

EPOCHS=400

MNIST_TRAIN="mnist_basic_baard_surro_train_eps2.0_size2000.pt"
MNIST_TEST="mnist_basic_baard_surro_test_eps2.0_size1000.pt"
python3 ./experiments/train_baard_surrogate.py --data mnist --model basic --train $MNIST_TRAIN --test $MNIST_TEST --epochs $EPOCHS

CIFAR10_RESNET_TRAIN="cifar10_resnet_baard_surro_train_eps2.0_size2000.pt"
CIFAR10_RESNET_TEST="cifar10_resnet_baard_surro_test_eps2.0_size1000.pt"
python3 ./experiments/train_baard_surrogate.py --data cifar10 --model resnet --train $CIFAR10_RESNET_TRAIN --test $CIFAR10_RESNET_TEST --epochs $EPOCHS

CIFAR10_VGG_TRAIN="cifar10_vgg_baard_surro_train_eps2.0_size2000.pt"
CIFAR10_VGG_TEST="cifar10_vgg_baard_surro_test_eps2.0_size1000.pt"
python3 ./experiments/train_baard_surrogate.py --data cifar10 --model vgg --train $CIFAR10_VGG_TRAIN --test $CIFAR10_VGG_TEST --epochs $EPOCHS
