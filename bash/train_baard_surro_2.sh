#!/bin/bash

python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file mnist_basic_baard_surro_train_eps2.0_size2000.pt
python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file mnist_basic_baard_surro_test_eps2.0_size1000.pt
python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file cifar10_resnet_baard_surro_train_eps2.0_size2000.pt
python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file cifar10_resnet_baard_surro_test_eps2.0_size1000.pt
python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file cifar10_vgg_baard_surro_train_eps2.0_size2000.pt
python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file cifar10_vgg_baard_surro_test_eps2.0_size1000.pt
