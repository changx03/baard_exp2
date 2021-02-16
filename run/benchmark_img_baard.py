import argparse
import os
import sys

sys.path.append(os.getcwd())

from experiments.pytorch_attack_against_baard_img import \
    pytorch_attack_against_baard_img


def run_mnist(i):
    data = 'mnist'
    model = 'dnn'

    eps = [0.063, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0.]
    att = 'boundary'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0.]
    att = 'deepfool'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0., 5., 10.]
    att = 'cw2'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [1.]
    att = 'line'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)


def run_cifar10_resnet(i):
    data = 'cifar10'
    model = 'resnet'

    eps = [0.031, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0.]
    att = 'deepfool'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0., 5., 10.]
    att = 'cw2'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0., 1.]
    att = 'line'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)


def run_cifar10_vgg(i):
    data = 'cifar10'
    model = 'vgg'

    eps = [0.031, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0.]
    att = 'deepfool'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0., 5., 10.]
    att = 'cw2'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)

    eps = [0., 1.]
    att = 'line'
    pytorch_attack_against_baard_img(data, model, att, epsilons=eps, idx=i)


def run(i):
    run_mnist(i)
    run_cifar10_resnet(i)
    run_cifar10_vgg(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)
    run(args.idx)


# Example: Running from terminal
# nohup python3 ./run/benchmark_img_baard.py -i 0 > ./log/benchmark_img_baard_0.out &
