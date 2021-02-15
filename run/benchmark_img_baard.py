import argparse
import os
import sys

sys.path.append(os.getcwd())

from experiments.pytorch_attack_against_baard_img import pytorch_attack_against_baard


def run_mnist(i):
    data = 'mnist'
    model = 'dnn'
    params = []
    for j in range(1, 4):
        path = os.path.join('params', 'baard_mnist_{}.json'.format(j))
        params.append(path)

    eps = [0.063, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        for p in params:
            pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0.]
    att = 'boundary'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0.]
    att = 'deepfool'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0., 5., 10.]
    att = 'cw2'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [1.]
    att = 'line'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)


def run_cifar10_resnet(i):
    data = 'cifar10'
    model = 'resnet'
    params = []
    for j in range(1, 4):
        path = os.path.join('params', 'baard_cifar10_{}.json'.format(j))
        params.append(path)

    eps = [0.031, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        for p in params:
            pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0.]
    att = 'deepfool'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0., 5., 10.]
    att = 'cw2'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0., 1.]
    att = 'line'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)


def run_cifar10_vgg(i):
    data = 'cifar10'
    model = 'vgg'
    params = []
    for j in range(1, 4):
        path = os.path.join('params', 'baard_cifar10_{}.json'.format(j))
        params.append(path)

    eps = [0.031, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for att in attacks:
        for p in params:
            pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    att = 'apgd2'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0.]
    att = 'deepfool'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0., 5., 10.]
    att = 'cw2'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)

    eps = [0., 1.]
    att = 'line'
    for p in params:
        pytorch_attack_against_baard(data, model, att, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=True)


def run(n):
    for i in range(n):
        run_mnist(i)
        run_cifar10_resnet(i)
        run_cifar10_vgg(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_times', type=int, default=1)
    args = parser.parse_args()
    print('args:', args)
    run(args.n_times)

# # Testing
# if __name__ == '__main__':
#     pytorch_attack_against_baard('mnist', 'dnn', 'apgd', epsilons=[0.3], idx=0, baard_param='./params/baard_mnist_3.json', fresh_att=False, fresh_def=True)

# Example: Running from terminal
# nohup python3 ./run/benchmark_img_baard.py -n 2 > ./log/benchmark_img_baard.out 2> ./log/benchmark_img_baard.err & tail -f ./log/benchmark_img_baard.out
