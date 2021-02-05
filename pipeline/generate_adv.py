import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.util import get_correct_examples
from misc.util import set_seeds
from models.torch_util import predict_numpy, validate

from pipeline.run_attack import ATTACKS, run_attack_untargeted
from pipeline.train_model import train_model

PATH_DATA = 'data'
EPOCHS = 200


def run_generate_adv(data,
                     model_name,
                     path,
                     seed,
                     att_name,
                     eps):
    set_seeds(seed)

    # Line attack takes no hyperparameter
    if att_name == 'line':
        eps = [1]
    print('args:', data, model_name, path, seed, att_name, eps)

    if not os.path.exists(path):
        print('Output folder does not exist. Create:', path)
        os.mkdir(path)

    # Get data
    n_classes = 10
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data == 'mnist':
        dataset_train = datasets.MNIST(PATH_DATA, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(PATH_DATA, train=False, download=True, transform=transform)
    elif data == 'cifar10':
        transform_train = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor()])
        dataset_train = datasets.CIFAR10(PATH_DATA, train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10(PATH_DATA, train=False, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(data))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    file_model = os.path.join(path, '{}_{}_model.pt'.format(data, model_name))
    print('Start training {} model on {}...'.format(model_name, data))
    model = train_model(data, model_name, dataset_train, dataset_test, EPOCHS, device, file_model)

    # Split data
    tensor_X, tensor_y = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
    dataset = TensorDataset(tensor_X, tensor_y)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc_perfect = validate(model, loader, nn.CrossEntropyLoss(), device)
    print('Accuracy on {} filtered test set: {:.2f}%'.format(tensor_y.size(0), acc_perfect * 100))
    # Split rules:
    # 1. Benchmark_defence_test: 1000 (def_test)
    # 2. Benchmark_defence_val:  1000 (def_val)
    # 3. Test white-box attack:  2000 (att_test)
    # 5. Train surrogate model:  2000 (sur_train)
    #    -----------------Total: 6000
    idx_shuffle = np.random.permutation(tensor_X.size(0))[:6000]
    X = tensor_X[idx_shuffle].cpu().detach().numpy()
    y = tensor_y[idx_shuffle].cpu().detach().numpy()

    print('-------------------------------------------------------------------')
    print('Start generating {} adversarial examples...'.format(len(idx_shuffle)))

    advs = []
    for e in eps:
        adv, X, y = run_attack_untargeted(file_model, X, y, att_name=att_name, eps=e, device=device)
        advs.append(adv)
    advs = np.array(advs, dtype=np.float32)

    print('-------------------------------------------------------------------')
    print('Start testing adversarial examples...')
    for i, e in enumerate(eps):
        adv = advs[i]
        pred = predict_numpy(model, adv, device)
        print('Attack: {} Eps={} Acc on adv: {:.4f}'.format(att_name, e, np.mean(pred == y)))


if __name__ == '__main__':
    path_cur = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_cur, 'seeds.json')) as j:
        json_obj = json.load(j)
        seeds = json_obj['seeds']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='dnn', choices=['dnn', 'resnet', 'vgg'])
    parser.add_argument('--attack', type=str, default='apgd2', choices=ATTACKS)
    parser.add_argument('--eps', type=float, nargs='+', default=[2.0])
    parser.add_argument('--idx', type=int, default=0, choices=list(range(len(seeds))))
    args = parser.parse_args()
    print(args)

    idx = args.idx
    run_generate_adv(
        data=args.data,
        model_name=args.model,
        path='result_{}'.format(str(idx)),
        seed=seeds[idx],
        att_name=args.attack,
        eps=args.eps)
