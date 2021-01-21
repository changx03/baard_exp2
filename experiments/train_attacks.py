import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.datasets as datasets
from art.attacks.evasion import (AutoProjectedGradientDescent, BasicIterativeMethod,
                                 BoundaryAttack, DeepFool,
                                 FastGradientMethod, SaliencyMapMethod)
from art.estimators.classification import PyTorchClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Adding the parent directory.
sys.path.append(os.getcwd())
from attacks.carlini import CarliniWagnerAttackL2
from models.mnist import BaseModel
from models.cifar10 import Resnet, Vgg
from models.numeric import NumericModel
from experiments.train_pt import validate
from defences.util import get_shape, get_correct_examples


def load_csv(file_path):
    """Load a pre-processed CSV file."""
    df = pd.read_csv(file_path, sep=',')
    y = df['Class'].to_numpy().astype(np.long)
    X = df.drop(['Class'], axis=1).to_numpy().astype(np.float32)
    return X, y


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--attack', type=str, required=True, choices=data_params['attacks'])
    parser.add_argument('--eps', type=float, default=0.3)
    # NOTE: In CW_L2 attack, eps is the upper bound of c.
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--random_state', type=int, default=int(2**12))
    args = parser.parse_args()

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)
    print('Running attack: {}'.format(args.attack))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Prepare data
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    if args.data in ['banknote', 'htru2', 'segment', 'texture', 'yeast']:
        data_path = os.path.join(args.data_path, data_params['data'][args.data]['file_name'])
        print('Read file:', data_path)
        X, y = load_csv(data_path)

        # The label 10 is very strange.
        if args.data == 'texture':
            idx_not10 = np.where(y != 10)[0]
            X = X[idx_not10]
            y = y[idx_not10]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_params['data'][args.data]['n_test'],
            random_state=args.random_state)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        dataset_train = TensorDataset(
            torch.from_numpy(X_train).type(torch.float32),
            torch.from_numpy(y_train).type(torch.long))
        dataset_test = TensorDataset(
            torch.from_numpy(X_test).type(torch.float32),
            torch.from_numpy(y_test).type(torch.long))
    elif args.data == 'mnist':
        dataset_train = datasets.MNIST(
            args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.MNIST(
            args.data_path, train=False, download=True, transform=transforms)
    elif args.data == 'cifar10':
        dataset_train = datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.CIFAR10(
            args.data_path, train=False, download=True, transform=transforms)
    else:
        raise ValueError('{} is not supported.'.format(args.data))

    dataloader_train = DataLoader(
        dataset_train, args.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, args.batch_size, shuffle=False)

    shape_train = get_shape(dataloader_train.dataset)
    shape_test = get_shape(dataloader_test.dataset)
    print('Train set:', shape_train)
    print('Test set:', shape_test)

    # Load model
    use_prob = args.attack not in ['apgd', 'cw2']
    print('Attack:', args.attack)
    print('Using softmax layer:', use_prob)
    if args.data == 'mnist':
        model = BaseModel(use_prob=use_prob).to(device)
        model_name = 'basic'
    elif args.data == 'cifar10':
        model_name = args.pretrained.split('_')[1]
        if model_name == 'resnet':
            model = Resnet(use_prob=use_prob).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=use_prob).to(device)
        else:
            raise ValueError('Unknown model: {}'.format(model_name))
    else:
        n_features = data_params['data'][args.data]['n_features']
        n_classes = data_params['data'][args.data]['n_classes']
        model = NumericModel(
            n_features,
            n_hidden=n_features * 4,
            n_classes=n_classes,
            use_prob=use_prob).to(device)
        model_name = 'basic' + str(n_features * 4)

    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, dataloader_train, loss, device)
    _, acc_test = validate(model, dataloader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train*100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test*100))

    # Create a subset which only contains recognisable samples.
    tensor_test_X, tensor_test_y = get_correct_examples(
        model, dataset_test, device=device, return_tensor=True)
    dataset_perfect = TensorDataset(tensor_test_X, tensor_test_y)
    loader_perfect = DataLoader(dataset_perfect, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_perfect, loss, device)
    print('Accuracy on {} filtered test examples: {:.4f}%'.format(
        len(dataset_perfect), acc_perfect*100))

    # Generate adversarial examples
    n_features = data_params['data'][args.data]['n_features']
    n_classes = data_params['data'][args.data]['n_classes']
    if isinstance(n_features, int):
        n_features = (n_features,)
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        input_shape=n_features,
        optimizer=optimizer,
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type='gpu')
    if args.attack == 'apgd':
        eps_step = args.eps / 10.0 if args.eps <= 0.1 else args.eps / 4.0
        max_iter = 1000 if args.eps <= 0.1 else 100
        attack = AutoProjectedGradientDescent(
            estimator=classifier,
            eps=args.eps,
            eps_step=eps_step,
            max_iter=max_iter,
            batch_size=args.batch_size)
    elif args.attack == 'bim':
        eps_step = args.eps / 10.0
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=args.eps,
            eps_step=eps_step,
            max_iter=1000)
    elif args.attack == 'boundary':
        attack = BoundaryAttack(
            estimator=classifier,
            targeted=False,
            max_iter=1000)
    elif args.attack == 'cw2':
        attack = CarliniWagnerAttackL2(
            model=model,
            n_classes=n_classes,
            targeted=False,
            lr=1e-2,
            binary_search_steps=9,
            max_iter=1000,
            confidence=0.0,
            initial_const=1e-3,
            c_range=(0, args.eps),
            abort_early=True,
            batch_size=args.batch_size,
            clip_values=(0.0, 1.0),
            check_prob=True,
            verbose=True)
    elif args.attack == 'deepfool':
        attack = DeepFool(
            classifier=classifier, 
            epsilon=args.eps,
            batch_size=args.batch_size)
    elif args.attack == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, eps=args.eps)
    elif args.attack == 'jsma':
        attack = SaliencyMapMethod(
            classifier=classifier,
            gamma=args.eps,
            batch_size=args.batch_size)
    else:
        raise NotImplementedError

    if len(dataset_perfect) > args.n_samples:
        n = args.n_samples
    else:
        n = len(dataset_perfect)
    # idx_shuffle = torch.randperm(len(tensor_test_X))[:n]
    # X_benign = tensor_test_X[idx_shuffle].cpu().detach().numpy()
    # y = tensor_test_y[idx_shuffle].cpu().detach().numpy()
    X_benign = tensor_test_X[:n].cpu().detach().numpy()
    y = tensor_test_y[:n].cpu().detach().numpy()


    print('Creating {} adversarial examples with Epsilon={}'.format(n,
                                                                    args.eps))
    time_start = time.time()
    adv = attack.generate(x=X_benign)
    time_elapsed = time.time() - time_start
    print('Total time spend: {}'.format(
        str(datetime.timedelta(seconds=time_elapsed))))

    pred_benign = np.argmax(classifier.predict(X_benign), axis=1)
    acc_benign = np.sum(pred_benign == y) / n
    pred_adv = np.argmax(classifier.predict(adv), axis=1)
    acc_adv = np.sum(pred_adv == y) / n
    print("Accuracy on benign samples: {:.4f}%".format(acc_benign*100))
    print("Accuracy on adversarial examples: {:.4f}%".format(acc_adv*100))
    print()

    # Save results
    path_x = os.path.join(
        args.output_path, '{}_{}_{}_{}_x.npy'.format(
            args.data, model_name, args.attack, str(args.eps)))
    path_y = os.path.join(
        args.output_path, '{}_{}_{}_{}_y.npy'.format(
            args.data, model_name, args.attack, str(args.eps)))
    path_adv = os.path.join(
        args.output_path, '{}_{}_{}_{}_adv.npy'.format(
            args.data, model_name, args.attack, str(args.eps)))
    np.save(path_x, X_benign)
    np.save(path_y, y)
    np.save(path_adv, adv)


if __name__ == '__main__':
    main()
