import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from art.attacks.evasion import AutoProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from defences.util import get_correct_examples
from experiments.util import set_seeds
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.torch_util import validate


class SurrogateModel(nn.Module):
    """This is the surrogate model for BAARD"""

    def __init__(self, in_channels=1, use_prob=True):
        super(SurrogateModel, self).__init__()
        self.use_prob = use_prob

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(9216, 200)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(200, 2)
        self.softmax = nn.Softmax(dim=1)

    def before_softmax(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.before_softmax(x)
        if self.use_prob:
            x = self.softmax(x)
        return x


def train_adv(data='mnist',
              model_name='basic',
              n_samples=2000,
              eps=2.,
              path_output='results',
              path_data='data',
              is_test=False,
              batch_size=128,
              device='cpu'):
    # Prepare data
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data == 'mnist':
        dataset_test = datasets.MNIST(path_data, train=False, download=True, transform=transforms)
    elif data == 'cifar10':
        dataset_test = datasets.CIFAR10(path_data, train=False, download=True, transform=transforms)
    else:
        raise NotImplementedError
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Load model
    if data == 'mnist':
        model = BaseModel(use_prob=False).to(device)
        n_features = (1, 28, 28)
        pretrained = 'mnist_200.pt'
    elif data == 'cifar10':
        n_features = (3, 32, 32)
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
            pretrained = 'cifar10_resnet_200.pt'
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
            pretrained = 'cifar10_vgg_200.pt'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    pretrained_path = os.path.join(path_output, pretrained)
    model.load_state_dict(torch.load(pretrained_path))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on test set: {:.4f}%'.format(acc_test * 100))

    tensor_test_X, tensor_test_y = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
    # Get samples from the tail
    if not is_test:
        # This is for training the surrogate model
        tensor_test_X = tensor_test_X[-n_samples:]
        tensor_test_y = tensor_test_y[-n_samples:]
    else:
        # This is for testing the surrogate model
        tensor_test_X = tensor_test_X[-n_samples - 2000:-2000]
        tensor_test_y = tensor_test_y[-n_samples - 2000:-2000]
    dataset_test = TensorDataset(tensor_test_X, tensor_test_y)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    _, acc_perfect = validate(model, loader_test, loss, device)
    print('Accuracy on {} filtered test set: {:.4f}%'.format(len(dataset_test), acc_perfect * 100))

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        input_shape=n_features,
        optimizer=optimizer,
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type='gpu')
    attack = AutoProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=0.1,
        max_iter=1000,
        batch_size=batch_size,
        targeted=False)

    X_benign = tensor_test_X.cpu().detach().numpy()
    y_true = tensor_test_y.cpu().detach().numpy()
    adv = attack.generate(x=X_benign)
    pred_adv = np.argmax(classifier.predict(adv), axis=1)
    acc_adv = np.mean(pred_adv == y_true)
    print("Accuracy on adversarial examples: {:.4f}%".format(acc_adv * 100))

    if not is_test:
        output_file = '{}_{}_baard_train_surro_train_eps{}_size{}.pt'.format(data, model_name, eps, n_samples)
    else:
        output_file = '{}_{}_baard_train_surro_test_eps{}_size{}.pt'.format(data, model_name, eps, n_samples)
    file_path = os.path.join(path_output, output_file)
    output = {
        'X': X_benign,
        'adv': adv,
        'y': y_true}
    torch.save(output, file_path)
    print('Save to:', file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, required=True, choices=['basic', 'resnet', 'vgg'])
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--eps', type=float, default=2.)
    parser.add_argument('--test', type=int, default=0, choices=[0, 1])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    time_start = time.time()
    train_adv(args.data, args.model, args.n_samples, args.eps, args.output_path, args.data_path, args.test, args.batch_size, device)
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
    print()


if __name__ == '__main__':
    main()
