import argparse
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from defences.util import dataset2tensor
from models.cifar10 import Resnet, Vgg
from models.torch_util import print_acc_per_label, train, validate
from experiments.util import set_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--model', type=str,
                        default='resnet18', choices=['resnet', 'vgg'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pretrained', type=str, nargs='?')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()

    set_seeds(args.random_state)

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Fetch dataset
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.ToTensor()])
    transform_test = tv.transforms.Compose([tv.transforms.ToTensor()])
    dataset_train = datasets.CIFAR10(
        args.data_path, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10(
        args.data_path, train=False, download=True, transform=transform_test)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    print('Train set: {}, Test set: {}'.format(
        len(dataset_train), len(dataset_test)))

    # Prepare model
    if args.model == 'resnet':
        model = Resnet().to(device)
    elif args.model == 'vgg':
        model = Vgg().to(device)
    else:
        raise ValueError('Does not support {}'.format(args.model))

    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Load pre-trained model
    if args.pretrained is not None:
        pretrained_path = os.path.join(args.output_path, args.pretrained)
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

    # Train model
    since = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        tr_loss, tr_acc = train(model, dataloader_train,
                                loss, optimizer, device)
        va_loss, va_acc = validate(model, dataloader_test, loss, device)
        scheduler.step()

        time_elapsed = time.time() - start
        print(('{:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, ' +
               'Test Loss: {:.4f} Acc: {:.4f}%').format(
            epoch +
            1, args.epochs, str(datetime.timedelta(seconds=time_elapsed)),
            tr_loss, tr_acc * 100.,
            va_loss, va_acc * 100.))

    time_elapsed = time.time() - since
    print('Total run time: {:.0f}m {:.1f}s'.format(
        time_elapsed // 60,
        time_elapsed % 60))

    # Save model
    file_name = os.path.join(
        args.output_path, 'cifar10_{:s}_{:d}.pt'.format(args.model, args.epochs))
    print('Output file name: {}'.format(file_name))
    torch.save(model.state_dict(), file_name)

    # Test accuracy per class:
    print('Training set:')
    X, y = dataset2tensor(dataset_train)
    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    print_acc_per_label(model, X, y, device)

    print('Test set:')
    X, y = dataset2tensor(dataset_test)
    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    print_acc_per_label(model, X, y, device)


if __name__ == '__main__':
    main()
