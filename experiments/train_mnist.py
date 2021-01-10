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

# Adding the parent directory.
sys.path.append(os.getcwd())
from models.mnist import BaseModel

def train(model, loader, loss, optimizer, device):
    model.train()
    total_loss = .0
    corrects = .0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()
        outputs = model(x)
        l = loss(outputs, y)
        l.backward()
        optimizer.step()

        # for display
        total_loss += l.item() * batch_size
        preds = outputs.max(1, keepdim=True)[1]
        corrects += preds.eq(y.view_as(preds)).sum().item()

    n = len(loader.dataset)
    total_loss = total_loss / n
    accuracy = corrects / n
    return total_loss, accuracy


def validate(model, loader, loss, device):
    model.eval()
    total_loss = .0
    corrects = .0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)
            outputs = model(x)
            l = loss(outputs, y)
            total_loss += l.item() * batch_size
            preds = outputs.max(1, keepdim=True)[1]
            corrects += preds.eq(y.view_as(preds)).sum().item()

    n = len(loader.dataset)
    total_loss = total_loss / n
    accuracy = corrects / n
    return total_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_path', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Fetch dataset
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])
    dataset_train = datasets.MNIST(
        args.data_path, train=True, download=True, transform=transforms)
    dataset_test = datasets.MNIST(
        args.data_path, train=False, download=True, transform=transforms)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True)

    print('Train set: {}, Test set: {}'.format(
        len(dataset_train), len(dataset_test)))

    # Prepare model
    model = BaseModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    # Train model
    since = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        tr_loss, tr_acc = train(model, dataloader_train,
                                loss, optimizer, device)
        va_loss, va_acc = validate(model, dataloader_test, loss, device)

        time_elapsed = time.time() - start
        print(('{:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, ' +
               'Test Loss: {:.4f} Acc: {:.4f}%').format(
            epoch +
            1, args.epochs, str(datetime.timedelta(seconds=time_elapsed)),
            tr_loss, tr_acc*100.,
            va_loss, va_acc*100.))

    time_elapsed = time.time() - since
    print('Total run time: {:.0f}m {:.1f}s'.format(
        time_elapsed // 60,
        time_elapsed % 60))

    # Save model
    file_name = os.path.join(
        args.model_path, 'mnist_{}.pt'.format(args.epochs))
    print('Output file name: {}'.format(file_name))
    torch.save(model.state_dict(), file_name)


if __name__ == '__main__':
    main()
