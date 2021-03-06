import datetime
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.torch_util import train, validate


def train_model(data, model_name, dataset_train, dataset_test, epochs, device, file_model):
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)
    print('Train set: {}, Test set: {}'.format(len(dataset_train), len(dataset_test)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    if data == 'mnist':
        model = BaseModel(use_prob=True).to(device)
    elif data == 'cifar10' and model_name == 'resnet':
        model = Resnet(use_prob=True).to(device)
    elif data == 'cifar10' and model_name == 'vgg':
        model = Vgg(use_prob=True).to(device)
    else:
        raise NotImplementedError

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if not os.path.exists(file_model):
        since = time.time()
        for e in range(epochs):
            start = time.time()
            tr_loss, tr_acc = train(model, dataloader_train, loss, optimizer, device)
            va_loss, va_acc = validate(model, dataloader_test, loss, device)
            scheduler.step()
            time_elapsed = time.time() - start
            print(('{:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, Test Loss: {:.4f} Acc: {:.4f}%').format(
                e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100., va_loss, va_acc * 100.))

        time_elapsed = time.time() - since
        print('Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

        torch.save(model.state_dict(), file_model)
        print('Save base model to:', file_model)
    else:
        print('Found existing file:', file_model)
        model.load_state_dict(torch.load(file_model, map_location=device))
    return model
