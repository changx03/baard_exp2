import datetime
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import train, validate

with open('metadata.json') as data_json:
    METADATA = json.load(data_json)


def train_model(data_name,
                model_name,
                dataset_train,
                dataset_test,
                device,
                file_model,
                epochs=10,
                batch_size=128):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    print('[CLASSIFIER] Train set: {}, Test set: {}'.format(len(dataset_train), len(dataset_test)))

    if data_name == 'mnist':
        model = BaseModel(use_prob=False).to(device)
        model_temp = BaseModel(use_prob=False).to(device)
    elif data_name == 'cifar10':
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
            model = Resnet(use_prob=False).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
        else:
            raise NotImplementedError
    else:
        n_features = METADATA['data'][data_name]['n_features']
        n_hidden = n_features * 4
        n_classes = METADATA['data'][data_name]['n_classes']
        model = NumericModel(n_features=n_features, n_hidden=n_hidden, n_classes=n_classes, use_prob=False).to(device)

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
            print(('[CLASSIFIER] {:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.4f}%, Test Loss: {:.4f} Acc: {:.4f}%').format(
                e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100., va_loss, va_acc * 100.))

        time_elapsed = time.time() - since
        print('[CLASSIFIER] Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

        torch.save(model.state_dict(), file_model)
        print('[CLASSIFIER] Save base model to:', file_model)
    else:
        print('[CLASSIFIER] Found existing file:', file_model)
        model.load_state_dict(torch.load(file_model, map_location=device))
    return model
