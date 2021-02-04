import datetime
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import (AutoProjectedGradientDescent, DeepFool,
                                 FastGradientMethod)
from art.estimators.classification import PyTorchClassifier
from attacks.carlini import CarliniWagnerAttackL2

# Adding the parent directory.
sys.path.append(os.getcwd())
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel

BATCH_SIZE = 128


def run_attack_untargeted(file_model, X, y, att_name, eps, device):
    path = file_model.split('/')[0]
    file_str = file_model.split('/')[-1]
    name_arr = file_str.split('_')
    data = name_arr[0]
    model_name = name_arr[1]
    file_data = os.path.join(path, '{}_{}_{}_{}.pt'.format(data, model_name, att_name, eps))

    if os.path.exists(file_data):
        print('Found existing file:', file_data)
        obj = torch.load(file_data)
        return obj['adv']

    if data == 'mnist':
        n_features = (1, 28, 28)
        n_classes = 10
        model = BaseModel(use_prob=False).to(device)
    elif data == 'cifar10':
        n_features = (3, 32, 32)
        n_classes = 10
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(file_model))
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        input_shape=n_features,
        optimizer=optimizer,
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type='gpu')

    if att_name == 'apgd':
        eps_step = eps / 4. if eps <= 0.2 else 0.1
        attack = AutoProjectedGradientDescent(
            estimator=classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=1000,
            batch_size=BATCH_SIZE,
            targeted=False)
    elif att_name == 'apgd2':
        attack = AutoProjectedGradientDescent(
            estimator=classifier,
            norm=2,
            eps=eps,
            eps_step=0.1,
            max_iter=1000,
            batch_size=BATCH_SIZE,
            targeted=False)
    elif att_name == 'cw2':
        # Do not increase the batch_size
        attack = CarliniWagnerAttackL2(
            model=model,
            n_classes=n_classes,
            confidence=eps,
            verbose=True,
            check_prob=False,
            batch_size=32,
            targeted=False)
    elif att_name == 'deepfool':
        # Do not adjust Epsilon
        attack = DeepFool(
            classifier=classifier,
            batch_size=BATCH_SIZE)
    elif att_name == 'fgsm':
        attack = FastGradientMethod(
            estimator=classifier,
            eps=eps,
            batch_size=BATCH_SIZE)
    else:
        raise NotImplementedError

    time_start = time.time()
    adv = attack.generate(x=X)
    time_elapsed = time.time() - time_start
    print('Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

    obj = {
        'X': X,
        'y': y,
        'adv': adv
    }
    torch.save(obj, file_data)
    print('Save data to:', file_data)

    return adv
