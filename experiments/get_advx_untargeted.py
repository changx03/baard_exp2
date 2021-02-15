import os
import sys

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import json
import numpy as np
import torch
from art.attacks.evasion import (AutoProjectedGradientDescent,
                                 BasicIterativeMethod, BoundaryAttack,
                                 CarliniLInfMethod, DeepFool,
                                 FastGradientMethod)
from art.estimators.classification import PyTorchClassifier
from attacks.carlini import CarliniWagnerAttackL2
from attacks.line_attack import LineAttack

with open('metadata.json') as data_json:
    METADATA = json.load(data_json)


def get_advx_untargeted(model, data_name, att_name, eps, device, X, y=None, batch_size=128):
    n_classes = METADATA['data'][data_name]['n_classes']
    n_features = METADATA['data'][data_name]['n_features']
    input_shape = (n_features,) if isinstance(n_features, int) else tuple(n_features)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    device_type = 'cpu' if device == 'cpu' else 'gpu'

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        input_shape=input_shape,
        optimizer=optimizer,
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type=device_type)

    if att_name == 'apgd':
        eps_step = eps / 10.0 if eps <= 0.1 else 0.1
        attack = AutoProjectedGradientDescent(
            estimator=classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=1000,
            batch_size=batch_size,
            targeted=False,
            verbose=False)
    elif att_name == 'apgd2':
        attack = AutoProjectedGradientDescent(
            estimator=classifier,
            norm=2,
            eps=eps,
            eps_step=0.1,
            max_iter=1000,
            batch_size=batch_size,
            targeted=False,
            verbose=False)
    elif att_name == 'bim':
        eps_step = eps / 10.0
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=1000,
            batch_size=batch_size,
            targeted=False)
    elif att_name == 'boundary':
        attack = BoundaryAttack(
            estimator=classifier,
            max_iter=1000,
            sample_size=batch_size,
            targeted=False,
            verbose=False)
    elif att_name == 'cw2':
        attack = CarliniWagnerAttackL2(
            model=classifier._model._model,
            n_classes=n_classes,
            confidence=eps,
            check_prob=False,
            batch_size=batch_size,
            targeted=False,
            verbose=False)
    elif att_name == 'cwinf':
        attack = CarliniLInfMethod(
            classifier=classifier,
            confidence=eps,
            max_iter=1000,
            batch_size=batch_size,
            targeted=False,
            verbose=False)
    elif att_name == 'deepfool':
        attack = DeepFool(
            classifier=classifier,
            # epsilon=eps,
            batch_size=batch_size,
            verbose=False)
    elif att_name == 'fgsm':
        attack = FastGradientMethod(
            estimator=classifier,
            eps=eps,
            batch_size=batch_size)
    elif att_name == 'line':
        if data_name == 'mnist':
            color = eps
        elif data_name == 'cifar10':
            color = (eps, eps, eps)
        else:
            raise NotImplementedError
        attack = LineAttack(color=color, thickness=1)
    else:
        raise NotImplementedError

    adv = attack.generate(x=X)
    return adv
