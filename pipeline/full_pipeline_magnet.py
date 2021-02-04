"""
This script does not run standalone. Run BAARD first. MagNet will use the same 
training and validation sets that are used by BAARD.
"""
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.util import acc_on_adv, get_correct_examples
from misc.util import set_seeds
from models.torch_util import predict_numpy, validate

from pipeline.run_attack import run_attack_untargeted
from pipeline.train_model import train_model
from pipeline.train_defence import train_magnet

PATH_DATA = 'data'
EPOCHS = 200


def run_full_pipeline_magnet(data,
                             model_name,
                             path,
                             seed,
                             json_param=os.path.join('params', 'magnet_param.json'),
                             att_name='apgd2',
                             eps=2.0):
    set_seeds(seed)

    print('args:', data, model_name, path, seed, json_param, att_name, eps)

    if not os.path.exists(path):
        print('Output folder does not exist. Create:', path)
        os.mkdir(path)

    # Get data
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

    adv = run_attack_untargeted(file_model, X, y, att_name=att_name, eps=eps, device=device)

    print('-------------------------------------------------------------------')
    print('Start testing adversarial examples...')
    pred = predict_numpy(model, adv, device)
    print('Acc on adv:', np.mean(pred == y))

    X_def_test = X[:1000]
    y_def_test = y[:1000]
    adv_def_test = adv[:1000]
    pred_adv_def_test = pred[:1000]

    X_def_val = X[1000:2000]
    # y_def_val = y[1000:2000]
    # adv_def_val = adv[1000:2000]
    # pred_adv_def_val = pred[1000:2000]

    # X_att_test = X[2000:4000]
    # y_att_test = y[2000:4000]
    # adv_att_test = adv[2000:4000]
    # pred_adv_att_test = pred[2000:4000]

    # X_surro_train = X[4000:]
    # y_surro_train = y[4000:]
    # adv_surro_train = adv[4000:]
    # pred_adv_surro_train = pred[4000:]

    print('-------------------------------------------------------------------')
    print('Start training MagNet...')
    # Run preprocessing
    tensor_X, tensor_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    X_train = tensor_X.cpu().detach().numpy()
    y_train = tensor_y.cpu().detach().numpy()

    with open(json_param) as j:
        param = json.load(j)

    time_start = time.time()
    detector = train_magnet(data, model_name, X_train, y_train, X_def_val, param, device, path, EPOCHS, model=model)
    time_elapsed = time.time() - time_start
    print('Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

    print('-------------------------------------------------------------------')
    print('Start testing MagNet...')

    time_start = time.time()
    adv_reformed_test, label_adv = detector.detect(adv_def_test, pred_adv_def_test)
    X_reformed_test, label_clean = detector.detect(X_def_test, y_def_test)
    time_elapsed = time.time() - time_start
    print('Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

    pred_adv_reformed = predict_numpy(model, adv_reformed_test, device)
    acc = acc_on_adv(pred_adv_reformed, y_def_test, label_adv)
    fpr = np.mean(label_clean)
    print('Acc_on_adv:', acc)
    print('FPR:', fpr)

    obj = {
        'X': X_def_test,
        'y': y_def_test,
        'adv': adv_def_test,
        'label_adv': label_adv,
        'label_clean': label_clean,
        'pred_adv': pred_adv_def_test,
        'X_reformed': X_reformed_test,
        'adv_reformed': adv_reformed_test,
        'pred_adv_reformed': pred_adv_reformed
    }
    file_detector_output = os.path.join(path, '{}_{}_{}_{}_magnet_output.pt'.format(data, model_name, att_name, eps))
    torch.save(obj, file_detector_output)
    print('Save to:', file_detector_output)

    print('DONE!')
    print('-------------------------------------------------------------------\n')


if __name__ == '__main__':
    seeds = [65558, 87742, 47709, 33474, 83328]
    for i in range(len(seeds)):
        path = 'result_{}'.format(str(i))
        run_full_pipeline_magnet('mnist', 'dnn', path, seed=seeds[i])
        run_full_pipeline_magnet('cifar10', 'resnet', path, seed=seeds[i])
