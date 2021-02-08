import argparse
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

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from defences.util import acc_on_adv, get_correct_examples
from misc.util import set_seeds
from models.torch_util import predict_numpy, validate

from pipeline.preprocess_baard import preprocess_baard
from pipeline.run_attack import ATTACKS, run_attack_untargeted
from pipeline.train_model import train_model
from pipeline.train_surrogate import get_pretrained_surrogate, train_surrogate

PATH_DATA = 'data'
EPOCHS = 200


def run_full_pipeline_baard(data,
                            model_name,
                            path,
                            seed,
                            json_param,
                            att_name,
                            eps):
    set_seeds(seed)

    # Line attack takes no hyperparameter
    if att_name == 'line':
        eps = 1
    print('args:', data, model_name, path, seed, json_param, att_name, eps)

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

    adv, X, y = run_attack_untargeted(file_model, X, y, att_name=att_name, eps=eps, device=device)

    print('-------------------------------------------------------------------')
    print('Start testing adversarial examples...')
    pred = predict_numpy(model, adv, device)
    print('Acc on adv:', np.mean(pred == y))

    X_def_test = X[:1000]
    y_def_test = y[:1000]
    adv_def_test = adv[:1000]
    pred_adv_def_test = pred[:1000]

    X_def_val = X[1000:2000]
    y_def_val = y[1000:2000]
    adv_def_val = adv[1000:2000]  # Unused by BAARD
    pred_adv_def_val = pred[1000:2000]  # Unused by BAARD

    X_att_test = X[2000:4000]
    y_att_test = y[2000:4000]
    adv_att_test = adv[2000:4000]
    pred_adv_att_test = pred[2000:4000]

    X_surro_train = X[4000:]
    y_surro_train = y[4000:]
    adv_surro_train = adv[4000:]

    # concatenate the adversarial examples computed for different epsilon
    if data == 'mnist':
        eps_1 = 1
        eps_2 = 5
        eps_3 = 8
        eps_4 = 3
    elif data == "cifar10":
        eps_1 = 0.05
        eps_2 = 0.1
        eps_3 = 0.5
        eps_4 = 1
    else:
        raise ValueError("dataset idx unknown")

    print('-------------------------------------------------------------------')
    print('Start training BAARD...')
    # Run preprocessing
    file_baard_train = os.path.join(path, '{}_{}_baard_s1_train_data.pt'.format(data, model_name))
    if os.path.exists(file_baard_train):
        print('Found existing BAARD preprocess data:', file_baard_train)
        obj = torch.load(file_baard_train)
        X_baard_train_s1 = obj['X_s1']
        X_baard_train= obj['X']
        y_baard_train = obj['y']
    else:
        tensor_X, tensor_y = get_correct_examples(model, dataset_train,
                                                  device=device,
                                                  return_tensor=True)
        X_baard_train = tensor_X.cpu().detach().numpy()
        y_baard_train = tensor_y.cpu().detach().numpy()

        # fixme: this gives an error as it expect a PIL image
        X_baard_train_s1 = preprocess_baard(data, X_baard_train
                                            ).cpu().detach().numpy()
        obj = {
            'X_s1': X_baard_train_s1,
            'X': X_baard_train,
            'y': y_baard_train
        }
        torch.save(obj, file_baard_train)
        print('Save BAARD training data to:', file_baard_train)
    
    print('X_baard_train_s1', X_baard_train_s1.shape)

    with open(json_param) as j:
        baard_param = json.load(j)
    print('Param:', baard_param)
    sequence = baard_param['sequence']
    stages = []
    if sequence[0]:
        stages.append(ApplicabilityStage(n_classes=n_classes, quantile=baard_param['q1']))
    if sequence[1]:
        stages.append(ReliabilityStage(n_classes=n_classes, k=baard_param['k_re'], quantile=baard_param['q2']))
    if sequence[2]:
        stages.append(DecidabilityStage(n_classes=n_classes, k=baard_param['k_de'], quantile=baard_param['q3']))
    print('BAARD stages:', len(stages))
    detector = BAARDOperator(stages=stages)
    assert X_baard_train.shape == X_baard_train_s1.shape, 'Unmatched size: {}, {}'.format(X_baard_train.shape, X_baard_train_s1.shape)
    assert X_baard_train_s1.shape[0] == y_baard_train.shape[0]
    detector.stages[0].fit(X_baard_train_s1, y_baard_train)
    for stage in detector.stages[1:]:
        stage.fit(X_baard_train, y_baard_train)

    file_baard_threshold = os.path.join(path, '{}_{}_baard_threshold.pt'.format(data, model_name))
    if os.path.exists(file_baard_threshold):
        print('Found existing BAARD thresholds:', file_baard_threshold)
        detector.load(file_baard_threshold)
    else:
        # Search thresholds
        detector.search_thresholds(X_def_val, y_def_val, np.zeros_like(y_def_val))
        detector.save(file_baard_threshold)

    print('-------------------------------------------------------------------')
    print('Start testing BAARD...')

    time_start = time.time()
    label_adv = detector.detect(adv_def_test, pred_adv_def_test)
    label_clean = detector.detect(X_def_test, y_def_test)
    time_elapsed = time.time() - time_start
    print('Total run time:', str(datetime.timedelta(seconds=time_elapsed)))

    acc = acc_on_adv(pred_adv_def_test, y_def_test, label_adv)
    fpr = np.mean(label_clean)
    print('Acc_on_adv:', acc)
    print('FPR:', fpr)

    obj = {
        'X': X_def_test,
        'y': y_def_test,
        'adv': adv_def_test,
        'label_adv': label_adv,
        'label_clean': label_clean,
        'pred_adv': pred_adv_def_test
    }
    file_baard_output = os.path.join(path, '{}_{}_{}_{}_baard_output.pt'.format(data, model_name, att_name, round(eps * 1000)))
    torch.save(obj, file_baard_output)
    print('Save to:', file_baard_output)

    print('-------------------------------------------------------------------')
    print('Start training surrogate model...')
    file_surro = os.path.join(path, '{}_{}_baard_surrogate.pt'.format(data, model_name))
#    if os.path.exists(file_surro):
#        print('Found existing surrogate model:', file_surro)
#        surrogate = get_pretrained_surrogate(file_surro, device)
#    else:
        # Prepare data for surrogate model
        # file_surro_data = os.path.join(path, '{}_{}_surrogate_data.pt'.format(data, model_name))
        # if os.path.exists(file_surro_data):
        #     print('Found existing surrogate dataset:', file_surro_data)
        #     obj = torch.load(file_surro_data)
        #     X_train = obj['X_train']
        #     label_train = obj['label_train']
        #     X_test = obj['X_test']
        #     label_test = obj['label_test']
        #     print(X_train.shape, label_train.shape, X_test.shape, label_test.shape)
        #     print('Labelled as adv:', np.mean(label_train == 1), np.mean(label_test == 1))
        # else:

    file_surro_data = os.path.join(path, '{}_{}_surrogate_data.pt'.format(data,
                                                                          model_name))

    adv_surro_train_2 = \
    run_attack_untargeted(file_model, X_surro_train,
                          y_surro_train,
                          att_name=att_name,
                          eps=eps_1, device=device)[0]
    adv_surro_train_3 = \
    run_attack_untargeted(file_model, X_surro_train,
                          y_surro_train,
                          att_name=att_name,
                          eps=eps_2, device=device)[0]
    adv_surro_train_4 = \
    run_attack_untargeted(file_model, X_surro_train,
                          y_surro_train,
                          att_name=att_name,
                          eps=eps_3, device=device)[0]
    adv_surro_train_5 = \
    run_attack_untargeted(file_model, X_surro_train,
                          y_surro_train,
                          att_name=att_name,
                          eps=eps_4, device=device)[0]
    adv_surro_train = np.append(adv_surro_train,adv_surro_train_2,axis = 0)
    adv_surro_train = np.append(adv_surro_train,adv_surro_train_3,axis=0)
    adv_surro_train = np.append(adv_surro_train,adv_surro_train_4, axis=0)
    adv_surro_train = np.append(adv_surro_train,adv_surro_train_5, axis=0)

    # augment also the number of benign dataset to avoid having an
    # unbalanced data
    X_surro_train_replicated = np.append(X_surro_train,X_surro_train,axis = 0)
    X_surro_train_replicated = np.append(X_surro_train_replicated, X_surro_train, axis=0)
    X_surro_train_replicated = np.append(X_surro_train_replicated, X_surro_train, axis=0)
    X_surro_train_replicated = np.append(X_surro_train_replicated, X_surro_train, axis=0)

    # classify the surrogate set
    pred_adv_surro_train = predict_numpy(model, adv_surro_train, device)
    label_adv_train = detector.detect(adv_surro_train, pred_adv_surro_train)
    label_X_train = detector.detect(X_surro_train_replicated, y_surro_train)
    # concatenate the clean and the adversarial samples
    X_train = np.concatenate((X_surro_train_replicated, adv_surro_train))
    label_train = np.concatenate((label_X_train, label_adv_train))

    label_adv_test = detector.detect(adv_att_test[:1000], pred_adv_att_test[:1000])
    label_X_test = detector.detect(X_att_test[:1000], y_att_test[:1000])
    X_test = np.concatenate((X_att_test[:1000], adv_att_test[:1000]))
    label_test = np.concatenate((label_X_test, label_adv_test))
    print(X_train.shape, label_train.shape, X_test.shape, label_test.shape)
    print('Labelled as adv:', np.mean(label_train == 1), np.mean(label_test == 1))

    obj = {
        'X_train': X_train,
        'y_train': np.concatenate((y_surro_train, y_surro_train)),
        'pred_train': np.concatenate((y_surro_train, pred_adv_surro_train)),
        'label_train': label_train,
        'X_test': X_test,
        'y_test': np.concatenate((y_att_test[:1000], y_att_test[:1000])),
        'pred_test': np.concatenate((y_att_test[:1000], pred_adv_att_test[:1000])),
        'label_test': label_test
    }

    torch.save(obj, file_surro_data)
    print('Save surrogate training data to:', file_surro_data)

    surrogate = train_surrogate(X_train, X_test, label_train, label_test, epochs=EPOCHS, device=device)
    torch.save(surrogate.state_dict(), file_surro)
    print('Save surrogate model to:', file_surro)

    print('-------------------------------------------------------------------')
    print('Start testing surrogate model...')
    X_test = np.concatenate((X_att_test[1000:], adv_att_test[1000:]))
    pred_test = predict_numpy(model, X_test, device)
    label_test = detector.detect(X_test, pred_test)
    acc = acc_on_adv(pred_test[1000:], y_att_test[1000:], label_test[1000:])
    fpr = np.mean(label_test[:1000])
    print('BAARD Acc_on_adv:', acc)
    print('BAARD FPR:', fpr)

    label_surro = predict_numpy(surrogate, X_test, device)
    acc = np.mean(label_surro == label_test)
    print('Acc on surrogate:', acc)

    print('DONE!')
    print('-------------------------------------------------------------------\n')


if __name__ == '__main__':
    path_cur = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_cur, 'seeds.json')) as j:
        json_obj = json.load(j)
        seeds = json_obj['seeds']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10', choices=[
        'mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='resnet', choices=['dnn',
                                                                   'resnet', 'vgg'])
    parser.add_argument('--attack', type=str, default='apgd2', choices=ATTACKS)
    parser.add_argument('--eps', type=float, default=2.0)
    parser.add_argument('--json', type=str, default=None, help="JSON file BAARD's hyperparameters")
    parser.add_argument('--idx', type=int, default=2, choices=list(range(
        len(seeds))))
    args = parser.parse_args()
    print(args)
    if args.json is None:
        args.json = os.path.join('params', 'baard_{}_3.json'.format(args.data))
    if not os.path.exists(args.json):
        raise FileExistsError('Cannot file JSON param file for BAARD.')
    idx = args.idx
    run_full_pipeline_baard(
        data=args.data,
        model_name=args.model,
        path='result_{}'.format(str(idx)),
        seed=seeds[idx],
        json_param=args.json,
        att_name=args.attack,
        eps=args.eps)

# python3 ./pipeline/full_pipeline_baard.py --data mnist --model dnn --attack apgd2 --eps 2.0 --json "./params/baard_mnist_3.json" --idx 0

# python3 ./pipeline/full_pipeline_baard.py --data cifar10 --model resnet --attack apgd2 --eps 2.0 --json "./params/baard_cifar10_3.json" --idx 0
