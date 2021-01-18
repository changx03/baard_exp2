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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Adding the parent directory.
sys.path.append(os.getcwd())
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from defences.feature_squeezing import (GaussianSqueezer, MedianSqueezer, 
                                        DepthSqueezer, FeatureSqueezingTorch)
from defences.lid import LidDetector
from defences.util import (get_correct_examples, get_shape, 
                           merge_and_generate_labels, score)
from experiments.train_pt import validate, predict
from experiments.util import load_csv
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel

# This seed ensures the pre-trained models have the same train and test sets.
RANDOM_STATE = int(2**12)

DATA_NAMES = ['mnist', 'cifar10', 'banknote', 'htru2', 'segment', 'texture']
DATA = {
    'mnist': {'n_features': (1, 28, 28), 'n_classes': 10},
    'cifar10': {'n_features': (3, 32, 32), 'n_classes': 10},
    'banknote': {'file_name': 'banknote_preprocessed.csv', 'n_features': 4, 'n_test': 400, 'n_classes': 2},
    'htru2': {'file_name': 'htru2_preprocessed.csv', 'n_features': 8, 'n_test': 4000, 'n_classes': 2},
    'segment': {'file_name': 'segment_preprocessed.csv', 'n_features': 18, 'n_test': 400, 'n_classes': 7},
    'texture': {'file_name': 'texture_preprocessed.csv', 'n_features': 40, 'n_test': 600, 'n_classes': 11}}
ATTACKS = ['apgd', 'bim', 'boundary', 'cw2', 'deepfool', 'fgsm', 'jsma']
DEFENCES = ['baard', 'fs', 'lid', 'magnet', 'rc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=DATA_NAMES)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--adv', type=str, required=True,
                        help="Example: 'mnist_basic_apgd_0.3'")
    parser.add_argument('--defence', type=str, required=True, choices=DEFENCES)
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--suffix', type=str)
    args = parser.parse_args()

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)
    print('Pretrained samples:', args.adv + '_adv.npy')
    print('Defence:', args.defence)

    with open(args.param) as param_json:
        param = json.load(param_json)
    param['n_classes'] = DATA[args.data]['n_classes']
    print('Param:', param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Prepare data
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    if args.data in ['banknote', 'htru2', 'segment', 'texture', 'yeast']:
        data_path = os.path.join(args.data_path, DATA[args.data]['file_name'])
        print('Read file:', data_path)
        X, y = load_csv(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=DATA[args.data]['n_test'],
            random_state=RANDOM_STATE)
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

    loader_train = DataLoader(
        dataset_train, batch_size=512, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    shape_train = get_shape(loader_train.dataset)
    shape_test = get_shape(loader_test.dataset)
    print('Train set:', shape_train)
    print('Test set:', shape_test)

    # Load model
    use_prob = True
    if args.defence == 'lid':  # Handle divide by zero issue
        use_prob = False
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
        n_features = DATA[args.data]['n_features']
        n_classes = DATA[args.data]['n_classes']
        model = NumericModel(
            n_features,
            n_hidden=n_features * 4,
            n_classes=n_classes,
            use_prob=use_prob).to(device)
        model_name = 'basic' + str(n_features * 4)

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train*100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test*100))

    # Create a subset which only contains recognisable samples.
    # The original train and test sets are no longer needed.
    tensor_train_X, tensor_train_y = get_correct_examples(
        model, dataset_train, device=device, return_tensor=True)
    dataset_train = TensorDataset(tensor_train_X, tensor_train_y)
    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_train, loss, device)
    print('Accuracy on {} filtered train set: {:.4f}%'.format(
        len(dataset_train), acc_perfect*100))

    tensor_test_X, tensor_test_y = get_correct_examples(
        model, dataset_test, device=device, return_tensor=True)
    dataset_test = TensorDataset(tensor_test_X, tensor_test_y)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_test, loss, device)
    print('Accuracy on {} filtered test set: {:.4f}%'.format(
        len(dataset_test), acc_perfect*100))

    # Load pre-trained adversarial examples
    path_benign = os.path.join(args.output_path, args.adv + '_x.npy')
    path_adv = os.path.join(args.output_path, args.adv + '_adv.npy')
    path_y = os.path.join(args.output_path, args.adv + '_y.npy')
    X_benign = np.load(path_benign)
    adv = np.load(path_adv)
    y_true = np.load(path_y)

    dataset = TensorDataset(
        torch.from_numpy(X_benign), torch.from_numpy(y_true))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} benign samples: {:.4f}%'.format(
        len(dataset), acc*100))
    dataset = TensorDataset(
        torch.from_numpy(adv), torch.from_numpy(y_true))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} adversarial examples: {:.4f}%'.format(
        len(dataset), acc*100))

    # Do NOT shuffle the indices, so different defences can use the same test set.
    # shuffle_idx = np.random.permutation(len(X_benign))
    # X_benign = X_benign[shuffle_idx]
    # y_true = y_true[shuffle_idx]
    # adv = adv[shuffle_idx]

    dataset = TensorDataset(torch.from_numpy(adv))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    pred_adv = predict(model, loader, device).cpu().detach().numpy()

    # Find the thresholds using the 2nd half
    n = len(X_benign) // 2
    # Merge benign samples and adversarial examples into one set.
    # This labels indicate a sample is an adversarial example or not.
    X_val, labels_val = merge_and_generate_labels(
        adv[n:], X_benign[n:], flatten=False)
    # The predictions for benign samples are exactly same as the true labels.
    pred_val = np.concatenate((pred_adv[n:], y_true[n:]))

    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()
    # Train defence
    time_start = time.time()
    if args.defence == 'baard':
        sequence = param['sequence']
        stages = []
        if sequence[0]:
            stages.append(ApplicabilityStage(
                n_classes=param['n_classes'], quantile=param['quantile']))
        if sequence[1]:
            stages.append(ReliabilityStage(
                n_classes=param['n_classes'], k=param['k_re']))
        if sequence[2]:
            stages.append(DecidabilityStage(
                n_classes=param['n_classes'], k=param['k_de'], 
                n_bins=param['n_bins']))
        print('BAARD: # of stages:', len(stages))
        detector = BAARDOperator(stages=stages)

        # Fit the model with the filtered the train set.
        detector.fit(X_train, y_train)
        detector.search_thresholds(X_val, pred_val, labels_val)
    elif args.defence == 'fs':
        squeezers = []
        squeezers.append(GaussianSqueezer(x_min=0.0, x_max=1.0, noise_strength=0.025, std=1.0))
        squeezers.append(DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=8))
        if args.data in ['mnist', 'cifar10']:
            squeezers.append(MedianSqueezer(x_min=0.0, x_max=1.0, kernel_size=3))
        print('FS: # of squeezers:', len(squeezers))
        detector = FeatureSqueezingTorch(
            classifier=model,
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4,
            loss=loss,
            batch_size=128,
            x_min=0.0,
            x_max=1.0,
            squeezers=squeezers,
            n_classes=param['n_classes'],
            device=device)
        detector.fit(X_train, y_train, epochs=param['epochs'], verbose=1)
        detector.search_thresholds(X_val, pred_val, labels_val)
    elif args.defence == 'lid':
        # This batch_size is not same as the mini batch size for the neural network.
        detector = LidDetector(
            model, 
            k=param['k'], 
            batch_size=param['batch_size'], 
            x_min=0.0,
            x_max=1.0,
            device=device)
        # LID uses different training set
        X_train, y_train = detector.get_train_set(
            X_benign[n:], adv[n:], std_dominator=param['std_dominator'])
        detector.fit(X_train, y_train, verbose=1)
    elif args.defence == 'magnet':
        raise NotImplementedError
    elif args.defence == 'rc':
        raise NotImplementedError
    else:
        raise ValueError('{} is not supported!'.format(args.defence))
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))

    # Test defence
    time_start = time.time()
    X_test, labels_test = merge_and_generate_labels(
        adv[:n], X_benign[:n], flatten=False)
    pred_test = np.concatenate((pred_adv[:n], y_true[:n]))
    y_test = np.concatenate((y_true[:n], y_true[:n]))

    res_test = detector.detect(X_test, pred_test)
    acc = score(res_test, y_test, pred_test, labels_test)
    print('Success rate: {:.4f}%'.format(acc*100))
    time_elapsed = time.time() - time_start
    print('Total test time:', str(datetime.timedelta(seconds=time_elapsed)))
    
    # Save results
    suffix = '_' + args.suffix if args.suffix is not None else ''
    
    path_result = os.path.join(args.output_path, '{}_{}{}.pt'.format(
        args.adv, args.defence, suffix))
    torch.save({
        'X_val': X_val,
        'y_val': np.concatenate((y_true[n:], y_true[n:])),
        'labels_val': labels_val,
        'X_test': X_test,
        'y_test': y_test,
        'labels_test': labels_test,
        'res_test': res_test,
        'param': param}, path_result)
    print('Saved to:', path_result)
    print()


if __name__ == '__main__':
    main()
