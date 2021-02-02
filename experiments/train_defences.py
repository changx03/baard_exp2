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
from defences.feature_squeezing import (DepthSqueezer, FeatureSqueezingTorch,
                                        MedianSqueezer, NLMeansColourSqueezer)
from defences.lid import LidDetector
from defences.magnet import (Autoencoder1, Autoencoder2,
                             MagNetAutoencoderReformer, MagNetDetector,
                             MagNetOperator)
from defences.region_based_classifier import RegionBasedClassifier
from defences.util import (acc_on_adv, dataset2tensor, get_correct_examples,
                           get_shape, merge_and_generate_labels)
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.numeric import NumericModel
from models.torch_util import (AddGaussianNoise, predict, predict_numpy,
                               validate)

from experiments.util import load_csv


def baard_preprocess(data, tensor_X):
    """Preprocess training data"""
    if data == 'cifar10':
        # return tensor_X
        transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            # tv.transforms.RandomCrop(32, padding=4),
            AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)
    elif data == 'mnist':
        # return tensor_X
        transform = tv.transforms.Compose([
            tv.transforms.RandomRotation(5)
            # AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)
    else:
        # return tensor_X
        transform = tv.transforms.Compose([
            AddGaussianNoise(mean=0., std=1., eps=0.02)
        ])
        return transform(tensor_X)


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--adv', type=str, required=True, help="Example: 'mnist_basic_apgd_0.3'")
    parser.add_argument('--defence', type=str, required=True, choices=data_params['defences'])
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--random_state', type=int, default=1234)
    parser.add_argument('--save', type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    print(args)

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)
    print('Pretrained samples:', args.adv + '_adv.npy')
    print('Defence:', args.defence)

    with open(args.param) as param_json:
        param = json.load(param_json)
    param['n_classes'] = data_params['data'][args.data]['n_classes']
    print('Param:', param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Prepare data
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    if args.data == 'mnist':
        dataset_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transforms)
    elif args.data == 'cifar10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms)
        dataset_test = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transforms)
    else:
        data_path = os.path.join(args.data_path, data_params['data'][args.data]['file_name'])
        print('Read file:', data_path)
        X, y = load_csv(data_path)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_params['data'][args.data]['n_test'],
            random_state=args.random_state)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        dataset_train = TensorDataset(torch.from_numpy(X_train).type(torch.float32), torch.from_numpy(y_train).type(torch.long))
        dataset_test = TensorDataset(torch.from_numpy(X_test).type(torch.float32), torch.from_numpy(y_test).type(torch.long))

    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    shape_train = get_shape(loader_train.dataset)
    shape_test = get_shape(loader_test.dataset)
    print('Train set:', shape_train)
    print('Test set:', shape_test)
    use_prob = True
    print('Using softmax layer:', use_prob)

    # Load model
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
        n_features = data_params['data'][args.data]['n_features']
        n_classes = data_params['data'][args.data]['n_classes']
        model = NumericModel(n_features, n_hidden=n_features * 4, n_classes=n_classes, use_prob=use_prob).to(device)
        model_name = 'basic' + str(n_features * 4)

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train * 100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test * 100))

    # Create a subset which only contains recognisable samples.
    # The original train and test sets are no longer needed.
    tensor_train_X, tensor_train_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    dataset_train = TensorDataset(tensor_train_X, tensor_train_y)
    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_train, loss, device)
    print('Accuracy on {} filtered train set: {:.4f}%'.format(len(dataset_train), acc_perfect * 100))

    tensor_test_X, tensor_test_y = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
    dataset_test = TensorDataset(tensor_test_X, tensor_test_y)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=True)
    _, acc_perfect = validate(model, loader_test, loss, device)
    print('Accuracy on {} filtered test set: {:.4f}%'.format(len(dataset_test), acc_perfect * 100))

    # Load pre-trained adversarial examples
    path_benign = os.path.join(args.output_path, args.adv + '_x.npy')
    path_adv = os.path.join(args.output_path, args.adv + '_adv.npy')
    path_y = os.path.join(args.output_path, args.adv + '_y.npy')
    X_benign = np.load(path_benign)
    adv = np.load(path_adv)
    y_true = np.load(path_y)

    dataset = TensorDataset(torch.from_numpy(X_benign), torch.from_numpy(y_true))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} benign samples: {:.4f}%'.format(len(dataset), acc * 100))

    dataset = TensorDataset(torch.from_numpy(adv), torch.from_numpy(y_true))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    _, acc = validate(model, loader, loss, device)
    print('Accuracy on {} adversarial examples: {:.4f}%'.format(len(dataset), acc * 100))

    # Do NOT shuffle the indices, so different defences can use the same test set.
    dataset = TensorDataset(torch.from_numpy(adv))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    pred_adv = predict(model, loader, device).cpu().detach().numpy()

    # Find the thresholds using the 2nd half
    n = len(X_benign) // 2
    # Merge benign samples and adversarial examples into one set.
    # This labels indicate a sample is an adversarial example or not.
    X_val, labels_val = merge_and_generate_labels(adv[n:], X_benign[n:], flatten=False)
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
            stages.append(ApplicabilityStage(n_classes=param['n_classes'], quantile=param['q1']))
        if sequence[1]:
            stages.append(ReliabilityStage(n_classes=param['n_classes'], k=param['k_re'], quantile=param['q2']))
        if sequence[2]:
            stages.append(DecidabilityStage(n_classes=param['n_classes'], k=param['k_de'], quantile=param['q3']))
        print('BAARD: # of stages:', len(stages))
        detector = BAARDOperator(stages=stages)

        # Run preprocessing
        X_baard = baard_preprocess(args.data, tensor_train_X).cpu().detach().numpy()
        # Fit the model with the filtered the train set.
        detector.stages[0].fit(X_baard, y_train)
        detector.stages[1].fit(X_train, y_train)
        if len(detector.stages) == 3:
            detector.stages[2].fit(X_train, y_train)
        detector.search_thresholds(X_val, pred_val, labels_val)
    elif args.defence == 'fs':
        squeezers = []
        if args.data == 'mnist':
            squeezers.append(DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=1))
            squeezers.append(MedianSqueezer(x_min=0.0, x_max=1.0, kernel_size=2))
        elif args.data == 'cifar10':
            squeezers.append(DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=4))
            squeezers.append(MedianSqueezer(x_min=0.0, x_max=1.0, kernel_size=2))
            squeezers.append(NLMeansColourSqueezer(x_min=0.0, x_max=1.0, h=2, templateWindowsSize=3, searchWindowSize=13))
        else:
            raise NotImplementedError
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
        path_fs = os.path.join(args.output_path, '{}_fs.pt'.format(args.pretrained.split('.')[0]))
        detector.load(path_fs)
        detector.search_thresholds(X_val, pred_val, labels_val)
    elif args.defence == 'lid':
        # This batch_size is not same as the mini batch size for the neural network.
        before_softmax = args.data == 'cifar10'
        detector = LidDetector(
            model,
            k=param['k'],
            batch_size=param['batch_size'],
            x_min=0.0,
            x_max=1.0,
            device=device,
            before_softmax=before_softmax)
        # LID uses different training set
        X_train, y_train = detector.get_train_set(X_benign[n:], adv[n:], std_dominator=param['std_dominator'])
        detector.fit(X_train, y_train, verbose=1)
    elif args.defence == 'magnet':
        magnet_detectors = []
        # Different datasets require different autoencoders.
        if args.data == 'mnist':
            # autoencoder1 and autoencoder2
            magnet_detectors.append(MagNetDetector(
                encoder=Autoencoder1(n_channel=1),
                classifier=model,
                lr=param['lr'],
                batch_size=param['batch_size'],
                weight_decay=param['weight_decay'],
                x_min=0.0,
                x_max=1.0,
                noise_strength=param['noise_strength'],
                algorithm='error',
                p=1,
                device=device))
            magnet_detectors.append(MagNetDetector(
                encoder=Autoencoder2(n_channel=1),
                classifier=model,
                lr=param['lr'],
                batch_size=param['batch_size'],
                weight_decay=param['weight_decay'],
                x_min=0.0,
                x_max=1.0,
                noise_strength=param['noise_strength'],
                algorithm='error',
                p=2,
                device=device))
        elif args.data == 'cifar10':
            autoencoder = Autoencoder2(
                n_channel=data_params['data'][args.data]['n_features'][0])
            # There are 3 autoencoder based detectors, but they use the same architecture.
            magnet_detectors.append(MagNetDetector(
                encoder=autoencoder,
                classifier=model,
                lr=param['lr'],
                batch_size=param['batch_size'],
                weight_decay=param['weight_decay'],
                x_min=0.0,
                x_max=1.0,
                noise_strength=param['noise_strength'],
                algorithm='error',
                p=2,
                device=device))
            magnet_detectors.append(MagNetDetector(
                encoder=autoencoder,
                classifier=model,
                lr=param['lr'],
                batch_size=param['batch_size'],
                weight_decay=param['weight_decay'],
                x_min=0.0,
                x_max=1.0,
                noise_strength=param['noise_strength'],
                algorithm='prob',
                temperature=10,
                device=device))
            magnet_detectors.append(MagNetDetector(
                encoder=autoencoder,
                classifier=model,
                lr=param['lr'],
                batch_size=param['batch_size'],
                weight_decay=param['weight_decay'],
                x_min=0.0,
                x_max=1.0,
                noise_strength=param['noise_strength'],
                algorithm='prob',
                temperature=40,
                device=device))
        else:
            raise ValueError('Magnet requires autoencoder.')

        for i, ae in enumerate(magnet_detectors, start=1):
            ae_path = os.path.join(args.output_path, 'autoencoder_{}_{}_{}.pt'.format(args.data, model_name, i))
            ae.load(ae_path)
            tensor_X_test, _ = dataset2tensor(dataset_test)
            X_test = tensor_X_test.cpu().detach().numpy()
            print('Autoencoder {} MSE training set: {:.6f}, test set: {:.6f}'.format(i, ae.score(X_train), ae.score(X_test)))
            print('Autoencoder {} threshold: {}'.format(i, ae.threshold))

        reformer = MagNetAutoencoderReformer(
            encoder=magnet_detectors[0].encoder,
            batch_size=param['batch_size'],
            device=device)

        detector = MagNetOperator(
            classifier=model,
            detectors=magnet_detectors,
            reformer=reformer,
            batch_size=param['batch_size'],
            device=device)
    elif args.defence == 'rc':
        detector = RegionBasedClassifier(
            model=model,
            r=param['r'],
            sample_size=param['sample_size'],
            n_classes=param['n_classes'],
            x_min=0.0,
            x_max=1.0,
            batch_size=param['batch_size'],
            r0=param['r0'],
            step_size=param['step_size'],
            stop_value=param['stop_value'],
            device=device)
        # Region-based classifier only uses benign samples to search threshold.
        # The r value is already set to the optimal. We don't need to search it.
        # detector.search_thresholds(X_val, pred_val, labels_val, verbose=0)
    else:
        raise ValueError('{} is not supported!'.format(args.defence))
    time_elapsed = time.time() - time_start
    print('Total training time:', str(datetime.timedelta(seconds=time_elapsed)))

    # Test defence
    time_start = time.time()
    X_test, labels_test = merge_and_generate_labels(adv[:n], X_benign[:n], flatten=False)
    pred_test = np.concatenate((pred_adv[:n], y_true[:n]))
    y_test = np.concatenate((y_true[:n], y_true[:n]))

    # Only MegNet uses reformer.
    X_reformed = None
    if args.defence == 'magnet':
        X_reformed, res_test = detector.detect(X_test, pred_test)
        y_pred = predict_numpy(model, X_reformed, device)
    elif args.defence == 'rc':
        y_pred = detector.detect(X_test, pred_test)
        res_test = np.zeros_like(y_pred)
    else:
        res_test = detector.detect(X_test, pred_test)
        y_pred = pred_test

    acc = acc_on_adv(y_pred[:n], y_test[:n], res_test[:n])
    if args.defence == 'rc':
        fpr = np.mean(y_pred[n:] != y_test[n:])
    else:
        fpr = np.mean(res_test[n:])
    print('Acc_on_adv:', acc)
    print('FPR:', fpr)
    time_elapsed = time.time() - time_start
    print('Total test time:', str(datetime.timedelta(seconds=time_elapsed)))

    # Save results
    suffix = '_' + args.suffix if args.suffix is not None else ''

    if args.save:
        path_result = os.path.join(args.output_path, '{}_{}{}.pt'.format(args.adv, args.defence, suffix))
        torch.save({
            'X_val': X_val,
            'y_val': np.concatenate((y_true[n:], y_true[n:])),
            'labels_val': labels_val,
            'X_test': X_test,
            'y_test': y_test,
            'labels_test': labels_test,
            'res_test': y_pred if args.defence == 'rc' else res_test,
            'X_reformed': X_reformed,
            'param': param}, path_result)
        print('Saved to:', path_result)
    else:
        print('No file is save!')
    print()


if __name__ == '__main__':
    main()
