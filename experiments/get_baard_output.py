import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision as tv
import torchvision.datasets as datasets

sys.path.append(os.getcwd())
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage, flatten)
from defences.util import acc_on_adv, get_correct_examples
from models.cifar10 import Resnet, Vgg
from models.mnist import BaseModel
from models.torch_util import predict_numpy

from experiments.util import set_seeds

N_CLASSES = 10


def count_class(y):
    for i in range(N_CLASSES):
        print(i, np.sum(y == i))


def get_baard_output(data, model_name, data_path, output_path, file_name, param, batch_size, device):
    """This function reads a dataset object. It runs BAARD, applies clipping and 
    adds label_as_adv to the object.
    """
    file_path = os.path.join(output_path, file_name)
    print('file_path:', file_path)

    obj = torch.load(file_path)
    X = obj['X']
    adv = obj['adv']
    y = obj['y']

    # Load model
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data == 'mnist':
        dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms)
        model = BaseModel(use_prob=False).to(device)
        pretrained = 'mnist_200.pt'
    elif data == 'cifar10':
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms)
        if model_name == 'resnet':
            model = Resnet(use_prob=False).to(device)
            pretrained = 'cifar10_resnet_200.pt'
        elif model_name == 'vgg':
            model = Vgg(use_prob=False).to(device)
            pretrained = 'cifar10_vgg_200.pt'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    pretrained_path = os.path.join(output_path, pretrained)
    model.load_state_dict(torch.load(pretrained_path))
    pred = predict_numpy(model, X, device)
    acc = np.mean(pred == y)
    print('Accuracy on {} clean samples: {}'.format(X.shape[0], acc))

    tensor_train_X, tensor_train_y = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()

    # Load the preprocessed training set
    baard_train_path = os.path.join(output_path, '{}_{}_baard_train.pt'.format(data, model_name))
    obj = torch.load(baard_train_path)
    X_baard = obj['X_train']

    # Load the original validation set for BAARD
    # eg: ./results/mnist_basic_apgd2_2.0_adv.npy
    file_root = '{}_{}_apgd2_2.0'.format(data, model_name)
    path_benign = os.path.join(output_path, file_root + '_x.npy')
    path_y = os.path.join(output_path, file_root + '_y.npy')
    X_val = np.load(path_benign)
    y_val = np.load(path_y)
    n = X_val.shape[0] // 2
    X_val = X_val[n:]
    y_val = y_val[n:]

    stages = []
    stages.append(ApplicabilityStage(n_classes=N_CLASSES, quantile=param['q1']))
    stages.append(ReliabilityStage(n_classes=N_CLASSES, k=param['k_re'], quantile=param['q2']))
    stages.append(DecidabilityStage(n_classes=N_CLASSES, k=param['k_de'], quantile=param['q3']))
    print('BAARD: # of stages:', len(stages))

    detector = BAARDOperator(stages=stages)
    detector.stages[0].fit(X_baard, y_train)
    detector.stages[1].fit(X_train, y_train)
    detector.stages[2].fit(X_train, y_train)
    detector.search_thresholds(X_val, y_val, np.zeros_like(y_val))

    pred_adv = predict_numpy(model, adv, device)
    print('Acc on adv without clip:', np.mean(pred_adv == y))

    # count_class(pred_adv)

    # TODO: After clipping, the 1st stage still blocks samples. I don't know why?!
    # To bypass the 1st stage, we want to clip all adversarial examples with the bounding boxes
    applicability = detector.stages[0]
    thresholds = applicability.thresholds_
    adv_clipped = adv.copy()
    for c in range(N_CLASSES):
        idx = np.where(pred_adv == c)[0]
        # Adversarial examples do NOT have the same distribution as the true classes
        if len(idx) == 0:
            continue
        bounding_boxes = thresholds[c]
        low = bounding_boxes[0]
        high = bounding_boxes[1]
        shape = adv_clipped[idx].shape
        subset = flatten(adv[idx])
        # clipped_subset = np.clip(subset, low, high)
        subset = np.minimum(subset, high)
        subset = np.maximum(subset, low)
        adv_clipped[idx] = subset.reshape(shape)

    pred_adv_clip = predict_numpy(model, adv_clipped, device)
    print('Acc on adv with clip:', np.mean(pred_adv_clip == y))
    print('Class changed after clipping:', np.sum(pred_adv != pred_adv_clip))

    pred_X = predict_numpy(model, X, device)
    assert not np.all([pred_X, y])
    baard_label_adv = detector.detect(adv_clipped, pred_adv_clip)

    s1_blocked = detector.stages[0].predict(adv_clipped, pred_adv_clip)
    print('Blocked by Stage1:', np.sum(s1_blocked))

    acc = acc_on_adv(pred_adv_clip, y, baard_label_adv)
    print('Acc_on_adv:', acc)

    baard_label_x = detector.detect(X, y)
    print('FPR:', np.mean(baard_label_x))

    output = {
        'X': X,
        'adv': adv_clipped,
        'y': y,
        'baard_label_x': baard_label_x,
        'baard_label_adv': baard_label_adv}
    torch.save(output, file_path)
    print('Save to:', file_path)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    set_seeds(args.random_state)

    with open(args.param) as param_json:
        param = json.load(param_json)
    print('Param:', param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # eg: cifar10_resnet_baard_train_surrodata_eps2.0_size2000.pt
    name_parser = args.file.split('_')
    data = name_parser[0]
    model_name = name_parser[1]

    get_baard_output(data, model_name, args.data_path, args.output_path, args.file, param, args.batch_size, device)


if __name__ == '__main__':
    main()
