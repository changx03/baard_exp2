import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision as tv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from defences.feature_squeezing import (DepthSqueezer, FeatureSqueezingTorch,
                                        GaussianSqueezer)
from defences.util import dataset2tensor, get_shape
from models.numeric import NumericModel
from models.torch_util import validate
from experiments.util import load_csv


def main():
    with open('data.json') as data_json:
        data_params = json.load(data_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--random_state', type=int, default=1234)
    args = parser.parse_args()

    print('Dataset:', args.data)
    print('Pretrained model:', args.pretrained)

    with open(args.param) as param_json:
        param = json.load(param_json)
    param['n_classes'] = data_params['data'][args.data]['n_classes']
    print('Param:', param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    # Prepare data
    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

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

    # Note: Train set alway shuffle!
    loader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    shape_train = get_shape(loader_train.dataset)
    shape_test = get_shape(loader_test.dataset)
    print('Train set:', shape_train)
    print('Test set:', shape_test)
    use_prob = True
    print('Using softmax layer:', use_prob)

    # Load model
    n_features = data_params['data'][args.data]['n_features']
    n_classes = data_params['data'][args.data]['n_classes']
    model = NumericModel(n_features, n_hidden=n_features * 4, n_classes=n_classes, use_prob=use_prob).to(device)

    loss = nn.CrossEntropyLoss()
    pretrained_path = os.path.join(args.output_path, args.pretrained)
    model.load_state_dict(torch.load(pretrained_path))

    _, acc_train = validate(model, loader_train, loss, device)
    _, acc_test = validate(model, loader_test, loss, device)
    print('Accuracy on train set: {:.4f}%'.format(acc_train * 100))
    print('Accuracy on test set: {:.4f}%'.format(acc_test * 100))

    tensor_train_X, tensor_train_y = dataset2tensor(dataset_train)
    X_train = tensor_train_X.cpu().detach().numpy()
    y_train = tensor_train_y.cpu().detach().numpy()

    # Train defence
    squeezers = []
    squeezers.append(GaussianSqueezer(x_min=0.0, x_max=1.0, noise_strength=0.025, std=1.0))
    squeezers.append(DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=8))
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

    path_fs = os.path.join(args.output_path, '{}_fs.pt'.format(args.pretrained.split('.')[0]))
    detector.save(path_fs)
    print('Saved fs to:', path_fs)
    print()


if __name__ == '__main__':
    main()
