"""
Evalute BAARD on white-box surrogate attacks.
"""
import os
import sys

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import argparse
import datetime
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from art.attacks.evasion.auto_projected_gradient_descent_detectors import \
    AutoProjectedGradientDescentDetectors
from art.classifiers import PyTorchClassifier
from attacks.bypass_baard import BAARD_Clipper
from defences.baard import BAARDOperator, DecidabilityStage, ReliabilityStage
from models.cifar10 import Resnet
from models.mnist import BaseModel
from models.torch_util import predict_numpy, validate
from torch.utils.data import DataLoader, TensorDataset
from utils import acc_on_advx, get_correct_examples, mkdir, set_seeds

from experiments import get_baard, get_output_path, pytorch_train_classifier
from experiments.train_baard_surrogate import (SurrogateModel,
                                               train_surrogate_v2)

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 192
EPOCHS = 100  # Early stop by default
DEF_NAME = 'baard'
ATT_SURR = 'apgd2'  # the attack for training surrogate model.
# NOTE: We combine attacks with different epsilons when training the surrogate
# model. The goal is to obtain a good training set, so the surrogate model can
# behave as similar as BAARD.
EPS_SURR_MNIST = [1., 2., 5., 8.]
EPS_SURR_CIFAR10 = [0.1, 0.5, 1., 2.]
ATT_TEST = 'apgd2'  # This is for testing the surrogate model.
EPS_TEST = 2.


def whitebox_baard(data_name, epsilons, idx, baard_param=None):
    print('Runing whitebox_baard.py')
    seed = SEEDS[idx]
    set_seeds(seed)

    n_classes = 10
    if data_name == 'mnist':
        model_name = 'dnn'
        input_shape = (1, 28, 28)
    elif data_name == 'cifar10':
        model_name = 'resnet'
        input_shape = (3, 32, 32)
    else:
        raise NotImplementedError
    print('[CLASSIFIER] model_name:', model_name)

    path_results = get_output_path(idx, data_name, model_name)
    path_data = os.path.join(path_results, 'data')
    mkdir(path_data)
    path_wb_data = os.path.join(path_data, 'whitebox')
    mkdir(path_wb_data)
    path_results = os.path.join(path_results, 'results')
    mkdir(path_results)
    path_wb_results = os.path.join(path_results, 'whitebox')
    mkdir(path_wb_results)

    # Step 1: Load data
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    if data_name == 'mnist':
        dataset_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)
    else:  # elif data_name == 'cifar10':  # Already tested
        dataset_train = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(DATA_PATH, train=False, download=True, transform=transform)

    ############################################################################
    # Step 2: Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[CLASSIFIER] Device: {}'.format(device))

    file_model = os.path.join(path_data, '{}_{}_model.pt'.format(data_name, model_name))
    if not os.path.exists(file_model):
        pytorch_train_classifier(data_name, model_name, idx)

    if data_name == 'mnist':
        model = BaseModel(use_prob=False).to(device)
    else:  # if data_name == 'cifar10':  # Already tested
        model = Resnet(use_prob=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(file_model, map_location=device))

    ############################################################################
    # Step 3: Filter, split data
    # Filter data
    path_X_train = os.path.join(path_data, '{}_{}_X_train.npy'.format(data_name, model_name))
    if os.path.exists(path_X_train):
        print('[DATA] Found data:', path_X_train)
        X_train = np.load(path_X_train)
        y_train = np.load(os.path.join(path_data, '{}_{}_y_tain.npy'.format(data_name, model_name)))
        X_test = np.load(os.path.join(path_data, '{}_{}_X_test.npy'.format(data_name, model_name)))
        y_test = np.load(os.path.join(path_data, '{}_{}_y_test.npy'.format(data_name, model_name)))
    else:
        tensor_X_train, tensor_y_train = get_correct_examples(model, dataset_train, device=device, return_tensor=True)
        tensor_X_test, tensor_y_test = get_correct_examples(model, dataset_test, device=device, return_tensor=True)
        X_train = tensor_X_train.cpu().detach().numpy()
        y_train = tensor_y_train.cpu().detach().numpy()

        X_test = tensor_X_test.cpu().detach().numpy()
        y_test = tensor_y_test.cpu().detach().numpy()

        # Shuffle indices before saving
        idx_shuffle = np.random.permutation(X_test.shape[0])
        X_test = X_test[idx_shuffle]
        y_test = y_test[idx_shuffle]

        np.save(path_X_train, X_train)
        np.save(os.path.join(path_data, '{}_{}_X_test.npy'.format(data_name, model_name)), X_test)
        np.save(os.path.join(path_data, '{}_{}_y_tain.npy'.format(data_name, model_name)), y_train)
        np.save(os.path.join(path_data, '{}_{}_y_test.npy'.format(data_name, model_name)), y_test)
        print('[DATA] Save to:', path_X_train)

    # Testing
    dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    _, acc_perfect = validate(model, loader, loss, device)
    print('[DATA] Acc. on {} filtered test set: {:.2f}%'.format(y_test.shape[0], acc_perfect * 100))

    # Split rules:
    X_att = X_test[:1000]  # test set for defences
    y_att = y_test[:1000]
    X_val = X_test[1000: 2000]  # evaluation set for defences
    y_val = y_test[1000: 2000]
    X_wb_test = X_test[2000: 4000]  # test set for whitebox attacks
    y_wb_test = y_test[2000: 4000]
    X_surr_train = X_test[4000: 6000]  # training set for surrogate model
    y_surr_train = y_test[4000: 6000]
    print('[DATA] After split:', X_att.shape, X_val.shape, X_wb_test.shape, X_surr_train.shape)

    ############################################################################
    # Step 4: Load detector
    if baard_param is None:
        baard_param = os.path.join('params', 'baard_{}.json'.format(data_name))
    if not os.path.exists(baard_param):
        raise FileNotFoundError("Cannot find BAARD's config file: {}".format(baard_param))

    detector = get_baard(
        data_name=data_name,
        model_name=model_name,
        idx=idx,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        baard_param=baard_param)

    stages = []
    s2 = detector.stages[1]
    s3 = detector.stages[2]
    stages.append(s2)
    stages.append(s3)
    baard_no_s1 = BAARDOperator(stages=stages)

    ############################################################################
    # Step 5: Train surrogate model
    path_surr_model = os.path.join(path_wb_data, '{}_{}_baard_surrogate.pt'.format(data_name, model_name))
    if os.path.exists(path_surr_model):
        print('[SURROGATE] Found surrogate model:', path_surr_model)
        if data_name == 'mnist':
            n_channels = 1
        else:  # elif data == 'cifar10':
            n_channels = 3
        surrogate = SurrogateModel(in_channels=n_channels).to(device)
        surrogate.load_state_dict(torch.load(path_surr_model, map_location=device))
    else:
        idx_choice = np.random.choice(X_train.shape[0], size=10000, replace=False)
        surr_epsilons = EPS_SURR_MNIST if data_name == 'mnist' else EPS_SURR_CIFAR10
        surrogate = train_surrogate_v2(
            model=model,
            detector=baard_no_s1,
            data_name=data_name,
            X_train=X_train[idx_choice],
            y_train=y_train[idx_choice],
            X_test=X_att,
            y_test=y_att,
            epochs=EPOCHS,
            att_name=ATT_SURR,
            epsilons=surr_epsilons,
            eps_test=2.0,
            path=path_wb_data,
            device=device)
        torch.save(surrogate.state_dict(), path_surr_model)
        print('[SURROGATE] Save surrogate model to:', path_surr_model)

    ############################################################################
    # Step 6: Perform white-box attacks
    device_type = 'cpu' if device == 'cpu' else 'gpu'
    art_classifier = PyTorchClassifier(
        model,
        loss=loss,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type=device_type)

    optimizer_surr = torch.optim.SGD(surrogate.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    art_detector = PyTorchClassifier(
        model=surrogate,
        loss=loss,
        input_shape=input_shape,
        nb_classes=2,
        optimizer=optimizer_surr,
        device_type=device_type)
    clip_fun = BAARD_Clipper(detector)
    # clf_loss_multiplier = 1. / 36. if data_name == 'mnist' else 0.1  # 0.1 for CIFAR10

    # Adversarial examples come from the same benign set
    lbl_fp_2, lbl_fp_3 = baard_no_s1.detect(X_wb_test, y_wb_test, per_stage=True)
    fpr_2 = np.mean(lbl_fp_2)
    fpr_3 = np.mean(lbl_fp_3)  # whitebox attack requires FPR from BAARD
    lbl_fp_surr = predict_numpy(surrogate, X_wb_test, device)
    fpr_surr = np.mean(lbl_fp_surr)
    print('[SURROGATE] similarity B/T surr & BAARD (benign):', np.mean(lbl_fp_3 == lbl_fp_surr))

    accs_no_def = []  # Without defence
    rejects_2 = []  # Rejected by Stage 2
    rejects_3 = []  # Rejected by Stage 3
    rejects_all = []  # Rejected by BAARD
    accs_on_advx_2 = []  # Accuracy on adversarial examples after Stage 2
    accs_on_advx_3 = []  # Accuracy on adversarial examples after Stage 3
    accs_on_surr = []  # Accuracy on adversarial examples for surrogate model
    similarities = []  # Matched results between surrogate and BAARD on advx
    for e in epsilons:
        print('\n[WHITEBOX] Running whitebox attack eps={}...'.format(e))
        path_wb_adv = os.path.join(path_wb_data, '{}_{}_baard_whitebox_adv_{}.npy'.format(data_name, model_name, str(float(e))))
        if os.path.exists(path_wb_adv):
            print('[WHITEBOX] Found advx eps={}: {}'.format(e, path_wb_adv))
            adv_wb_test = np.load(path_wb_adv)
        else:
            print('[WHITEBOX] Start generating {} whitebox advx eps={}...'.format(X_wb_test.shape[0], e))
            start = time.time()
            attack = AutoProjectedGradientDescentDetectors(
                estimator=art_classifier,
                detector=art_detector,
                detector_th=0.,
                detector_clip_fun=clip_fun,
                norm=2,
                eps=e,
                eps_step=0.9,
                max_iter=100,
                batch_size=BATCH_SIZE,
                loss_type='logits_difference',
                verbose=False)
            adv_wb_test = attack.generate(x=X_wb_test, y=None)
            time_elapsed = time.time() - start
            print('[WHITEBOX] Time spend on generating {} advx: {}'.format(len(adv_wb_test), str(datetime.timedelta(seconds=time_elapsed))))
            np.save(path_wb_adv, adv_wb_test)
            print('[WHITEBOX] Save to', path_wb_adv)

        # Run BAARD
        print('[WHITEBOX] Evaluating whitebox advx...')
        # Without defence
        pred_wb = predict_numpy(model, adv_wb_test, device)
        acc_naked = np.mean(pred_wb == y_wb_test)
        rej_2 = np.mean(baard_no_s1.stages[0].predict(adv_wb_test, pred_wb))
        rej_3 = np.mean(baard_no_s1.stages[1].predict(adv_wb_test, pred_wb))
        lbl_adv_2, lbl_adv_3 = baard_no_s1.detect(adv_wb_test, pred_wb, per_stage=True)
        rej_all = np.mean(lbl_adv_3)
        acc_2 = acc_on_advx(pred_wb, y_wb_test, lbl_adv_2)
        acc_3 = acc_on_advx(pred_wb, y_wb_test, lbl_adv_3)
        lbl_surr = predict_numpy(surrogate, adv_wb_test, device)
        acc_surr = acc_on_advx(pred_wb, y_wb_test, lbl_surr)
        sim_surr = np.mean(lbl_surr == lbl_adv_3)

        print('[WHITEBOX] no defence:', acc_naked)
        print('[WHITEBOX] S2 reject rate:', rej_2)
        print('[WHITEBOX] S3 reject rate:', rej_3)
        print('[WHITEBOX] BAARD reject rate:', rej_all)
        print('[WHITEBOX] acc_on_adv (Stage 3):', acc_3)
        print('[WHITEBOX] fpr (Stage 3):', fpr_3)
        print('[WHITEBOX] acc_on_surr:', acc_surr)
        print('[WHITEBOX] fpr (Surrogate):', fpr_surr)
        print('[WHITEBOX] similarity B/T surr & BAARD:', sim_surr)

        accs_no_def.append(acc_naked)
        rejects_2.append(rej_2)
        rejects_3.append(rej_3)
        rejects_all.append(rej_all)
        accs_on_advx_2.append(acc_2)
        accs_on_advx_3.append(acc_3)
        accs_on_surr.append(acc_surr)
        similarities.append(sim_surr)

    # Save results
    n_eps = len(epsilons)
    data = {
        'data': np.repeat(data_name, n_eps),
        'model': np.repeat(model_name, n_eps),
        'epsilon': np.array(epsilons),
        'acc_no_def': np.array(accs_no_def),
        'reject_by_s2': np.array(rejects_2),
        'reject_by_s3': np.array(rejects_3),
        'reject_by_baard': np.array(rejects_all),
        'acc_on_adv_2': np.array(accs_on_advx_2),
        'fpr_2': np.repeat(fpr_2, n_eps),
        'acc_on_adv_3': np.array(accs_on_advx_3),
        'fpr_3': np.repeat(fpr_3, n_eps),
        'accs_on_surr': np.array(accs_on_surr),
        'similarity': np.array(similarities)}
    df = pd.DataFrame(data)
    path_csv = os.path.join(path_wb_results, '{}_{}_baard_whitebox.csv'.format(data_name, model_name))
    df.to_csv(path_csv)
    print('Save to:', path_csv)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('-e', '--eps', type=float, required=True, nargs='+')
    parser.add_argument('-i', '--idx', type=int, default=0, choices=list(range(len(SEEDS))))
    parser.add_argument('-p', '--param', type=str)
    args = parser.parse_args()
    print('args:', args)

    idx = args.idx
    data = args.data
    epsilons = args.eps
    seed = SEEDS[args.idx]
    print('data:', data)
    print('epsilons:', epsilons)
    print('seed:', seed)

    whitebox_baard(data, epsilons, idx, args.param)

    # Testing
    # whitebox_baard('mnist', [1., 2., 3., 5., 8.], 0)
    # whitebox_baard('cifar10', [0.05, 0.1, 0.5, 1., 2.], 0)

# python ./experiments/whitebox_baard.py -d mnist -i 0 -e 1.0 2.0 3.0 5.0 8.0
# python ./experiments/whitebox_baard.py -d cifar10 -i 0 -e 0.05 0.1 0.5 1.0 2.0
