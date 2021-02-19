"""
Evalute BAARD on white-box surrogate attacks.
"""
import os
import sys

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.datasets as datasets
from models.cifar10 import Resnet
from models.mnist import BaseModel
from models.torch_util import predict_numpy, validate
from torch.utils.data import DataLoader, TensorDataset
from utils import acc_on_advx, get_correct_examples, mkdir, set_seeds

from experiments import (get_advx_untargeted, get_baard, get_output_path,
                         pytorch_train_classifier)
from experiments.train_baard_surrogate import SurrogateModel, train_surrogate

DATA_PATH = 'data'
with open('metadata.json') as data_json:
    METADATA = json.load(data_json)
with open('SEEDS') as f:
    SEEDS = [int(s) for s in f.read().split(',')]
BATCH_SIZE = 192
EPOCHS = 200  # Maybe overkill, but the surrogate model is a very small.
DEF_NAME = 'baard'
ATT_SURR = 'apgd2'  # the attack for training surrogate model.
# NOTE: We combine attacks with different epsilons when training the surrogate
# model. The goal is to obtain a good training set, so the surrogate model can
# behave as similar as BAARD.
EPS_SURR_MNIST = [1., 2., 5., 8.]
EPS_SURR_CIFAR10 = [0.1, 0.5, 1., 2.]
ATT_TEST = 'apgd2'  # This is for testing the surrogate model.
EPS_TEST = 2.


def surrogate_sim(data_name, epsilons, idx, baard_param=None):
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
    # X_wb_test = X_test[2000: 4000]  # test set for whitebox attacks
    # y_wb_test = y_test[2000: 4000]
    X_surr_train = X_test[4000: 6000]  # training set for surrogate model
    y_surr_train = y_test[4000: 6000]
    # print('[DATA] After split:', X_att.shape, X_val.shape, X_wb_test.shape, X_surr_train.shape)

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

    ############################################################################
    # Step 5: Build surrogate train set (Train adversarial examples)
    # Save X
    path_X_surr_train = os.path.join(path_wb_data, '{}_{}_X_surro_train.npy'.format(data_name, model_name))
    path_y_surr_train = os.path.join(path_wb_data, '{}_{}_y_surro_train.npy'.format(data_name, model_name))
    # Check both files are exist.
    if os.path.exists(path_X_surr_train) and os.path.exists(path_y_surr_train):
        print('[SURROGATE] Found surrogate train set:', path_X_surr_train)
        X_surr_train = np.load(path_X_surr_train)
        y_surr_train = np.load(path_y_surr_train)
    else:
        np.save(path_X_surr_train, X_surr_train)
        np.save(path_y_surr_train, y_surr_train)
        print('[SURROGATE] Save surrogate train set to:', path_X_surr_train)

    surr_epsilons = EPS_SURR_MNIST if data_name == 'mnist' else EPS_SURR_CIFAR10
    # surr_epsilons = epsilons  # Using the same set of epsilons as whitebox attacks
    print('[SURROGATE] Epsilon for train set:', surr_epsilons)
    adv_surr_train = []
    for i, e in enumerate(surr_epsilons):
        # This is a slow process, we save each segment.
        path_adv_surr_train = os.path.join(
            path_wb_data, '{}_{}_adv_surro_train_{}_{}.npy'.format(data_name, model_name, ATT_SURR, str(float(e))))
        if os.path.exists(path_adv_surr_train):
            print('[SURROGATE] Found surrogate advx:', path_adv_surr_train)
            adv = np.load(path_adv_surr_train)
        else:
            print('[SURROGATE] Start training {} {} eps={}...'.format(X_surr_train.shape[0], ATT_SURR, e))
            adv = get_advx_untargeted(
                model,
                data_name,
                ATT_SURR,
                eps=e,
                device=device,
                X=X_surr_train,
                batch_size=BATCH_SIZE)
            np.save(path_adv_surr_train, adv)
            print('[SURROGATE] Save surrogate advx to:', path_adv_surr_train)
        adv_surr_train.append(adv)

        # pred = predict_numpy(model, adv, device)
        # lbl_adv = detector.detect(adv, pred, device)
        # acc = acc_on_advx(pred, y_surr_train, lbl_adv)
        # print('[DEBUG] acc:', acc)

    # Combine subsets into one
    adv_surr_train = np.concatenate(adv_surr_train)
    # Duplicate benign set to avoid imbalanced data
    X_surr_train = np.vstack([X_surr_train] * len(surr_epsilons))
    assert X_surr_train.shape == adv_surr_train.shape
    assert np.all(X_surr_train[0] == X_surr_train[2000])

    n_benign = len(X_surr_train)
    # The training set contains X = benign + advx, y = baard(benign) + baard(advx).
    X_surr_train = np.concatenate((X_surr_train, adv_surr_train))
    pred_surr_train = predict_numpy(model, X_surr_train, device)
    y_surr_train = np.tile(y_surr_train, len(surr_epsilons) * 2)
    assert np.all(y_surr_train[0:100] == y_surr_train[2000:2100])
    print('[SURROGATE] X_train, pred_train, y_train:', X_surr_train.shape, pred_surr_train.shape, y_surr_train.shape)
    print('[SURROGATE] Classifier acc. on advx (train):  ', np.mean(pred_surr_train[n_benign:] == y_surr_train[n_benign:]))
    print('[SURROGATE] Classifier acc. on benign (train):', np.mean(pred_surr_train[:n_benign] == y_surr_train[:n_benign]))

    path_lbl_surr_train = os.path.join(path_wb_data, '{}_{}_label_surro_train.npy'.format(data_name, model_name))
    if os.path.exists(path_lbl_surr_train):
        print('[SURROGATE] Found lbls for surr train set:', path_lbl_surr_train)
        lbl_surr_train = np.load(path_lbl_surr_train)
    else:
        print('[SURROGATE] BAARD is labelling surrogate train set...')
        lbl_surr_train = detector.detect(X_surr_train, pred_surr_train)
        np.save(path_lbl_surr_train, lbl_surr_train)
        print('[SURROGATE] Save surrogate train labels to:', path_lbl_surr_train)

    # Only 2nd half contains advx
    acc = acc_on_advx(pred_surr_train[n_benign:], y_surr_train[n_benign:], lbl_surr_train[n_benign:])
    fpr = np.mean(lbl_surr_train[:n_benign])
    print('[DEFENCE] BAARD acc on surrogate train set:', acc)
    print('[DEFENCE] BAARD fpr on surrogate train set:', fpr)

    ############################################################################
    # Step 6: Train surrogate model
    # Load a test set
    path_adv = os.path.join(path_data, '{}_{}_{}_{}_adv.npy'.format(data_name, model_name, ATT_TEST, str(float(EPS_TEST))))
    if os.path.exists(path_adv):
        print('[ATTACK] Find:', path_adv)
        adv_test = np.load(path_adv)
    else:
        print('[ATTACK] Start generating {} {} eps={} advx...'.format(X_att.shape[0], ATT_TEST, EPS_TEST))
        adv_test = get_advx_untargeted(
            model,
            data_name,
            ATT_TEST,
            eps=EPS_TEST,
            device=device,
            X=X_att,
            batch_size=BATCH_SIZE)
        np.save(path_adv, adv_test)
        print('[ATTACK] Save to', path_adv)
    X_surr_test = np.concatenate((X_att, adv_test))
    pred_surr_test = predict_numpy(model, X_surr_test, device)
    lbl_surr_test = detector.detect(X_surr_test, pred_surr_test)
    print('[SURROGATE] Classifier acc. on advx (test):  ', np.mean(pred_surr_test[len(y_att):] == y_att))
    print('[SURROGATE] Classifier acc. on benign (test):', np.mean(pred_surr_test[:len(y_att)] == y_att))

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
        surrogate = train_surrogate(
            X_train=X_surr_train,
            X_test=X_surr_test,
            y_train=lbl_surr_train,
            y_test=lbl_surr_test,
            epochs=EPOCHS,
            device=device)
        torch.save(surrogate.state_dict(), path_surr_model)
        print('[SURROGATE] Save surrogate model to:', path_surr_model)

    pred = predict_numpy(surrogate, X_surr_test, device)
    acc = np.mean(pred == lbl_surr_test)
    print('[SURROGATE] Test set acc on surrogate model:', acc)

    ############################################################################
    # Step 8: Perform white-box attacks
    # Define validation set
    X = X_val
    y = y_val
    adv = X.copy()
    pred = predict_numpy(model, X, device)
    assert np.all(pred == y)

    # Adversarial examples come from the same benign set
    lbl_fp_1, lbl_fp_2, lbl_fp_3 = detector.detect(X, pred, per_stage=True)
    fpr_1 = np.mean(lbl_fp_1)
    fpr_2 = np.mean(lbl_fp_2)
    fpr_3 = np.mean(lbl_fp_3)  # whitebox attack requires FPR from BAARD
    lbl_fp_surr = predict_numpy(surrogate, X, device)
    fpr_surr = np.mean(lbl_fp_surr)
    sim_benign = np.mean(lbl_fp_3 == lbl_fp_surr)
    print('[SURROGATE] similarity B/T surr & BAARD (benign):', sim_benign)

    accs_no_def = []  # Without defence
    accs_on_advx_1 = []  # Accuracy on adversarial examples after Stage 1
    accs_on_advx_2 = []  # Accuracy on adversarial examples after Stage 2
    accs_on_advx_3 = []  # Accuracy on adversarial examples after Stage 3
    accs_on_surr = []  # Accuracy on adversarial examples for surrogate model
    similarities = []  # Matched results between surrogate and BAARD on advx
    for e in epsilons:
        print('\n[SURROGATE] Start generating {} advx eps={}...'.format(X.shape[0], e))
        adv = get_advx_untargeted(
            model,
            data_name,
            ATT_SURR,
            eps=e,
            device=device,
            X=X,
            batch_size=BATCH_SIZE)

        # Run BAARD
        print('[SURROGATE] Evaluating surrogate model...')
        # Without defence
        pred_adv = predict_numpy(model, adv, device)
        acc_naked = np.mean(pred_adv == y)
        lbl_adv_1, lbl_adv_2, lbl_adv_3 = detector.detect(adv, pred, per_stage=True)
        acc_1 = acc_on_advx(pred_adv, y, lbl_adv_1)
        acc_2 = acc_on_advx(pred_adv, y, lbl_adv_2)
        acc_3 = acc_on_advx(pred_adv, y, lbl_adv_3)
        lbl_surr = predict_numpy(surrogate, adv, device)
        acc_surr = acc_on_advx(pred_adv, y, lbl_surr)
        sim_surr = np.mean(lbl_surr == lbl_adv_3)

        print('[SURROGATE] no defence:', acc_naked)
        print('[SURROGATE] acc_on_adv (Stage 3):', acc_3)
        print('[SURROGATE] fpr (Stage 3):', fpr_3)
        print('[SURROGATE] acc_on_surr:', acc_surr)
        print('[SURROGATE] fpr (Surrogate):', fpr_surr)
        print('[SURROGATE] similarity B/T surr & BAARD:', sim_surr)

        accs_no_def.append(acc_naked)
        accs_on_advx_1.append(acc_1)
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
        'acc_on_adv_1': np.array(accs_on_advx_1),
        'fpr_1': np.repeat(fpr_1, n_eps),
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

    surrogate_sim(data, epsilons, idx, args.param)

    # Testing
    # surrogate_sim('mnist', [1., 2., 3., 5., 8.], 0)
    # surrogate_sim('cifar10', [0.1, 0.5, 1., 2., 5.], 0)

# python ./diagnosis/surrogate_sim.py -d mnist -i 0 -e 1.0 2.0 3.0 5.0 8.0
# python ./diagnosis/surrogate_sim.py -d cifar10 -i 0 -e 0.1 .5 1.0 2.0 5.0
