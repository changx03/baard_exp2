import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
print(*sys.path, sep='\n')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import types

from art.attacks.evasion.auto_projected_gradient_descent_detectors import AutoProjectedGradientDescentDetectors
from art.classifiers import PyTorchClassifier
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from utils import acc_on_advx, set_seeds
from models.torch_util import predict_numpy
from pipeline.train_surrogate import SurrogateModel, get_pretrained_surrogate
from attacks.bypass_baard import BAARD_Clipper
import json
from models.mnist import BaseModel
from models.cifar10 import Resnet
import torch.nn as nn

from defences.magnet import Autoencoder1, Autoencoder2,\
    MagNetAutoencoderReformer, MagNetDetector, MagNetOperator

BATCH_SIZE = 256

with open(os.path.join('pipeline', 'seeds.json')) as j:
    json_obj = json.load(j)
    SEEDS = json_obj['seeds']


def cmpt_and_save_predictions(model, detector, device, x, y,
                              pred_folder, eps):

    pred_folder = pred_folder + "_{:}".format(eps)
    if not os.path.exists(pred_folder):
        path = Path(pred_folder)
        path.mkdir(parents=True, exist_ok=True)
        print('Cannot find folder. Created:', pred_folder)

    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)

    # labels
    pred_magnet = detector.detect(x)[1]

    print("Show results:")
    print('Acc classifier:', np.mean(y_pred == y))
    #print("acc surrogate detector", np.mean(pred_sur_det == y))
    print("acc baard ",np.mean(pred_magnet == y))
    print("acc on advx sistema completo ", acc_on_advx(y_pred, y,
                                                      pred_magnet))

    print("Save predictions")
    np.save(pred_folder + "_{:}".format("y"), y)
    np.save(pred_folder + "_{:}".format("y-pred"), y_pred)
    np.save(pred_folder + "_{:}".format("pred-baard"), pred_magnet)

    print("Predictions saved")


class CLFWrapper(nn.Module):
    """Wrap nn module to add predict.

    Parameters
    ----------
    classifier : torch.nn.Module object
        The classifier is only required when using probability divergence based
        detector.

    batch_size : int, default=512
        Number of samples in each batch.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, classifier, batch_size=512,
                 device='cpu'):

        super(CLFWrapper, self).__init__()
        self.classifier = classifier

        self.__batch_size = batch_size
        self.device = device

    def predict(self, X):

        scores = self.classifier(X)

        #return logits
        return scores

class NNModuleModelWithReformer(nn.Module):
    """Augment a model with a reformer.

    Parameters
    ----------
    classifier : torch.nn.Module object
        The classifier is only required when using probability divergence based
        detector.

    reformer : {MagNetNoiseReformer, MagNetAutoencoderReformer}

    batch_size : int, default=512
        Number of samples in each batch.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, classifier, reformer, batch_size=512,
                 device='cpu'):

        super(NNModuleModelWithReformer, self).__init__()
        self.classifier = classifier
        self.reformer = reformer

        self.__batch_size = batch_size
        self.device = device

    def forward(self, x):
        """Reform the samples and classify them."""

       # print("forward model with reformer")
        #print(x)
        #print(x.shape)
        #print("type x ", type(x))

        # Reform all samples in X
        #print("before reforming ")
        # reformer takes in input a numpy array (and is a tensor)
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy()
        X_reformed = self.reformer.reform(x)
        #print("after reforming ")

        # classify the sample
        #print("before classification")
        # convert to pytorch
        X_reformed = torch.from_numpy(X_reformed)
        scores = self.classifier(X_reformed)
        #print("after classification")

        # scores is a tensor
        scores = scores.detach()
        #print("scores shape ", scores.shape)

        # return the logits
        return scores

class NNModuleMagnetDetector(nn.Module):
    """Magnet detector that inherit from nn.Module

    Parameters
    ----------
    classifier : torch.nn.Module object
        The classifier is only required when using probability divergence based
        detector.

    detector : Magnet detector.

    batch_size : int, default=512
        Number of samples in each batch.

    device : torch.device, default='cpu'
        The device for PyTorch. Using 'cuda' is recommended.
    """

    def __init__(self, detector, batch_size=512,
                 device='cpu'):

        super(NNModuleMagnetDetector, self).__init__()
        self.detector = detector

        self.__batch_size = batch_size
        self.device = device

    def forward(self, x):
        """Make the detector predict if the samples are or not adversarial.
        Return the detector's score.
        A vector n_samples * 2 where the second colum represent the score of
        the malicious class.
        """

#        print("forward NNModuleMagnetDetector")

        # detect advx
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy()

        scores = self.detector.cmpt_detector_scores(x)

        scores = torch.from_numpy(scores)

        # those are the logits (n_samples, 2)
        return scores

# method that we add to the Magnet detector after loading
def cmpt_detector_scores(self, X):
    """Returns the scores in output from the detector.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError('X must be a ndarray.')

    n = X.shape[0]
    X_ae = self.predict(X)
    if self.algorithm == 'error':
        X = X.reshape(n, -1)
        X_ae = X_ae.reshape(n, -1)
        diff = np.abs(X - X_ae)
        scores = np.mean(np.power(diff, self.p), axis=1)
    else:  # self.algorithm == 'prob'
        scores = self.__get_js_divergence(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(X_ae).type(torch.float32))

    # create a binary vector
    scores_array = np.ones((scores.size, 2)) * self.threshold
    scores_array[:,1] = scores[:]

    return scores_array

def loadmagnet(data, model_name, param, device, path, model):

    detectors = []

    if data == 'mnist':
        n_channel = 1
        detectors.append(MagNetDetector(
            encoder=Autoencoder1(n_channel=n_channel),
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=1,
            device=device))

        detectors.append(MagNetDetector(
            encoder=Autoencoder2(n_channel),
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=2,
            device=device))

    elif data == 'cifar10':
        n_channel = 3
        autoencoder = Autoencoder2(n_channel=n_channel)
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=2,
            device=device))
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='prob',
            temperature=10,
            device=device))
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='prob',
            temperature=40,
            device=device))
    else:
        raise NotImplementedError

    # Load existing autoencoder or train new one
    for i, d in enumerate(detectors, start=1):
        file_ae = os.path.join(path, '{}_{}_magnet_autoencoder_{}.pt'.format(data, model_name, str(i)))
        if os.path.exists(file_ae):
            print('Found existing MagNet autoencoder:', file_ae)
            d.load(file_ae)
        else:
            raise ValueError("Magnet not found ")

    reformer = MagNetAutoencoderReformer(
        encoder=detectors[0].encoder,
        batch_size=BATCH_SIZE,
        device=device)

    # add predict function to the model
    loss = torch.nn.CrossEntropyLoss()
    pytorch_classifier = CLFWrapper(
        classifier=model,
    )

    full_magnet = MagNetOperator(
        classifier=pytorch_classifier,
        detectors=detectors,
        reformer=reformer,
        batch_size=BATCH_SIZE,
        device=device)

    print("detector created ")

    # bind a new method to the detector object
    full_magnet.detectors[-1]\
        .cmpt_detector_scores = types.MethodType(cmpt_detector_scores,
                                                 full_magnet.detectors[-1])
    print("new function binded to the detector object")

    # both the following objects inherith form

    # augment a model with the reformer
    model_with_reformer_nn_module = NNModuleModelWithReformer(
        classifier=model,
        reformer=reformer,
        batch_size=BATCH_SIZE,
        device=device)

    full_detector_nn_module = NNModuleMagnetDetector(
        detector = full_magnet.detectors[-1],
        batch_size=BATCH_SIZE,
        device=device)

    return model_with_reformer_nn_module, full_detector_nn_module, full_magnet


def main(seed, dataset_name, clf_name, detector_name, epsilon_lst,
         input_shape, json_param, path):
    set_seeds(SEEDS[seed])

    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # Load classifier
    print("load the classifier")
    file_model = os.path.join('result_{:}'.format(seed),
                              '{:}_{:}_model.pt'.format(dataset_name,
                                                        clf_name))
    if clf_name == 'dnn':
        model = BaseModel(use_prob=False).to(device)
    elif clf_name == 'resnet':
        model = Resnet(use_prob=False).to(device)
    else:
        raise ValueError("model idx unknown")
    model.load_state_dict(torch.load(file_model, map_location=device))

    file_data = os.path.join('result_{:}'.format(seed),
                             '{:}_{:}_apgd2_2000.pt'.format(dataset_name,
                                                        clf_name))
    obj = torch.load(file_data)
    X = obj['X']
    y = obj['y']
    adv = obj['adv']

    print("undefended model acc")
    pred = predict_numpy(model, X, device)
    print('Acc on clean:', np.mean(pred == y))

    # Split data
    X_att_test = X[2000:3000].astype(np.float32)
    y_att_test = y[2000:3000].astype(np.float32)

    print("x attr shape ", X_att_test.shape)

    #################################################################
    print("Load Magnet")
    with open(json_param) as j:
        param = json.load(j)

    print("before load magnet")
    model_with_reformer_nn_module, detector_nn_module, full_magnet_orig  = \
        loadmagnet(dataset_name, clf_name,param, device,path, model)

    print("Magnet loaded")

    loss = torch.nn.CrossEntropyLoss()
    # this one return the logits
    art_classifier = PyTorchClassifier(
        model=model_with_reformer_nn_module,
        loss=loss,
        input_shape=input_shape,
        nb_classes=10,
        optimizer=None
    )

    y_pred = model_with_reformer_nn_module(X)
    print("model_with_reformer_nn_module", y_pred.shape)

    y_pred = art_classifier.predict(X)
    print("art_classifier",y_pred.shape )

    print("check full magnet ")
    _, y_pred = full_magnet_orig.detect(X)
    print("full magnet", y_pred.shape)

    print("check detector nn module")
    # correcly return an array with the logits
    y_pred = detector_nn_module(X)
    print("y pred ", y_pred)
    print("detector_nn_module", y_pred.shape)


    print("create pytorch detector")
    # must be only the detector
    art_detector = PyTorchClassifier(
        model=detector_nn_module,
        loss=loss,
        input_shape=input_shape,
        nb_classes=2,
        optimizer=None
    )

    print("check art detector")
    y_pred = art_detector.predict(X)
    print("detector_nn_module", y_pred.shape)
    print("art detector ok")

    clip_fun = None
    #################################################################

    pred_folder = 'result_{:}/predictions_wb_eval/{:}_{:}_{:}'.format(seed,
                                                              dataset_name,
                                                       clf_name, detector_name)

    print("compute prediction for samples at epsilon 0")
    x = X_att_test[:1000]
    y = y_att_test[:1000]

    # compute and save predictions
    cmpt_and_save_predictions(art_classifier, full_magnet_orig, device, x, y,
                              pred_folder, 0)

    for eps in epsilon_lst:

        print("epsilon ", eps)

        if dataset_name == 'mnist':
            loss_multiplier = 1. / 36.
        else:
            loss_multiplier = 0.1

        attack = AutoProjectedGradientDescentDetectors(
            estimator=art_classifier,
            detector=art_detector,
            detector_th=0,
            clf_loss_multiplier=loss_multiplier,
            detector_clip_fun=clip_fun,
            loss_type='logits_difference',
            batch_size=128,
            norm=2,
            eps=eps,
            eps_step=0.9,
            beta=0.5,
            max_iter=100)

        adv_x = attack.generate(x=x, y=None)

        # compute and save predictions
        cmpt_and_save_predictions(art_classifier,  full_magnet_orig, device, adv_x,
                                  y, pred_folder, eps)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist', choices=[
        'mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='dnn', choices=['dnn',
                                                                   'resnet'])
    parser.add_argument('--i', type=int, default=1, choices=list(range(len(
        SEEDS))))
    path_json_magnet = os.path.join('params', 'magnet_param.json')
    parser.add_argument('--json', type=str, default=path_json_magnet)

    args = parser.parse_args()
    print(args)

    dataset_name = args.data
    clf_name = args.model
    seed = args.i

    detector_name = 'magnet'

    if args.data == 'mnist':
        clf_name = 'dnn'
        epsilon_lst = [1,2,3,5,8]
        input_shape = (1, 28, 28)
    else:
        clf_name = 'resnet'
        detector_name = 'baard'
        epsilon_lst = [0.05, 0.1, 0.2, 1, 2]
        input_shape = (3, 32, 32)

    main(seed, dataset_name, clf_name, detector_name, epsilon_lst,
         input_shape, json_param=args.json, path='result_{:}'.format(str(
            seed)))
