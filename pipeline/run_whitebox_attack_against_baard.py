import os
import sys

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
print(*sys.path, sep='\n')

import matplotlib.pyplot as plt
import numpy as np
import torch

from art.attacks.evasion.auto_projected_gradient_descent_detectors import AutoProjectedGradientDescentDetectors
from art.classifiers import PyTorchClassifier
from defences.baard import (ApplicabilityStage, BAARDOperator,
                            DecidabilityStage, ReliabilityStage)
from experiments.util import acc_on_adv, set_seeds
from models.mnist import BaseModel
from models.torch_util import predict_numpy
from pipeline.train_surrogate import SurrogateModel, get_pretrained_surrogate
from attacks.bypass_baard import BAARD_Clipper
import json
from defences.util import acc_on_adv

with open(os.path.join('pipeline', 'seeds.json')) as j:
    json_obj = json.load(j)
    SEEDS = json_obj['seeds']

def cmpt_and_save_predictions(model, art_detector, detector, device, x, y,
                              pred_folder, eps):

    pred_folder = pred_folder + "_{:}".format(eps)

    y_pred = predict_numpy(model, x, device)
    pred_sur_det = art_detector.predict(x)
    pred_baard = detector.detect(x, y_pred)

    # Test stage by stage
    reject_s1 = detector.stages[0].predict(x, y_pred)
    reject_s2 = detector.stages[1].predict(x, y_pred)
    reject_s3 = detector.stages[2].predict(x, y_pred)

    print("Show results:")
    print('Acc classifier:', np.mean(y_pred == y))
    print("acc surrogate detector", np.mean(pred_sur_det == y))
    print("acc baard ",np.mean(pred_sur_det == y))
    print("acc on advx sistema completo ", acc_on_adv(y_pred, y,
                                                      pred_baard))
    print('reject_s1', np.mean(reject_s1))
    print('reject_s2', np.mean(reject_s2))
    print('reject_s3', np.mean(reject_s3))

    print("Save predictions")
    np.save(pred_folder + "_{:}".format("y-pred"), y_pred)
    np.save(pred_folder + "_{:}".format("pred-sur-det"), pred_sur_det)
    np.save(pred_folder + "_{:}".format("pred-baard"), pred_baard)
    np.save(pred_folder + "_{:}".format("reject-s1"), reject_s1)
    np.save(pred_folder + "_{:}".format("reject-s2"), reject_s2)
    np.save(pred_folder + "_{:}".format("reject-s3"), reject_s3)

    print("Predictions saved")

def main(seed, dataset_name, clf_name, detector_name, epsilon_lst):
    set_seeds(SEEDS[seed])

    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # Load classifier
    print("load the classifier")
    file_model = os.path.join('result_{:}'.format(seed),
                              '{:}_{:}_model.pt'.format(dataset_name,
                                                        clf_name))
    model = BaseModel(use_prob=False).to(device)
    model.load_state_dict(torch.load(file_model, map_location=device))

    file_data = os.path.join('result_{:}'.format(seed),
                             '{:}_{:}_apgd2_2000.pt'.format(dataset_name,
                                                        clf_name))
    obj = torch.load(file_data)
    X = obj['X']
    y = obj['y']
    adv = obj['adv']

    pred = predict_numpy(model, X, device)
    print('Acc on clean:', np.mean(pred == y))

    pred = predict_numpy(model, adv, device)
    print('Acc on adv (epsilon 2):', np.mean(pred == y))

    # Split data
    X_att_test = X[2000:3000]
    y_att_test = y[2000:3000]

    # Load baard
    print("Load baard")
    file_baard_train = os.path.join(
        'result_{:}'.format(seed), '{:}_{:}_baard_s1_train_data.pt'.format(
                                                        dataset_name,
                                                        clf_name))
    obj = torch.load(file_baard_train)
    X_baard_train_s1 = obj['X_s1']
    X_baard_train = obj['X']
    y_baard_train = obj['y']

    stages = []
    stages.append(ApplicabilityStage(n_classes=10, quantile=1., verbose=False))
    stages.append(ReliabilityStage(n_classes=10, k=10, quantile=1., verbose=False))
    stages.append(DecidabilityStage(n_classes=10, k=100, quantile=1., verbose=False))
    detector = BAARDOperator(stages=stages)

    detector.stages[0].fit(X_baard_train_s1, y_baard_train)
    for stage in detector.stages[1:]:
        stage.fit(X_baard_train, y_baard_train)

    print("load baard's thresholds")
    file_baard_threshold = os.path.join(
        'result_{:}'.format(seed), '{:}_{:}_baard_threshold.pt'.format(
            dataset_name,
                                                        clf_name))

    thresholds = torch.load(file_baard_threshold)['thresholds']
    detector.load(file_baard_threshold)

    print("load the surrogate")
    file_surro = os.path.join('result_{:}'.format(seed),
                              '{:}_{:}_baard_surrogate.pt'.format(
                                  dataset_name,
                                                        clf_name))
    surrogate = get_pretrained_surrogate(file_surro, device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer_clf = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    art_classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        input_shape=(1, 28, 28),
        nb_classes=10,
        optimizer=optimizer_clf
    )

    optimizer_sur = torch.optim.SGD(
        surrogate.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    art_detector = PyTorchClassifier(
        model=surrogate,
        loss=loss,
        input_shape=(1, 28, 28),
        nb_classes=2,
        optimizer=optimizer_sur
    )

    clip_fun = BAARD_Clipper(detector)

    pred_folder = 'result_{:}/predictions_wb_eval/{:}_{:}_{:}'.format(seed,
                                                              dataset_name,
                                                       clf_name, detector_name)

    print("compute prediction for samples at epsilon 0")
    x = X_att_test[:1000]
    y = y_att_test[:1000]

    # compute and save predictions
    cmpt_and_save_predictions(model, art_detector, detector, device, x, y,
                              pred_folder, 0)

    for eps in epsilon_lst:

        print("epsilon ", eps)

        if dataset_name == 'mnist':
            loss_multiplier = 1. / 36.
        else:
            raise ValueError("loss multiplier not defined")

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
        cmpt_and_save_predictions(model, art_detector, detector, device, adv_x,
                                  y, pred_folder, eps)

if __name__ == '__main__':

    dataset_name = 'mnist'
    clf_name = 'dnn'
    detector_name = 'baard'
    seed = 1
    epsilon_lst = [1,2,3,5,8]
    main(seed, dataset_name, clf_name, detector_name, epsilon_lst)
