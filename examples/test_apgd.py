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
from utils import acc_on_advx, set_seeds
from models.mnist import BaseModel
from models.torch_util import predict_numpy
from pipeline.train_surrogate import SurrogateModel, get_pretrained_surrogate
from attacks.bypass_baard import BAARD_Clipper

SEED = 65558  # for result_0


def main():
    set_seeds(SEED)

    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # Load classifier
    file_model = os.path.join('result_0', 'mnist_dnn_model.pt')
    model = BaseModel(use_prob=False).to(device)
    model.load_state_dict(torch.load(file_model, map_location=device))

    file_data = os.path.join('result_0', 'mnist_dnn_apgd2_3000.pt')
    obj = torch.load(file_data)
    X = obj['X']
    y = obj['y']
    adv = obj['adv']

    pred = predict_numpy(model, X, device)
    print('Acc on clean:', np.mean(pred == y))

    pred = predict_numpy(model, adv, device)
    print('Acc on adv:', np.mean(pred == y))

    # Split data
    X_def_test = X[:1000]
    y_def_test = y[:1000]
    adv_def_test = adv[:1000]
    pred_adv_def_test = pred[:1000]

    X_def_val = X[1000:2000]
    y_def_val = y[1000:2000]
    adv_def_val = adv[1000:2000]
    pred_adv_def_val = pred[1000:2000]

    X_att_test = X[2000:4000]
    y_att_test = y[2000:4000]
    adv_att_test = adv[2000:4000]
    pred_adv_att_test = pred[2000:4000]

    X_surro_train = X[4000:]
    y_surro_train = y[4000:]
    adv_surro_train = adv[4000:]
    pred_adv_surro_train = pred[4000:]

    # Load baard
    file_baard_train = os.path.join(
        'result_0', 'mnist_dnn_baard_s1_train_data.pt')
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

    file_baard_threshold = os.path.join(
        'result_0', 'mnist_dnn_baard_threshold.pt')
    thresholds = torch.load(file_baard_threshold)['thresholds']
    detector.load(file_baard_threshold)

    file_surro = os.path.join('result_0', 'mnist_dnn_baard_surrogate.pt')
    surrogate = get_pretrained_surrogate(file_surro, device)

    # Test surrogate model
    X_test = np.concatenate((X_att_test[1000:], adv_att_test[1000:]))
    pred_test = predict_numpy(model, X_test, device)
    label_test = detector.detect(X_test, pred_test)
    acc = acc_on_advx(pred_test[1000:], y_att_test[1000:], label_test[1000:])
    fpr = np.mean(label_test[:1000])
    print('BAARD Acc_on_adv:', acc)
    print('BAARD FPR:', fpr)

    label_surro = predict_numpy(surrogate, X_test, device)
    acc = np.mean(label_surro == label_test)
    print('Acc on surrogate:', acc)

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

    loss_multiplier = 1. / 36.
    clip_fun = BAARD_Clipper(detector)

    attack = AutoProjectedGradientDescentDetectors(
        estimator=art_classifier,
        detector=art_detector,
        detector_th=0, #fpr,
        clf_loss_multiplier=loss_multiplier,
        detector_clip_fun=clip_fun,
        loss_type='logits_difference',
        batch_size=128,
        norm=2,
        eps=8.0,
        eps_step=0.9,
        beta=0.5,
        max_iter=100)

    # X_toy = np.random.rand(128, 1, 28, 28).astype(np.float32)
    # pred_toy = art_classifier.predict(X_toy)
    # rejected_s1 = detector.stages[0].predict(X_toy, pred_toy)
    # print('Without:', np.mean(rejected_s1))

    # X_clipped = clip_fun(X_toy, art_classifier)
    # rejected_s1 = detector.stages[0].predict(X_clipped, pred_toy)
    # print('With:', np.mean(rejected_s1))
    # adv_x = attack.generate(x=X_toy)
    # pred_adv = predict_numpy(model, adv_x, device)
    # pred_sur = art_detector.predict(adv_x)
    # print('From surrogate model:', np.mean(pred_sur == 1))
    # labelled_as_adv = detector.detect(adv_x, pred_adv)
    # print('From BAARD', np.mean(labelled_as_adv == 1))

    # # Test it stage by stage
    # reject_s1 = detector.stages[0].predict(adv_x, pred_adv)
    # print('reject_s1', np.mean(reject_s1))
    # reject_s2 = detector.stages[1].predict(adv_x, pred_adv)
    # print('reject_s2', np.mean(reject_s2))
    # reject_s3 = detector.stages[2].predict(adv_x, pred_adv)
    # print('reject_s3', np.mean(reject_s3))

    x = X_att_test[:10]
    y = y_att_test[:10]
    adv_x = attack.generate(x=x, y=None)
    pred_adv = predict_numpy(model, adv_x, device)
    pred_sur = art_detector.predict(adv_x)

    pred = predict_numpy(model, adv_x, device)
    print('Acc classifier:', np.mean(pred == y))

    print('From surrogate model:', np.mean(pred_sur == 1))
    labelled_as_adv = detector.detect(adv_x, pred_adv)
    print('From BAARD', np.mean(labelled_as_adv == 1))

    # Test it stage by stage
    reject_s1 = detector.stages[0].predict(adv_x, pred_adv)
    print('reject_s1', np.mean(reject_s1))
    reject_s2 = detector.stages[1].predict(adv_x, pred_adv)
    print('reject_s2', np.mean(reject_s2))
    reject_s3 = detector.stages[2].predict(adv_x, pred_adv)
    print('reject_s3', np.mean(reject_s3))
    print()


if __name__ == '__main__':
    main()
