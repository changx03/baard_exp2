"""
This script show the decision region of a classifier augmented with the
detector and the adversarial example found by the apgd attack.
"""
import os
import sys
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
print("sys.path ", sys.path)
from acc_on_advx import acc_on_advx

from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import make_classification, make_blobs
#from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.auto_projected_gradient_descent_detectors import \
    AutoProjectedGradientDescentDetectors
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import random

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    """
    Model with input size (-1, 5) for blobs dataset
    with 5 features
    """
    input_shape = (2,)

    def __init__(self, n_features, n_classes):
        """Example network."""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def dataset_creation_blobs():

    n_samples_tr = 500  # number of training set samples
    n_samples_ts = 100  # number of testing set samples

    patterns, labels = make_blobs(
                        n_samples=n_samples_tr + n_samples_ts,
                        n_features=2,
                        cluster_std = 0.5,
                        centers = ((1,4),(2,1)),
                        random_state=0)

    patterns = patterns.astype(np.float32)
    labels = labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split( patterns, labels,
                                                         test_size=0.33,
                                                         random_state=42)

    return X_train, X_test, y_train, y_test

def net_generation_and_train(X_train, y_train, net_type):

    loss = nn.CrossEntropyLoss()
    net = net_type(n_features=2, n_classes=2)
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1, momentum=0.9)

    clf = PyTorchClassifier(model=net, loss=loss, input_shape=(1,2),
                                nb_classes=2, optimizer=optimizer)
    clf.fit(X_train, y_train, batch_size=10, nb_epochs=100)

    return clf

def plot_decision_region(X, clf):

    plot_step = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # chose the colors depending on the classifier predictions
    points_to_classify = np.c_[xx.ravel(), yy.ravel()]

    points_to_classify = points_to_classify.astype(np.float32)

    scores = clf.predict(points_to_classify)
    Z = scores.argmax(axis=1)

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)


def plot_detector_decision_region(X, clf, detector):

    colors = ['blue', 'red', 'gray', 'black', 'green', 'cyan']
    cmap = colors[:3]
    # Convert list of colors to colormap
    cmap = ListedColormap(cmap)

    plot_step = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # chose the colors depending on the classifier predictions
    points_to_classify = np.c_[xx.ravel(), yy.ravel()]

    points_to_classify = points_to_classify.astype(np.float32)

    clf_scores = clf.predict(points_to_classify)
    Z = clf_scores.argmax(axis=1)

    det_scores = detector.predict(points_to_classify)
    det_pred = det_scores.argmax(axis=1)

    #set the prediction for the samples rejected by the detector equal to
    #class 2
    Z[det_pred==1] = 2

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha = 0.5)

def plot_clf_decision_region(X, clf):

    colors = ['blue', 'red', 'gray', 'black', 'green', 'cyan']
    cmap = colors[:3]
    # Convert list of colors to colormap
    cmap = ListedColormap(cmap)

    plot_step = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # chose the colors depending on the classifier predictions
    points_to_classify = np.c_[xx.ravel(), yy.ravel()]

    points_to_classify = points_to_classify.astype(np.float32)

    clf_scores = clf.predict(points_to_classify)
    Z = clf_scores.argmax(axis=1)

    Z = Z.reshape(xx.shape)
    #cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha = 0.5)
    cs = plt.contour(xx, yy, Z, colors='black', alpha=0.5, levels=[0])


def _apgd_grad(x, y, attack):

    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)

    grad_clf = attack.estimator.loss_gradient(x, y) * \
            (1 - 2 * int(attack.targeted))

    # normalize the gradient for visualization
    norm = np.linalg.norm(grad_clf)
    if norm > 0:
        grad_clf /= norm

    return grad_clf

def _apgd_obj_func(x, y, attack):

    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)

    loss = attack.estimator.loss(x=x, y=y,
                        reduction="none")
    loss = loss

    return loss


def _apgd_det_grad(x, y, attack):

    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)

    grad = attack._cmpt_grad(x, y)

    return grad

def _apgd_det_obj_func(x, y, attack):
    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)

    loss = attack._cmpt_loss_func(x, y)

    return loss


def plot_attacker_objective_function(X, attack,attack_obf_fun,grad_obj_fun):

    plot_clf_decision_region(X, clf)

    #plot_step = 0.02
    plot_step = 0.3

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #fixme: remove
    x_min = -3
    x_max = 8
    y_min = -4
    y_max = 8

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # chose the colors depending on the classifier predictions
    points_to_classify = np.c_[xx.ravel(), yy.ravel()]
    points_to_classify = points_to_classify.astype(np.float32)

    n_vals = points_to_classify.shape[0]
    Y = np.zeros((n_vals, 2))
    Y[:,1] = 1

    Z = np.zeros((n_vals))
    for p_idx in range(n_vals):
        Z[p_idx] = attack_obf_fun(points_to_classify[p_idx,:], Y[p_idx,:],
                                      attack)

    print("z min  ", Z.min(), "z max ", Z.max())

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.seismic, alpha = 0.5)

    # plot gradients
    grad_point_values = np.zeros((n_vals, 2))
    # compute gradient on each grid point
    for p_idx in range(n_vals):
        grad_point_values[p_idx, :] = grad_obj_fun(
            points_to_classify[p_idx, :], Y[p_idx,:], attack)

    U = grad_point_values[:, 0].reshape(
        (xx.shape[0], xx.shape[1]))
    V = grad_point_values[:, 1].reshape(
        (xx.shape[0], xx.shape[1]))

    plt.quiver(xx, yy, U, V)

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)


def plot_points(X, y, n_classes, fixed_color = None, edgecolor=None):
    """
    If fixed color is not None plots all the samples using the "fixed_color".
    """
    plot_colors = ['b', 'r']

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):

        if fixed_color is not None:
            color = fixed_color
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color,
                    cmap=plt.cm.RdYlBu, edgecolor=edgecolor, s=15)


def evaluate_attack_efficacy(adv_x):

    # show the detector's decision region and the generated adversarial samples.
    plot_detector_decision_region(X_tr_det, clf, detector)

    # plot dataset samples
    plot_points(X_train, y_train, 2)

    # plot dummy adversarial samples
    plot_points(dummy_adv_x, np.ones((dummy_adv_x.shape[0],)), 2, 'gray')

    # plot adversarial examples
    plot_points(adv_x, y_test, 2, fixed_color=None,
                edgecolor='black')

# todo: readd
#    plt.xlim((-1, 3))
#    plt.ylim((0, 5))

    plt.show()

    scores = clf.predict(adv_x)
    y_pred = scores.argmax(axis=1)
    scores = detector.predict(adv_x)
    detected_as_advx = scores.argmax(axis=1)
    acc = acc_on_advx(y_pred, y_test, detected_as_advx)
    print("accuracy on the apgd advx", acc)


def show_attackers_obj_function(X_test, attack, adv_x, attack_obf_fun,
                                grad_obj_fun):

    plot_attacker_objective_function(X_test, attack, attack_obf_fun,
                                     grad_obj_fun)

    # plot dataset samples
    plot_points(X_train, y_train, 2)

    # plot dummy adversarial samples
    plot_points(dummy_adv_x, np.ones((dummy_adv_x.shape[0],)), 2, 'gray')

    # plot adversarial examples
    plot_points(adv_x, y_test, 2, fixed_color=None,
                edgecolor='black')

    #fixme: re-add
   # plt.xlim((-1, 3))
   # plt.ylim((0, 5))
    #
    plt.show()

    scores = clf.predict(adv_x)
    y_pred = scores.argmax(axis=1)
    scores = detector.predict(adv_x)
    detected_as_advx = scores.argmax(axis=1)
    acc = acc_on_advx(y_pred, y_test, detected_as_advx)
    print("accuracy on the apgd advx", acc)

set_seeds(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

X_train, X_test, y_train, y_test = dataset_creation_blobs()

# create and train the classifier
clf = net_generation_and_train(X_train, y_train, net_type=Net)

# generate dummy adversarial examples
dummy_adv_x, _ = make_blobs(n_samples=100,
                        n_features=2,
                        cluster_std = 0.50,
                        centers = ((-1,1),(1,2),(2,2),(4,3)),
                        random_state=0)

# train a detector
# create a dataset that contains the cleean sample (with label 0) + the
# adversarial examples (with label 1)
X_tr_det = np.append(X_test, dummy_adv_x, axis=0).astype(np.float32)
y_tr_det = np.append(np.zeros((y_test.shape)), np.ones((
    y_test.shape))).astype(np.int)
detector = net_generation_and_train(X_tr_det, y_tr_det, net_type=Net)

###############################

def clip_fun(x, y):
    clip_min = np.array([-1.,-1,]).astype(np.float32)
    clip_max = np.array([5.,5]).astype(np.float32)
    x = np.clip(x, clip_min, clip_max)
    return x

attack_class = 'apgd_detector'# can be apgd or apgd_detector

if attack_class == "apgd":
    #attack the detector with apgd (black box attack)
    attack = AutoProjectedGradientDescent(estimator=clf, norm=2, eps=3.0,
                                          #loss_type='cross_entropy',
                                          loss_type ='logits_difference',
                                          eps_step = 0.5, # 0.5
                                          )

    attack_obf_fun = _apgd_obj_func
    grad_obj_fun = _apgd_grad

else:

    attack_obf_fun = _apgd_det_obj_func
    grad_obj_fun = _apgd_det_grad

    # attack the detector with apgd detector (white box against detector)
    attack = AutoProjectedGradientDescentDetectors(estimator = clf,
                                                   detector=detector,
                                                   detector_clip_fun=clip_fun,
                                                   norm=2,
                                                   eps = 5.0,
                                                   eps_step=0.9,
                                                   beta=0.005,
                                                   max_iter=100,
                                                   loss_type=
                                                   #'cross_entropy',
                                                    'logits_difference',
                                                   clf_loss_multiplier=0.0001,
                                                   # clf_loss_multiplier
                                                   # chosen to bring the
                                                   # loss in 0 1
                                                   detector_th=0.0)

# attack the detector
adv_x = attack.generate(x=X_test, y=y_test)

#plt.subplot(1, 2, 1)
#evaluate_attack_efficacy(adv_x)

#plt.subplot(2, 2, 2)
print("show attacker objective function")
show_attackers_obj_function(X_test, attack, adv_x, attack_obf_fun,
 grad_obj_fun)

##################################################
# from art.estimators.classification import DetectorClassifier
#
# class BinaryNetDetector(PyTorchClassifier):
#
#     def __init__(self, net_detector):
#         self.net_detector = net_detector
#         self._input_shape = net_detector.input_shape
#         self._nb_classes = net_detector.nb_classes
#
#     def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
#         """
#         Return the scores wrt the malicious class
#         """
#         return self.net_detector.predict(x)[:,1]
#
# class ClfAugmentedWithBinaryNetDetector(DetectorClassifier):
#
#     def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
#         pass
#
#     def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
#         pass
#
# binary_net = BinaryNetDetector(clf)
# augmented_clf = DetectorClassifier(classifier=clf, detector=binary_net)
#
# attack = AutoProjectedGradientDescent(estimator = augmented_clf,
#                                                 norm=2,
#                                                 eps = 3.0)
# evaluate_attack_efficacy(attack)