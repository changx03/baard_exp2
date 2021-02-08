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
from utils import acc_on_advx, set_seeds
import json


def load_predictions_and_compute_acc_on_advx(pred_folder, eps):

    pred_folder = pred_folder + "_{:}".format(eps)

    print("Load predictions")

    y = np.load(pred_folder + "_{:}".format("y"))
    y_pred = np.load(pred_folder + "_{:}".format("y-pred.npy"))

    # nb, those are the logits
    pred_sur_det = np.load(pred_folder + "_{:}".format("pred-sur-det.npy"))

    pred_baard = np.load(pred_folder + "_{:}".format("pred-baard.npy"))
    reject_s1= np.load(pred_folder + "_{:}".format("reject-s1.npy"))
    reject_s2= np.load(pred_folder + "_{:}".format("reject-s2.npy"))
    reject_s3 = np.load(pred_folder + "_{:}".format("reject-s3.npy"))
    print("Predictions loaded")

    # #print( y )
    # print(y_pred)
    # print(pred_sur_det)
    # print(pred_baard)
    # print(reject_s1)
    # print(reject_s2)
    # print(reject_s3)

    #y = np.ones((y_pred.size,))

    acc = acc_on_advx(y_pred, y, pred_baard)

    return acc


def get_acc_for_rep(seed, dataset_name, clf_name, detector_name, epsilon_lst):
    """
    Compute the accuracy for a single rep
    """

    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    pred_folder = 'result_{:}/predictions_wb_eval/{:}_{:}_{:}'.format(seed,
                                                              dataset_name,
                                                       clf_name, detector_name)
    # compute and save predictions
    accs = []
    accs.append(load_predictions_and_compute_acc_on_advx(pred_folder, 0))

    for eps in epsilon_lst:
        accs.append( load_predictions_and_compute_acc_on_advx(pred_folder,

                                                              eps))
    return np.array(accs)

def get_acc_for_defence(seedslst, dataset_name, clf_name, detector_name,
                    epsilon_lst):
    all_eps = [0] + epsilon_lst

    n_eps = len(all_eps)
    n_reps = len(seedslst)

    acc_matrix = np.ones((n_reps, n_eps))

    for rep_num, rep_idx in enumerate(seedslst):
        acc_matrix[rep_num,:] =  get_acc_for_rep(rep_idx, dataset_name,
                                                clf_name,
                                 detector_name,
                               epsilon_lst)
    return acc_matrix


def main(defencelst, defencenamelst, defencecolorlst, seedslst, \
                                                 dataset_idx,
    dataset_name, clf_name,
                    epsilon_lst):

    all_eps = [0] + epsilon_lst

    fig, ax = plt.subplots(1)

    for defence, defence_name, color in zip(defencelst, defencenamelst,
                                   defencecolorlst):
        acc_matrix = get_acc_for_defence(seedslst, dataset_idx, clf_name,
                                defence,
                            epsilon_lst)

        plot_res(ax, all_eps, acc_matrix, defence_name, color)

    ax.set_title(r'White-box attack ({:})'.format(dataset_name))
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Accuracy')
    ax.grid()

    plt.tight_layout()

    plt.savefig(dataset_idx +'_wb-seceval.pdf')


def plot_res(ax, all_eps, acc_matrix, defence_name,
             defence_color):
    # plot sec eval

    mu1 = acc_matrix.mean(axis=0)
    sigma1 = acc_matrix.std(axis=0)

    # plot it!
    ax.plot(all_eps, mu1, lw=2, label=defence_name, color=defence_color)
    ax.fill_between(all_eps, mu1 + sigma1, mu1 - sigma1, facecolor='blue', alpha=0.5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist', choices=[
        'mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='dnn', choices=['dnn',
                                                                   'resnet'])
    parser.add_argument('--seedslst', type=list, default=[1])
    parser.add_argument('--defencelst', type=list, default=['baard'])
    parser.add_argument('--defencenamelst', type=list, default=['BAARD'])
    parser.add_argument('--defencecolorlst', type=list, default=['blue'])

    args = parser.parse_args()
    print(args)

    dataset_idx = args.data
    clf_name = args.model

    if args.data == 'mnist':
        clf_name = 'dnn'
        epsilon_lst = [1,2,3,5,8]
        input_shape = (1, 28, 28)
        dataset_name = "MNIST NN"
    else:
        clf_name = 'resnet'
        epsilon_lst = [0.5, 1, 2, 3, 4]
        input_shape = (3, 32, 32)
        dataset_name = "CIFAR-10 Resnet"

    main(args.defencelst, args.defencenamelst, args.defencecolorlst,
         args.seedslst, dataset_idx, dataset_name,
         clf_name, epsilon_lst)
