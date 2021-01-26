"""
This module implements the watermark attack.
"""

import numpy as np
from sklearn.neighbors import BallTree
from tqdm import trange


class WaterMarkAttack:
    def __init__(self,
                 eps=0.3,
                 base=1.0,
                 n_classes=10,
                 x_min=0.0,
                 x_max=1.0,
                 targeted=False,
                 verbose=True):
        self.eps = eps
        self.base = base
        self.n_classes = n_classes
        self.x_min = x_min
        self.x_max = x_max
        self.targeted = targeted
        self.verbose = verbose

        self.trees_ = []
        self.X_train_ = []

    def fit(self, X, y):
        X = np.reshape(X, (X.shape[0], -1))
        if self.targeted:
            # tree contains only targeted samples.
            for i in trange(self.n_classes, disable=not self.verbose):
                idx = np.where(y == i)[0]
                if len(idx) > 1000:
                    idx = np.random.choice(idx, size=1000, replace=False)
                tree = BallTree(X[idx], leaf_size=32)
                self.trees_.append(tree)
                self.X_train_.append(X[idx])
        else:  # untargeted
            # tree contains samples other than the true label.
            for i in trange(self.n_classes, disable=not self.verbose):
                idx = np.where(y != i)[0]
                if len(idx) > 5000:
                    idx = np.random.choice(idx, size=5000, replace=False)
                tree = BallTree(X[idx], leaf_size=128)
                self.trees_.append(tree)
                self.X_train_.append(X[idx])

    def generate(self, x, y):
        shape = x.shape
        x = np.reshape(x, (x.shape[0], -1))
        X_adv = np.copy(x)
        for i in trange(self.n_classes, disable=not self.verbose):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            idx_neigh = self.trees_[i].query(x[idx], k=1, return_distance=False)
            idx_neigh = np.squeeze(idx_neigh)
            X_neigh = self.X_train_[i][idx_neigh]
            temp_adv = self.base * x[idx] + self.eps * X_neigh
            temp_adv = np.clip(temp_adv, self.x_min, self.x_max)
            X_adv[idx] = temp_adv
        return np.reshape(X_adv, shape)
