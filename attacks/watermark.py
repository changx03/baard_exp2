"""
This module implements the watermark attack.
"""

import numpy as np
from sklearn.neighbors import BallTree


class WaterMarkAttack:
    def __init__(self,
                 eps=0.3,
                 base=1.0,
                 n_classes=10,
                 x_min=0.0,
                 x_max=1.0,
                 targeted=False):
        self.eps = eps
        self.base = base
        self.n_classes = n_classes
        self.x_min = x_min
        self.x_max = x_max
        self.targeted = targeted

        self.trees_ = []
        self.X_train_ = []

    def fit(self, X, y):
        X = np.reshape(X, (X.shape[0], -1))
        if self.targeted:
            # tree contains only targeted samples.
            for i in range(self.n_classes):
                idx = np.where(y == i)[0]
                tree = BallTree(X[idx], leaf_size=32)
                self.trees_.append(tree)
                self.X_train_.append(X[idx])
        else:  # untargeted
            # tree contains samples other than the true label.
            for i in range(self.n_classes):
                idx = np.where(y != i)[0]
                tree = BallTree(X[idx], leaf_size=128)
                self.trees_.append(tree)
                self.X_train_.append(X[idx])

    def generate(self, X, y):
        shape = X.shape
        X = np.reshape(X, (X.shape[0], -1))
        X_adv = np.copy(X)
        for i in range(self.n_classes):
            idx = np.where(y == i)[0]
            if len(idx) == 0:
                continue
            idx_neigh = self.trees_[i].query(X[idx], k=1, return_distance=False)
            idx_neigh = np.squeeze(idx_neigh)
#             print(self.X_train_[i].shape)
            X_neigh = self.X_train_[i][idx_neigh]
#             print(X_neigh.shape)
            temp_adv = self.base * X[idx] + self.eps * X_neigh
#             print(temp_adv.shape)
            temp_adv = np.clip(temp_adv, self.x_min, self.x_max)
            X_adv[idx] = temp_adv
        return np.reshape(X_adv, shape)
