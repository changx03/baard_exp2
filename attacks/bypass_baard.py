import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from defences.baard import flatten


class BAARD_Clipper:
    def __init__(self, baard_detector):
        self.baard_detector = baard_detector

    def __call__(self, X, classifier):
        thresholds = self.baard_detector.stages[0].thresholds_
        pred = classifier.predict(X)
        y_pred = np.argmax(pred, axis=1)
        return clip_by_threshold(X, y_pred, thresholds)


def clip_by_threshold(X, y, thresholds):
    n_classes = thresholds.shape[0]
    shape = X.shape
    X_flat = flatten(X)
    out_flat = X_flat.copy()

    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        bounding_box = thresholds[c]
        low = bounding_box[0]
        high = bounding_box[1]
        subset = X_flat[idx]
        subset_clipped = np.clip(subset, low, high)
        out_flat[idx] = subset_clipped

    return out_flat.reshape(shape)
