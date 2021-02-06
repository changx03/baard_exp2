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
        return clip_by_threshold(X, pred, thresholds)


# FIXME: This function still does not work with BAARD.
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


if __name__ == '__main__':
    # Load adversrial examples
    path = os.path.join('result_0', 'mnist_dnn_apgd_300.pt')
    obj = torch.load(path)
    X = obj['X']
    adv = obj['adv']
    y = obj['y']  # NOTE: This is wrong, should be the prediction! Test only!
    # Load thresholds
    path = os.path.join('result_0', 'mnist_dnn_baard_threshold.pt')
    obj = torch.load(path)
    thresholds = obj['thresholds'][0]
    X_clip = clip_by_threshold(X, y, thresholds)
    adv_clip = clip_by_threshold(adv, y, thresholds)
    print('L2 dist X:', np.linalg.norm(X - X_clip))
    print('L2 dist adv:', np.linalg.norm(adv - adv_clip))
