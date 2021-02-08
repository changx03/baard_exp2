import numpy as np


def acc_on_advx(y_pred, y_true, detected_as_adv):
    """Compute the accuracy on the adversarial examples.

    Parameters
    ----------
    y_pred: numpy array of integers
        labels predicted by the classifier for the adversarial examples.

    y_true: numpy array of integers
        true labels.

    detected_as_advx: numpy array of boolean value
        the i-th value is true if the sample has been detected as an adversarial
        example, false otherwise.

    Returns
    -------
    accuracy: float
        Accuracy on adversarial examples.
    """
    correct_classified = y_pred == y_true
    correct_classified_and_detected = np.logical_or(correct_classified, detected_as_adv)
    return np.mean(correct_classified_and_detected)
