import numpy as np

def acc_on_advx(y_pred, y_true, detected_as_advx):
    """
    Compute the accuracy on the adversarial examples.

    :param y_pred: numpy array of integers
                 labels predicted by the classifier for the adversarial
                 examples.
    examples.
    :param y_true: numpy array of integers
                  true labels
    :param detected_as_advx: numpy array of boolean value
                the i-th value is true if the sample has been detected as an
                adversarial example, false otherwise
    """

    # Boolean vector. Each element is true if y_pred is equal to y_true
    correct_clf_pred = (y_pred == y_true)

    #print("correct_clf_pred", correct_clf_pred)

    # Apply the logic or to the correct_pred vector and the vector detected
    # as advx, namely it return a boolean vector where the i-th element is
    # True if (y_pred[i] == y_true) or (detected_as_advx == 1),
    # False otherwise.
    correct_clf_and_detector_pred = np.logical_or(correct_clf_pred,
                                                  detected_as_advx)

    #print("correct_clf_and_detector_pred", correct_clf_and_detector_pred)

    # the mean function considers the value True equal to 1 and False equal
    # to 0
    return np.mean(correct_clf_and_detector_pred)

############################################################################
# example
#y_pred = np.array([1,2,1,1,0])
#y_true = np.array([1,0,0,0,0])
#
# detected_as_advx = np.array([0,0,1,0,0])
#
# acc = acc_on_advx(y_pred, y_true, detected_as_advx)
#
# print("acc on advx ", acc)