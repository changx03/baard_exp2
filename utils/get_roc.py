import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def get_roc(y_true, y_prob, show_plot=False):
    """Returns False-Positive-Rate, True-Positive-Rate, AUC score and thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score, thresholds
