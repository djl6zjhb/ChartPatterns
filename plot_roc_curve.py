import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curve(y_true, probs):
    """
    Plot precision-recall curve using all fold predictions
    out-of-fold predictions.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    fpr, tpr, _ = roc_curve(y_true, probs)


    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (overall)")
    plt.grid(True)
    plt.tight_layout()

    plt.show()
