import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(y_true, probs):
    """
    Plot overall ROC across all folds
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    # PR
    precision, recall, _ = precision_recall_curve(y_true, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (overall)")
    plt.grid(True)
    plt.tight_layout()

    plt.show()
