import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(fold_metrics, metric="roc_auc"):
    """
    Plot a single metric across folds.

    Parameters
    ----------
    fold_metrics : list of dict
        Output from evaluate_classifier()["fold_metrics"].

    metric : str
        Which metric to plot (e.g. "roc_auc", "accuracy", "f1").
    """
    values = [fm[metric] for fm in fold_metrics]
    folds = np.arange(1, len(values) + 1)

    plt.figure()
    plt.plot(folds, values, marker="o")
    plt.xlabel("Fold")
    plt.ylabel(metric)
    plt.title(f"{metric} by fold (walk-forward CV)")
    plt.grid(True)
    plt.tight_layout()
