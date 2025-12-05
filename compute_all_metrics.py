# ChatGPT generated code block
# Used in conjunction with permutation test to ensure statistical signficance of model findings

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,  # PR-AUC
    brier_score_loss,
)

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute a suite of classification metrics from true labels and predicted probs.
    Assumes binary classification with positive class = 1.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }

    return metrics