import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from walk_forward_split import WalkForwardSplit

def evaluate_classifier(model, X_train, y_train, X_test, y_test, threshold=0.5,):
    """
    Run walk-forward cross-validation and compute classification metrics to evaluate classifier performance.

    Parameters
    ----------
    model : estimator

    X_train, y_train : Feature matrix and labels for training.

    X_test, y_test : Feature matrix and labels for testing.

    threshold : probability threshold for classification

    Returns
    -------
    results : dictionary of classifer evaluation results
    """

    y_true_all = []
    probs_all = []
    preds_all = []
    fold_ids = []

    model.fit(X_train, y_train)

    # Predict probabilities for positive class
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Compute metrics for this fold
    results = {}

    results["accuracy"] = accuracy_score(y_test, preds)
    results["precision"] = precision_score(y_test, preds, zero_division=0)
    results["recall"] = recall_score(y_test, preds, zero_division=0)       
    results["f1"] = f1_score(y_test, preds, zero_division=0)
    results["roc_auc"] = roc_auc_score(y_test, probs)
    results["pr_auc"] = average_precision_score(y_test, probs)
    results["brier"] = brier_score_loss(y_test, probs)

    print(
            f"Best Model Results:"
            f"acc={results['accuracy']:.3f}, "
            f"prec={results['precision']:.3f}, "
            f"recall={results['recall']:.3f}, "
            f"f1={results['f1']:.3f}, "
            f"roc_auc={results['roc_auc']:.3f}, "
            f"pr_auc={results['pr_auc']:.3f}"
            f"brier={results['brier']:.3f}"       
        )
    
    return results
