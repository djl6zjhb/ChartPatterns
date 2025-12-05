# ChatGPT generated code block
# I wanted to ensure the statistical significance of my model's findings
# Due to the importance of this test for my results, I wanted to ensure it was done correctly

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.utils import shuffle

from xgboost import XGBClassifier

from compute_all_metrics import compute_all_metrics

def permutation_test_all_metrics(
    base_model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_permutations=500,
):
    """
    Run a permutation test for multiple metrics at once.

    Parameters
    ----------
    base_model : sklearn-compatible classifier (unfitted)
        e.g. XGBClassifier with chosen hyperparameters (NOT already fit).
    X_train, y_train, X_test, y_test : pd.DataFrame();
        pre-split labeled events, respecting dates of events
    n_permutations : int
        Number of label-shuffle runs.

    Returns
    -------
    metrics_real : dict
        Metric values for the real (unshuffled) labels.
    metrics_null : dict[str, np.ndarray]
        Null distributions for each metric (length = n_permutations).
    p_values : dict
        p-value for each metric, based on the null distribution.
    """
    
    # 1. Train REAL model
    model_real = clone(base_model)
    model_real.fit(X_train, y_train)
    y_prob_real = model_real.predict_proba(X_test)[:, 1]

    metrics_real = compute_all_metrics(y_test, y_prob_real)

    # 2. Prepare storage for null distributions
    metrics_null = {name: np.zeros(n_permutations) for name in metrics_real.keys()}

    # 3. Permutation loop
    for i in range(n_permutations):
        # Shuffle the training labels
        y_perm = shuffle(y_train, random_state=i)

        # Refit model on permuted labels
        model_perm = clone(base_model)
        model_perm.fit(X_train, y_perm)

        # Predict on the SAME test set
        y_prob_perm = model_perm.predict_proba(X_test)[:, 1]
        m_perm = compute_all_metrics(y_test, y_prob_perm)

        # Store each metric
        for name in metrics_null.keys():
            metrics_null[name][i] = m_perm[name]


    # 4. Compute p-values for each metric
    #    For "higher is better": p = P(null >= real)
    #    For "lower is better" (Brier): p = P(null <= real)
    higher_is_better = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    lower_is_better = {"brier"}

    p_values = {}
    for name, real_value in metrics_real.items():
        null_vals = metrics_null[name]

        if name in higher_is_better:
            p = np.mean(null_vals >= real_value)
        elif name in lower_is_better:
            p = np.mean(null_vals <= real_value)
        else:
            # default to higher_is_better, but you can adjust
            p = np.mean(null_vals >= real_value)

        p_values[name] = p

    # Preventing hard zero for p-values
    p_values = {k: ((float(v) + 1) / (n_permutations + 1))  for k, v in p_values.items()}


    return metrics_real, metrics_null, p_values



