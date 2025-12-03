import numpy as np

def cohen_d_two_sample(a, b):
    """
    Compute a cohnen's d effect size between two independent samples.
    
    Parameters
    ----------
    a : pd.Series
        First sample of returns.
    b : array-like
        Second sample of returns.
    """
    a = a.dropna().values
    b = b.dropna().values

    if len(a) < 5 or len(b) < 5:
        return np.nan

    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)

    # pooled sd
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    
    return (m1 - m2) / s_pooled if s_pooled > 0 else np.nan
