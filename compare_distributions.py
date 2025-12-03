import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind

def compare_distributions(ret_a, ret_b):
    """
    Computing comparison statistics of two return series using the Mann-Whitney U test, Kolmogorov-Smirnov test, and two-sample t-test

    Parameters
    ----------
    ret_a : pd.Series
        Return series for one event type.
    ret_b : pd.Series
        Return series for second event type.
    """
    if len(ret_a) < 5 or len(ret_b) < 5:
        return {
            'mw_u': np.nan, 'mw_p': np.nan,
            'ks_stat': np.nan, 'ks_p': np.nan,
            't_stat': np.nan, 't_p': np.nan
        }

    mw_u, mw_p = mannwhitneyu(ret_a, ret_b, alternative='two-sided')
    ks_stat, ks_p = ks_2samp(ret_a, ret_b)
    t_stat, t_p = ttest_ind(ret_a, ret_b, equal_var=False)

    return {
        'mw_u': mw_u,
        'mw_p': mw_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        't_stat': t_stat,
        't_p': t_p
    }