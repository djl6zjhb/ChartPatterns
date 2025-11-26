import numpy as np
from scipy import stats
from compute_forward_returns import compute_forward_returns

def summarize_returns(ret_series):
    """
    ret_series: pandas Series of returns for a given horizon.
    """
    ret = ret_series.dropna()
    if len(ret) < 5:
        return {
            'n': len(ret),
            'mean': np.nan,
            'std': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'hit_ratio_neg': np.nan,
            'sharpe': np.nan,
            'cohen_d': np.nan
        }

    mean = ret.mean()
    std = ret.std(ddof=1)
    t_stat, p_value = stats.ttest_1samp(ret, popmean=0.0)
    hit_ratio_neg = (ret < 0).mean()
    sharpe = mean / std if std > 0 else np.nan
    cohen_d = mean / std if std > 0 else np.nan  # same formula

    return {
        'n': len(ret),
        'mean': mean,
        'std': std,
        't_stat': t_stat,
        'p_value': p_value,
        'hit_ratio_neg': hit_ratio_neg,
        'sharpe': sharpe,
        'cohen_d': cohen_d
    }
