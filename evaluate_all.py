import pandas as pd
from summarize_returns import summarize_returns

def evaluate_all(events_dt, events_rand, events_ma, horizons=(5, 20, 60)):
    """
    Build a summary DataFrame for a single ticker over horizons and event types.

    Parameters
    ----------
    events_dt : pd.DataFrame
        DataFrame of double-top events with forward returns.
    events_rand : pd.DataFrame
        DataFrame of random baseline events with forward returns.
    events_ma : pd.DataFrame
        DataFrame of moving-average crossover baseline events with forward returns.
    horizons : tuple of int
        Forward-return horizons in trading days (default is (5, 20, 60))
    """
    records = []
    for h in horizons:
        col = f'ret_{h}d'
        for label, df_src in [('double_top', events_dt),
                              ('random', events_rand),
                              ('ma_crossover', events_ma)]:
            stats_dict = summarize_returns(df_src[col])
            stats_dict.update({'horizon': h, 'type': label})
            records.append(stats_dict)

    summary_df = pd.DataFrame(records)
    return summary_df
