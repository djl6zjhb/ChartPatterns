import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from summarize_returns import summarize_returns

def evaluate_all(events_dt, events_rand, events_ma, horizons=(5, 20, 60)):
    """
    Build a summary DataFrame over horizons and event types.
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
