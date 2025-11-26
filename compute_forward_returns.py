import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from pos_to_date import pos_to_date

def compute_forward_returns(df, events_df, horizons=(5, 20, 60)):
    """
    For each event (by confirm_date), compute forward returns.
    Returns are simple percentage returns: Close[t+H] / Close[confirm] - 1.
    """
    events_df = events_df.copy()
    
    # events_df = pos_to_date(df, events_df)
    # print(events_df)
    for h in horizons:
        col = f'ret_{h}d'
        vals = []
        for _, row in events_df.iterrows():
            t0 = row['confirm_date']
            # print(t0)
            t_fwd = df.index[df.index.get_loc(t0) + h] if (df.index.get_loc(t0) + h) < len(df.index) else None
            if t_fwd is None:
                vals.append(np.nan)
                continue

            price0 = df.loc[t0, 'Close']
            price1 = df.loc[t_fwd, 'Close']
            vals.append(price1 / price0 - 1.0)
        events_df[col] = vals
    return events_df
