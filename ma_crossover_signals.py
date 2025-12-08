# This is another function contributed largely from ChatGPT. 
# My original attempt was very inefficient. 
# This version uses vectorized operations which operates orders of magnitude faster than my first attempt.
# The returns were verified using my own function to ensure correctness.

import pandas as pd
from compute_forward_returns import compute_forward_returns

def ma_crossover_signals(df, short_window=20, long_window=50, horizons=(5, 20, 60)):
    """
    Calculates long and short moving averages for a given ticker and evaluates returns when moving averages cross.
    Utilizes compute_forward_returns to calculate returns after crossover events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data for a single ticker from 2015-01-02 to 2024-12-30.
    short_window : int
        Window size for the short-term moving average (default is 20 days).
    long_window : int
        Window size for the long-term moving average (default is 50 days).
    horizons : tuple of int
        Forward-return horizons in trading days (default is (5, 20, 60))
    """
    data = df.copy()
    data['ma_short'] = data['Close'].rolling(short_window).mean()
    data['ma_long'] = data['Close'].rolling(long_window).mean()

    # short MA crossing from above to below long MA, indicating a downward turn in the market
    
    # begin ChatGPT section
    prev = data.shift(1)
    cond_prev = prev['ma_short'] > prev['ma_long']
    cond_now = data['ma_short'] <= data['ma_long']
    signals = data[cond_prev & cond_now].dropna()
    # end ChatGPT section

    # for all determined ma crossovers, loop through price data to get date and priec
    events = []
    for t0, row in signals.iterrows():
        events.append({'confirm_date': t0, 'confirm_price': row['Close']})
    ma_df = pd.DataFrame(events)

    # compute forward returns for these ma crossover events
    ma_df = compute_forward_returns(df, ma_df, horizons=horizons)
    ma_df['type'] = 'ma_crossover'

    return ma_df
