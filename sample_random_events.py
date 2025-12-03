import numpy as np
import pandas as pd
from compute_forward_returns import compute_forward_returns

def sample_random_events(df, n_events, horizons=(5, 20, 60), seed=42, buffer=60):
    """
    Sample random dates from DataFarme of stock price info, avoiding first/last `buffer` days so all horizons fit. 
    Computes forward returns at randomly seelected dates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data for a single ticker from 2015-01-02 to 2024-12-30.
    n_events : int
        Number of random events to sample. Intended to match number of double-top events.
    horizons : tuple of int
        Forward-return horizons in trading days (default is (5, 20, 60))
    seed : int
        Random seed for reproducibility (default is 42)
    buffer : int
        Number of days to avoid at beginning and end of DataFrame to ensure all forward-return horizons fit (default is 60)
    """
    # randomly selecting index positions for events, avoiding edges
    rng = np.random.default_rng(seed)
    idx = df.index
    valid_idx = idx[buffer:-buffer]  # avoid edges
    chosen_idx = rng.choice(valid_idx, size=min(n_events, len(valid_idx)), replace=False)

    # gathering price and date information for selected dates
    events = []
    for t0 in chosen_idx:
        row = {'confirm_date': t0, 'confirm_price': df.loc[t0, 'Close']}
        events.append(row)
    rand_df = pd.DataFrame(events)

    # compute forward returns at given dates
    rand_df = compute_forward_returns(df, rand_df, horizons=horizons)
    rand_df['type'] = 'random'
    return rand_df
