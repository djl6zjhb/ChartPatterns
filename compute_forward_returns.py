import numpy as np

def compute_forward_returns(df, events_df, predicted = False, horizons=(5, 20, 60)):
    """
    For each event (by confirm_date), compute forward returns at the given horizons.
    Returns are simple percentage returns: (Close[t+H] / Close[confirm]) - 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data for a single ticker.
    events_df : pd.DataFrame
        DataFrame of confirmed double top events with 'confirm_date' column.
    predicted : bool
        If True, indicates that events_df contains predicted events.
    horizons : tuple of int
        Forward-return horizons in trading days.
    """
    events_df = events_df.copy()
    
    # compute forward returns for each specified horizon
    for h in horizons:
        col = f'ret_{h}d'
        vals = []
        for _, row in events_df.iterrows():
            
            # index at double top confirmation date
            if predicted:
                t0 = row['peak2_pos'] + 1 # returns must be calculated from the first day after closing at a second peak
            else:
                t0 = row['confirm_date']

            # index at the given horizon if t0 + h is within df index range; edge case to avoid erroring out
            t_fwd = df.index[df.index.get_loc(t0) + h] if (df.index.get_loc(t0) + h) < len(df.index) else None
            if t_fwd is None:
                vals.append(np.nan)
                continue
            
            # Calculating forward return from confirmation date to horizon date
            price0 = df.loc[t0, 'Close']
            price1 = df.loc[t_fwd, 'Close']
            vals.append(price1 / price0 - 1.0)
        events_df[col] = vals
    return events_df
