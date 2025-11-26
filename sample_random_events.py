import numpy as np
import pandas as pd
from compute_forward_returns import compute_forward_returns

def sample_random_events(df, n_events, horizons=(5, 20, 60), seed=42, buffer=60):
    """
    Sample random confirmation dates from df index, avoiding first/last `buffer` days
    so all horizons fit. Returns a DataFrame similar to events_df with forward returns.
    """
    rng = np.random.default_rng(seed)
    idx = df.index

    valid_idx = idx[buffer:-buffer]  # avoid edges
    chosen_idx = rng.choice(valid_idx, size=min(n_events, len(valid_idx)), replace=False)

    events = []
    for t0 in chosen_idx:
        row = {'confirm_date': t0, 'confirm_price': df.loc[t0, 'Close']}
        events.append(row)
    rand_df = pd.DataFrame(events)

    # reuse forward-return logic
    rand_df = compute_forward_returns(df, rand_df, horizons=horizons)
    rand_df['type'] = 'random'
    return rand_df
