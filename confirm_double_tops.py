import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

def confirm_double_tops(df, events_df, max_confirm_days=20):
    """
    For each candidate double top, look for first close below trough (neckline)
    after the second peak. That date is confirmation_date.
    """
    confirmed = []

    for _, row in events_df.iterrows():
        t2 = row['peak2_date']
        neck_price = row['trough_price']

        start = t2 + 1
        end = t2 + max_confirm_days

        post = df.loc[(df.index >= start) & (df.index <= end)]

        below = post[post['Close'] < neck_price]
        if below.empty:
            continue

        confirm_date = below.index[0]
        confirm_price = below.loc[confirm_date, 'Close']

        r = row.to_dict()
        r.update({
            'confirm_date': confirm_date,
            'confirm_price': confirm_price
        })
        confirmed.append(r)

    confirmed_df = pd.DataFrame(confirmed)
    return confirmed_df
