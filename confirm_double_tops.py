import pandas as pd

def confirm_double_tops(df, events_df, max_confirm_days=20):
    """
    Confirmation of candidate double tops by observing future close below trough.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data for a single ticker from 2015-01-02 to 2024-12-30.
    events_df : pd.DataFrame
        DataFrame of candidate double top events detected by `detect_double_tops`.
    max_confirm_days : int, optional
        Maximum number of days after the second peak to look for confirmation (default is 20).
    """
    confirmed = []

    for _, row in events_df.iterrows():
        
        # extracting relevant candidate event details
        t2 = row['peak2_date']
        neck_price = row['trough_price']

        # defining window to search for confirmation
        start = t2 + 1
        end = t2 + max_confirm_days

        # creating subset of data within the confirmation window
        post = df.loc[(df.index >= start) & (df.index <= end)]

        # checking for close below neckline; if found, record confirmation price and date
        below = post[post['Close'] < neck_price]
        if below.empty:
            continue

        confirm_date = below.index[0]
        confirm_price = below.loc[confirm_date, 'Close']

        # add confirmed event to return df with additional confirmation details
        r = row.to_dict()
        r.update({
            'confirm_date': confirm_date,
            'confirm_price': confirm_price
        })
        confirmed.append(r)

    confirmed_df = pd.DataFrame(confirmed)
    return confirmed_df
