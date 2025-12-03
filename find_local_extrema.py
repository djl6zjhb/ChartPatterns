# This module was heavily edited with ChatGPT
# My original attempt at this function was very inefficient
# This is significantly optimized, and given the amount of times it runs in my pipeline,
# I wanted to ensure it was as fast as possible.

def find_local_extrema(df, window=3):
    """
    Identify local max and mins in the 'High' and 'Low' columns of df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data for a single ticker.
    window : int, optional
        Window size for detecting local peaks and troughs (default is 3). Measured in days.
    """
    highs = df['High']
    lows = df['Low']

    local_high = (highs == highs.rolling(window*2+1, center=True).max())
    local_low = (lows == lows.rolling(window*2+1, center=True).min())

    return local_high.fillna(False), local_low.fillna(False)
