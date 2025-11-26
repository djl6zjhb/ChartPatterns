import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

def find_local_extrema(df, window=3):
    """
    Very simple local high/low markers using a rolling window.
    A point is a local high if its High is the max over [t-window, t+window].
    Similar for local low.
    """
    highs = df['High']
    lows = df['Low']

    local_high = (highs == highs.rolling(window*2+1, center=True).max())
    local_low = (lows == lows.rolling(window*2+1, center=True).min())

    return local_high.fillna(False), local_low.fillna(False)
