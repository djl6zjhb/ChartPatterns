import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_return_distributions(dt_events, rand_events, horizon=20):
    col = f'ret_{horizon}d'
    plt.figure(figsize=(8, 5))
    plt.hist(rand_events[col].dropna(), bins=30, alpha=0.5, label='Random')
    plt.hist(dt_events[col].dropna(), bins=30, alpha=0.5, label='Double top')
    plt.axvline(0, linestyle='--')
    plt.title(f'{horizon}-day return distribution')
    plt.xlabel('Return')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
