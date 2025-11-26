import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_parameter_heatmap(dt_events, horizon=20, gap_bins=[15,20,25,30], vol_bins=[0.0,0.5,0.8,1.0]):
    col = f'ret_{horizon}d'
    data = dt_events.dropna(subset=[col]).copy()
    data['gap_bin'] = pd.cut(data['peak_gap_days'], bins=gap_bins, include_lowest=True)
    data['vol_bin'] = pd.cut(data['vol2_vol1_ratio'], bins=vol_bins, include_lowest=True)

    pivot = data.pivot_table(index='gap_bin', columns='vol_bin', values=col, aggfunc='mean')

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pivot.values, aspect='auto', origin='lower')
    plt.colorbar(im, label=f'Mean {horizon}-day return')
    plt.xticks(ticks=range(len(pivot.columns)), labels=[str(c) for c in pivot.columns], rotation=45)
    plt.yticks(ticks=range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    plt.xlabel('vol2/vol1 bin')
    plt.ylabel('peak_gap_days bin')
    plt.title('Double top parameter “heatmap”')
    plt.tight_layout()
    plt.show()
