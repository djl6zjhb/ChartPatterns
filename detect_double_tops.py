import numpy as np
import pandas as pd
from find_local_extrema import find_local_extrema

def detect_double_tops(
    df,
    peak_window=3,
    peak_tolerance=0.01,   
    min_peak_gap=15,
    max_peak_gap=30,
    min_trough_drop=0.03,
    require_lower_second_vol=True
):
    """
    Detect candidate double top patterns and return a DataFrame of pattern events.
    Confirmation of patterns by observing future close below trough done in a separate function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data for a single ticker from 2015-01-02 to 2024-12-30.
        Columns: 'Date', 'Open', 'Close', 'High', 'Low', and 'Volume'.
    peak_window : int, optional
        Window size for detecting local peaks and troughs (default is 3). Measured in days.
    peak_tolerance : float, optional
        Maximum allowed relative difference between the two peaks to consider them similar (default is 0.01).
    min_peak_gap : int, optional
        Minimum number of days between the two peaks (default is 15).
    max_peak_gap : int, optional
        Maximum number of days between the two peaks (default is 30).
    min_trough_drop : float, optional
        Minimum relative drop of the trough compared to the peaks (default is 0.03).
    require_lower_second_vol : bool, optional
        Whether to require the second peak's volume to be lower than the first (default is True).
    """

    local_high, local_low = find_local_extrema(df, window=peak_window)
    
    peaks = df[local_high].copy()
    troughs = df[local_low].copy()

    peaks_idx = peaks.index
    events = []

    for i in range(len(peaks_idx)):
        t1 = peaks_idx[i]
        p1 = peaks.loc[t1]

        # look for second peak within [min_peak_gap, max_peak_gap] days
        min_date = t1 + min_peak_gap
        max_date = t1 + max_peak_gap

        candidate_peaks2 = peaks[(peaks.index >= min_date) & (peaks.index <= max_date)]
        
        # edge case for no candidate second peaks, like for the last peak in the data
        if candidate_peaks2.empty:
            continue

        for t2, p2 in candidate_peaks2.iterrows():
            price1 = p1['High']
            price2 = p2['High']

            # peaks similar in height
            if abs(price2 - price1) / price1 > peak_tolerance:
                continue

            # trough between them
            mid_troughs = troughs[(troughs.index > t1) & (troughs.index < t2)]
            if mid_troughs.empty:
                continue
            
            trough_date = mid_troughs['Low'].idxmin()
            trough_price = mid_troughs.loc[trough_date, 'Low']

            # trough must be meaningfully below peaks
            avg_peak = (price1 + price2) / 2
            if (avg_peak - trough_price) / avg_peak < min_trough_drop:
                continue

            # second peak trading volume must be lower than first
            vol1 = p1['Volume']
            vol2 = p2['Volume']
            if require_lower_second_vol and not (vol2 < vol1):
                continue

            # add to return df
            # all dates saved as index rather than actual date; relevant in later functions
            events.append({
                'peak1_date': t1,
                'peak2_date': t2,
                'trough_date': trough_date,
                'peak1_price': price1,
                'peak2_price': price2,
                'trough_price': trough_price,
                'peak_gap_days': (t2 - t1),
                'vol1': vol1,
                'vol2': vol2,
                'vol2_vol1_ratio': vol2 / vol1 if vol1 > 0 else np.nan
            })

    events_df = pd.DataFrame(events)
    return events_df
