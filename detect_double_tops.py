import numpy as np
import pandas as pd
from find_local_extrema import find_local_extrema

def detect_double_tops(
    df,
    peak_window=3,
    peak_tolerance=0.01,     # peaks must be within 1%
    min_peak_gap=15,
    max_peak_gap=30,
    min_trough_drop=0.03,    # trough must be at least 3% below peaks
    require_lower_second_vol=True
):
    """
    Detect double top patterns and return a DataFrame of pattern events
    *before confirmation* (confirmation checked separately).
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

            # volume condition (second peak volume < first)
            vol1 = p1['Volume']
            vol2 = p2['Volume']
            if require_lower_second_vol and not (vol2 < vol1):
                continue

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
