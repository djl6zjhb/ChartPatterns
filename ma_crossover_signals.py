import pandas as pd
from compute_forward_returns import compute_forward_returns

def ma_crossover_signals(df, short_window=20, long_window=50, horizons=(5, 20, 60)):
    data = df.copy()
    data['ma_short'] = data['Close'].rolling(short_window).mean()
    data['ma_long'] = data['Close'].rolling(long_window).mean()

    # short MA crossing from above to below long MA
    prev = data.shift(1)
    cond_prev = prev['ma_short'] > prev['ma_long']
    cond_now = data['ma_short'] <= data['ma_long']
    signals = data[cond_prev & cond_now].dropna()

    events = []
    for t0, row in signals.iterrows():
        events.append({'confirm_date': t0, 'confirm_price': row['Close']})
    ma_df = pd.DataFrame(events)

    ma_df = compute_forward_returns(df, ma_df, horizons=horizons)
    ma_df['type'] = 'ma_crossover'
    return ma_df
