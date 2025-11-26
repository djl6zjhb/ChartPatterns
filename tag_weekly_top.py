import pandas as pd

def tag_weekly_top(df, events_df, lookback_weeks=12):
    """
    For each confirmation date, check if it's within a weekly bar that
    is near the max weekly close over a lookback window – crude 'weekly top'.
    """
    # Resample to weekly close
    weekly = df['Close'].resample('W-FRI').last()
    weekly_max = weekly.rolling(lookback_weeks).max()

    tags = []
    for _, row in events_df.iterrows():
        t_conf = row['confirm_date']
        # find the week label corresponding to this day
        week_label = weekly.index[weekly.index.get_loc(t_conf, method='bfill')]
        wk_close = weekly.loc[week_label]
        wk_max = weekly_max.loc[week_label]
        if pd.isna(wk_max):
            tags.append(False)
        else:
            # consider it a weekly top if within 1% of the rolling max
            tags.append((wk_max - wk_close) / wk_max <= 0.01)

    events_df = events_df.copy()
    events_df['is_weekly_top'] = tags
    return events_df
