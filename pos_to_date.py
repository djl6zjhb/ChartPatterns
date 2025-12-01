import pandas as pd

def pos_to_date(df, dt_events, confirm_pos=True) -> pd.DataFrame:
    """
    Convert position-based indices in events_df to actual dates from df index.
    Assumes events_df has a 'position' column with integer positions.
    """
    df = df.reset_index()
    dates = df.Date

    dt_events = dt_events.rename(columns={'peak1_date': 'peak1_pos','peak2_date': 'peak2_pos','trough_date': 'trough_pos','confirm_date': 'confirm_pos'})

    dt_events['peak1_date'] = dates.loc[dt_events['peak1_pos']].values
    dt_events['trough_date'] = dates.loc[dt_events['trough_pos']].values
    dt_events['peak2_date'] = dates.loc[dt_events['peak2_pos']].values
    if confirm_pos == True:
        dt_events['confirm_date'] = dates.loc[dt_events['confirm_pos']].values
    
    
    return dt_events