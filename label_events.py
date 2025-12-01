import pandas as pd
from pos_to_date import pos_to_date

def label_events(dt_candidates: pd.DataFrame, dt_events: pd.DataFrame) -> pd.DataFrame:
    """
    Label each candidate in dt_candidates as True if it is in dt_events, else False.
    """
    dt_events = dt_events.set_index(['peak1_date', 'trough_date', 'peak2_date'])
    dt_candidates = dt_candidates.set_index(['peak1_date', 'trough_date', 'peak2_date'])
    dt_events_set = set(dt_events.index)
    dt_candidates['label'] = dt_candidates.index.map(lambda idx: idx in dt_events_set)
    return dt_candidates.reset_index()