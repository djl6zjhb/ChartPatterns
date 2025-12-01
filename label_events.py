import pandas as pd

def label_events(dt_candidates: pd.DataFrame, dt_events: pd.DataFrame) -> pd.DataFrame:
    """
    Label each candidate in dt_candidates as True if it is in dt_events, else False.
    """
    dt_events_set = set(dt_events.index)
    dt_candidates['label'] = dt_candidates.index.map(lambda idx: idx in dt_events_set)
    return dt_candidates