import pandas as pd
import os

from detect_double_tops import detect_double_tops
from confirm_double_tops import confirm_double_tops
from label_events import label_events
from xgboost import XGBClassifier

def prediction_pipeline(
    tickers: list,
):
    """
    Pipeline function to detect and label double top events across multiple tickers.

    Using these potential and confirmed double top events, compute additional features and train a model to predict future double top events.
    
    Parameters
    ----------
    tickers: list
        list of ticker symbols to process
    """
    # Step 1: initialize an empty DataFrame to hold all labeled data
    labeled_data = pd.DataFrame()

    # Step 2: iterate over each ticker and process data to find candidate events and confirmed double tops
    for ticker in tickers:
        df = pd.read_csv(f"sp500/sp500/{ticker}.csv", 
                     header=0, 
                     skiprows=[1],
                     dtype={
                        "Open": float,
                        "High": float,
                        "Low": float,
                        "Close": float,
                        "Volume": float
                        }
            )
        dt_candidates = detect_double_tops(df, peak_window=3, peak_tolerance=0.01,min_peak_gap=15, max_peak_gap=30, min_trough_drop=0.03, require_lower_second_vol=True)

        dt_confirmed = confirm_double_tops(df, dt_candidates, max_confirm_days=40)
        dt_confirmed = dt_confirmed.drop(columns=['confirm_date', 'confirm_price'])

        all_events = label_events(dt_candidates, dt_confirmed)

        labeled_data = pd.concat([labeled_data, all_events]).reset_index(drop=True)

    return labeled_data
    # Step 3: feature engineering (placeholder - implement as needed)

    # Step 4: model training (placeholder - implement as needed)

if __name__ == "__main__":
    tickers = ['A', 'AAPL']  # Example list of tickers
    labeled_data = prediction_pipeline(tickers)
    print(labeled_data.sort_values(by='peak1_date', ascending=True))


