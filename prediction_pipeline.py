import pandas as pd
import os

from detect_double_tops import detect_double_tops
from confirm_double_tops import confirm_double_tops
from pos_to_date import pos_to_date
from label_events import label_events
from xgboost import XGBClassifier
from walk_forward_split import WalkForwardSplit

def prediction_pipeline(
    tickers: list,
    param_dict: dict = {'peak_window': 3, 'peak_tolerance': 0.01,
                        'min_peak_gap': 15, 'max_peak_gap': 30,
                        'min_trough_drop': 0.03, 'require_lower_second_vol': True}
):
    """
    Pipeline function to detect and label double top events across multiple tickers.

    Using these potential and confirmed double top events, compute additional features and train a model to predict future double top events.
    
    Parameters
    ----------
    tickers: list
        list of ticker symbols to process
    param_dict: dict
        optional parameter to adjust double top detection settings for further testing
    """
    # Step 1: initialize an empty DataFrame to hold all labeled data
    labeled_data = pd.DataFrame()

    # Step 2: iterate over each ticker and process data to find candidate events and confirmed double tops
    for ticker in tickers:
        df = pd.read_csv(f"sp500/sp500/{ticker}.csv", 
                     header=0, 
                     skiprows=[1],
                     dtype={
                        "Date": str,
                        "Open": float,
                        "High": float,
                        "Low": float,
                        "Close": float,
                        "Volume": float
                        }
            )
        
        dt_candidates = detect_double_tops(df, 
                                           peak_window=param_dict['peak_window'], 
                                           peak_tolerance=param_dict['peak_tolerance'], 
                                           min_peak_gap=param_dict['min_peak_gap'], 
                                           max_peak_gap=param_dict['max_peak_gap'],
                                           min_trough_drop=param_dict['min_trough_drop'], 
                                           require_lower_second_vol=param_dict['require_lower_second_vol'])

        dt_confirmed = confirm_double_tops(df, dt_candidates, max_confirm_days=40)
        if dt_confirmed.empty:
            print(f"[{ticker}] No confirmed double tops found. Consider loosening parameters.")
            continue
        else:
            dt_confirmed = dt_confirmed.drop(columns=['confirm_date', 'confirm_price'])

        all_events = label_events(dt_candidates, dt_confirmed)

        # Inserting actual dates of events for sorting purposes to prevent leakage
        all_events = pos_to_date(df, all_events, confirm_pos=False)
        all_events['Ticker'] = ticker

        labeled_data = pd.concat([labeled_data, all_events])

    # Step 3: feature engineering (placeholder - implement as needed)
    # Peak Height Difference (raw and pct)
    labeled_data['peak_height_diff'] = labeled_data['peak2_price'] - labeled_data['peak1_price']
    labeled_data['peak_height_diff_pct'] = labeled_data['peak_height_diff'] / labeled_data['peak1_price']

    # Retracement Depth (relative to first and second peak)
    labeled_data['retracement_depth1'] = (labeled_data['peak1_price'] - labeled_data['trough_price']) / labeled_data['peak1_price']
    labeled_data['retracement_depth2'] = (labeled_data['peak2_price'] - labeled_data['trough_price']) / labeled_data['peak2_price']

    # Raw Volume Difference
    labeled_data['volume_diff'] = labeled_data['vol1'] - labeled_data['vol2']

    # Time Intervals (days between peaks and troughs)
    labeled_data['peak1_to_trough'] = (labeled_data['trough_pos'] - labeled_data['peak1_pos'])
    labeled_data['trough_to_peak2'] = (labeled_data['peak2_pos'] - labeled_data['trough_pos'])
    
    # Momentum Indicators (need to add to initial dataset; implement if needed later)

    # Step 4: Sort by peak2_date to prevent leakage
    labeled_data = labeled_data.sort_values(by='peak2_date', ascending=True).reset_index(drop=True)

    # Step 5: train/test split based on date
        # Would prefer to use PurgedGroupTimeSeriesSplit from mlfinlab as found in quant research
        # However, this package is not compatible with Python 3.12, so I implemented a basic time-based split here
        # Source: López de Prado (2018), Advances in Financial Machine Learning, Chapter 7: Purged K-Fold CV (Wiley).

    features = ['peak1_price', 'peak2_price', 'trough_price', 'peak_gap_days', 'vol1',
                'vol2', 'vol2_vol1_ratio', 'peak_height_diff', 'peak_height_diff_pct', 
                'retracement_depth1', 'retracement_depth2', 'volume_diff', 
                'peak1_to_trough', 'trough_to_peak2']
    
    X = labeled_data[features]
    y = labeled_data['label']
    times = labeled_data["peak2_date"]

    splitter = WalkForwardSplit(n_splits=5)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
    )

    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, times=times), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        fold_scores.append(score)

        print(f"Fold {fold}: score = {score:.4f}")

    print("Mean CV score:", sum(fold_scores) / len(fold_scores))
    
    return labeled_data

if __name__ == "__main__":
    directory = "./sp500/sp500"

    tickers = []
    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            tickers.append(name.split('.')[0])

    # tickers = tickers[:10]  # limit to first 10 tickers for testing
    
    labeled_data = prediction_pipeline(tickers)

    


