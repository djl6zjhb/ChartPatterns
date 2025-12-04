import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

from compute_forward_returns import compute_forward_returns
from evaluate_all import evaluate_all
from sample_random_events import sample_random_events
from ma_crossover_signals import ma_crossover_signals

def evaluating_returns_for_predictions(data_file:str, model_name:str, feature_name:str, ind_full_summary:pd.DataFrame, comp_full_summary:pd.DataFrame):
    """
    Using the trained model, evaluate the returns for the predicted positive events and compare to moving average baseline.

    Parameters
    ----------
    data_file: str
        Path to the CSV file containing predicted data with features and true labels.
    model_name: str
        Name of the trained model file (without extension).
    feature_name: str
        Name of the feature file (without extension).
    """
    # load predictions
    df = pd.read_csv(data_file)
    
    # load trained model
    model = XGBClassifier()
    model.load_model(f"{model_name}.json")
    features = joblib.load(f"{feature_name}.pkl")

    split_index = int(len(df) * 0.8)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    X_train = train_data[features]
    y_train = train_data['label']
    
    X_test = test_data[features]
    y_test = test_data['label']

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    test_data['predicted_label'] = preds
    # print(test_data[['Ticker', 'peak2_date', 'label', 'predicted_label']])
    # print(test_data.where(test_data['predicted_label'] == 1).dropna())

    predicted_events = test_data.where(test_data['predicted_label'] == 1).dropna()

    for name in os.listdir('./sp500/sp500'):
        if os.path.isfile(os.path.join('./sp500/sp500', name)):
            ticker = name.split('.')[0]

            # pulling price data for ticker
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
            
            # Filtering predicted events for this ticker
            predicted_dt_events = predicted_events[predicted_events['Ticker'] == ticker]
            
            # compute forward returns for predicted events, if any
            if not predicted_dt_events.empty:
                predicted_forward_returns = compute_forward_returns(df, predicted_dt_events, predicted = True, horizons=[5, 20, 60])
                predicted_forward_returns['symbol'] = ticker
            
            else:
                predicted_forward_returns = pd.DataFrame(columns=['peak1_pos', 'trough_pos', 'peak2_pos', 'peak1_price', 'peak2_price',
                    'trough_price', 'peak_gap_days', 'vol1', 'vol2', 'vol2_vol1_ratio',
                    'label', 'peak1_date', 'trough_date', 'peak2_date', 'Ticker',
                    'peak_height_diff', 'peak_height_diff_pct', 'retracement_depth1',
                    'retracement_depth2', 'volume_diff', 'peak1_to_trough',
                    'trough_to_peak2', 'predicted_label', 'ret_5d', 'ret_20d', 'ret_60d',
                    'symbol'])

            # compute forward returns for baseline
            rand_events = sample_random_events(df, n_events=len(predicted_dt_events), horizons=[5, 20, 60])
            rand_events["symbol"] = ticker

            ma_events = ma_crossover_signals(df, horizons=[5, 20, 60])
            ma_events["symbol"] = ticker

            # record individual and comparison return statistics
            ind_return_summary_df, comp_return_summary_df = evaluate_all(predicted_forward_returns, rand_events, ma_events, horizons=[5, 20, 60])
            ind_return_summary_df["symbol"] = ticker
            comp_return_summary_df["symbol"] = ticker

            ind_full_summary = pd.concat([ind_full_summary, ind_return_summary_df], ignore_index=True).dropna()
            comp_full_summary = pd.concat([comp_full_summary, comp_return_summary_df], ignore_index=True).dropna()
                
        # break #testing only first file
    
    return ind_full_summary, comp_full_summary

    

if __name__ == "__main__":
    
    ind_full_summary = pd.DataFrame()
    comp_full_summary = pd.DataFrame()

    ind_full_summary, comp_full_summary = evaluating_returns_for_predictions('labeled_double_top_events.csv', 'best_xgb_model', 'feature_names', ind_full_summary, comp_full_summary)

    comp_full_summary.to_csv('pred_comp_full_summary.csv')