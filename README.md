# Empirical Validation of Double Tops, Confirmed and Predicted events

Applying strict classifcation rules to identify and evaluate the stock market returns of double top chart patterns

## Project Summary

### Overview
This project takes 10 years of S&P 500 stock data, identifies candidate double top events, confirms that the candidate events complete the pattern by closing below the neckline, and evaluates returns at 5, 20, and 60 days.

### Detection and return evaluation
run_double_top_pipeline.py will load the locally saved stock data, detect all cnadidate events, onfirm the double top, and evaluate returns against moving average crossover and random index trading. detect_double_tops.py does the detection on a per-ticker basis, and confirm_double_tops.py will confirm whether or not the candidate event closes below the neckline within a given window of days.

Detection criteria and confirmation window can be varied in function arguments

### Calculating returns
compute_forward_returns.py handles return calculation for any series of events. Generalized to be used for all events of interest: double tops, ma_crossover, and random. 

### Predicting double tops
Using data available after the second peak, an XGBoost gradient boosted classifier predicted whether the candidate pattern would complete. Early action on the double top price drop could lead to better returns when compared to baseline. 

prediction_pipeline.py runs the full prediction pipeline, including calculating returns on the predicted double tops and compares them to moving averages and random.

Model performance validated by permutation test in permutation_test_all_metrics.py

### Visualizations
plot_candles.py: plots candlestick charts. Optionally can label double top events, plot moving averages, and zoom in on a specified date range
plot_return_distributions.py: plot return distribution for all double top events, ma_crossover events, and random events for a given ticker
comparative_return_analysis: plots Altair histograms of the p-values from the Mann-Whitney U-test that evaluated return distributions of double top events vs ma_crossover vs random

## Getting Started

### Data Access
Data can be found in 'sp500/sp500.' Data is publicly available and originally fetched using yfinance

Each pipeline file loads data as necessary using local paths.

### Dependencies
See requirements.txt

## Key Findings
Confirmed double tops do not reliably predict market movement, they do not generate significantly different returns compared to moving average crossover or random index trading

The gradient-boosted classifier cand accurately and meaningfully predict when a double top pattern will complete when presented with a candidate event. The model performs adequately, with room for improvement with better feature engineering.

The predicted double top events perform slightly better on a per-ticker basis when compared to the confirmed double top events; however, they still do not achieve signifiance. 

In total, double tops alone do not predict market movement in a meaningful way. 

## Authors

Dan Lagalante
lagald@umich.edu