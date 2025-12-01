import pandas as pd
import os
from add_to_full_summary import add_to_full_summary
from detect_double_tops import detect_double_tops
from confirm_double_tops import confirm_double_tops
from compute_forward_returns import compute_forward_returns
from sample_random_events import sample_random_events
from ma_crossover_signals import ma_crossover_signals
from evaluate_all import evaluate_all
from plot_return_distributions import plot_return_distributions
from plot_parameter_heatmap import plot_parameter_heatmap
from summarize_pattern_performance import summarize_pattern_performance
from label_events import label_events

def run_double_top_pipeline(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    horizons=(5, 20, 60),
    double_top_params: dict = {},
    save_prefix: str | None = None,
    make_plots: bool = False,
    full_summary: pd.DataFrame = pd.DataFrame()
):
    """
    End-to-end pipeline to test Double Top performance for one ticker.

    Steps:
      1. Load stock data from local CSV files.
      2. Detect Double Tops (strict two-peak pattern with trough).
      3. Confirm when price closes below the trough (neckline).
      4. Compute forward returns for double-top events (5/20/60 days by default).
      5. Build baselines: random timestamps + bearish MA crossover.
      6. Compute summary stats & significance tests.
      7. Optionally save CSVs & generate plots.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPY", "AAPL".
    start : str
        Start date for data (YYYY-MM-DD).
    end : str
        End date for data (YYYY-MM-DD).
    horizons : tuple of int
        Forward-return horizons in trading days.
    save_prefix : str or None
        If not None, will save:
            {save_prefix}_events.csv
            {save_prefix}_summary.csv
    make_plots : bool
        If True, will show a distribution plot and a simple parameter heatmap.

    Returns
    -------
    dt_events : pd.DataFrame
        Double-top events with features + forward returns.
    rand_events : pd.DataFrame
        Random baseline events with forward returns.
    ma_events : pd.DataFrame
        Moving-average crossover baseline events.
    summary_df : pd.DataFrame
        Summary statistics across horizons & event types.
    text_summary_20d : str
        Plain-language summary at the 20-day horizon (if available).
    """

    # 1) pull data from local files
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

    # 2) detect + confirm double tops
    dt_candidates = detect_double_tops(df, peak_window=3, peak_tolerance=0.01,min_peak_gap=15, max_peak_gap=30, min_trough_drop=0.03, require_lower_second_vol=True)

    dt_confirmed = confirm_double_tops(df, dt_candidates, max_confirm_days=40)

    if dt_confirmed.empty:
        print(f"[{ticker}] No confirmed double tops found. Consider loosening parameters.")
        dt_events = pd.DataFrame()
        rand_events = pd.DataFrame()
        ma_events = pd.DataFrame()
        summary_df = pd.DataFrame()
        return dt_events, dt_candidates, rand_events, ma_events, summary_df, "", full_summary

    # optional weekly-level top tagging; not implemented correctly yet
    # dt_confirmed = tag_weekly_top(df, dt_confirmed)

    # 3) forward returns for double-top events
    dt_events = compute_forward_returns(df, dt_confirmed, horizons=horizons)
    dt_events["symbol"] = ticker
    dt_events["type"] = "double_top"

    # 4) baselines
    rand_events = sample_random_events(df, n_events=len(dt_events), horizons=horizons)
    rand_events["symbol"] = ticker

    ma_events = ma_crossover_signals(df, horizons=horizons)
    ma_events["symbol"] = ticker

    # 5) summary statistics
    summary_df = evaluate_all(dt_events, rand_events, ma_events, horizons=horizons)
    summary_df["symbol"] = ticker
    full_summary = add_to_full_summary(full_summary, summary_df)

    # 6) CSVs (if requested)
    if save_prefix is not None:
        dt_events.to_csv(f"{save_prefix}_events.csv", index=False)
        summary_df.to_csv(f"{save_prefix}_summary.csv", index=False)
        print(f"Saved events to {save_prefix}_events.csv and summary to {save_prefix}_summary.csv")

    # 7) plots (optional)
    if make_plots:
        # choose the middle horizon for prettier plots if available
        mid_h = horizons[len(horizons) // 2]
        plot_return_distributions(dt_events, rand_events, horizon=mid_h)
        plot_parameter_heatmap(dt_events, horizon=mid_h)

    # 8) plain-language summary at 20d if available
    for h_for_text in horizons:
        try:
            text_summary = summarize_pattern_performance(summary_df, horizon=h_for_text)
            # print("\nText summary:\n")
            # In test runs, I want to see how oftwen I can mark the double top as a success 
            if "NOT" not in text_summary:
                print(text_summary)
        except Exception as e:
            text_summary = f"Could not generate text summary (likely too few events). Error: {e}"
            print(text_summary)

    return dt_events, dt_candidates, rand_events, ma_events, summary_df, text_summary, full_summary


if __name__ == "__main__":
    dt_events = None
    dt_candidates = None
    rand_events = None
    ma_events = None
    summary_df = None
    text_summary = None

    root_path = './sp500/sp500/'
    full_summary = pd.DataFrame(columns=['5-day success','5-day return','5-day p-value','5-day hit ratio','5-day sharpe','5-day cohen d',
                                         '20-day success','20-day return','20-day p-value','20-day hit ratio','20-day sharpe','20-day cohen d',
                                         '60-day success','60-day return','60-day p-value','60-day hit ratio','60-day sharpe','60-day cohen d'])

    for current_dir, subdirs, files in os.walk(root_path):
        for fname in files:
            symbol = fname.split('.')[0]
            print(symbol)
            dt_events, dt_candidates, rand_events, ma_events, summary_df, text_summary, full_summary = run_double_top_pipeline(symbol, full_summary=full_summary)
            break  # only process the first file in this directory
        break  # stop after the top-level directory iteration
    
    print(dt_events)
    print(dt_candidates)
    dt_candidates = label_events(dt_candidates, dt_events)
    print(dt_candidates)

    # full_summary.to_csv('double_top_events_all.csv', index=True)