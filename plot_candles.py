import numpy as np
import pandas as pd
import mplfinance as mpf
from run_double_top_pipeline import run_double_top_pipeline
from pos_to_date import pos_to_date

def plot_candles(
    df,
    dt_events,
    volume=True,
    mav=None,
    date_range=None,
    fig_size=(8, 4),
    save_path=None,
    addplot_size = [200,200,180,240]
):
    """
    Plot a candlestick chart and overlay Double Top peaks, troughs, and confirmations.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex and columns:
        ['Open', 'High', 'Low', 'Close', 'Volume'].

    dt_events : pd.DataFrame
        Events DataFrame with at least:
        - 'peak1_date'
        - 'peak2_date'
        - 'trough_date'
        - 'confirm_date'

    title : str
        Chart title.

    volume : bool
        Whether to include the volume subplot.

    mav : tuple or None
        Moving averages to draw, e.g. (20, 50).

    date_range : tuple(str or Timestamp, str or Timestamp) or None
        Optional (start, end) to zoom, e.g. ('2020-01-01', '2021-01-01').
    
    fig_size : tuple(int, int)
        Figure size in inches (width, height).

    save_path : str or None
        If provided, saves the figure to this path.

    Assists in visualizing detected Double Top patterns on candlestick charts.
    
    Function originally by ChatGPT, modified by me to make sure that DTs were being detected correctly. 
    The purpose is to visualize the detected double top patterns on candlestick charts to ensure that the detection algorithm is functioning as intended.
    This is not used in the main pipeline, but is useful for debugging and validation.
    I have little experience with mplfinance, so I relied heavily on ChatGPT to generate this code.
    """
    
    dt_events = pos_to_date(df, dt_events)
    
    # Ensure datetime types in dt_events
    for col in ["peak1_date", "peak2_date", "trough_date", "confirm_date"]:
        if col in dt_events.columns:
            dt_events[col] = pd.to_datetime(dt_events[col])

    # Optionally zoom to a date range
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_plot = df.loc[(df.index >= start) & (df.index <= end)].copy()
        events_plot = dt_events[
            (dt_events["confirm_date"] >= start) & (dt_events["confirm_date"] <= end)
        ].copy()
    else:
        df_plot = df.copy()
        events_plot = dt_events.copy()

    if df_plot.empty:
        print("No data in selected date range.")
        return

    # Initialize marker series (index aligned with df_plot)
    idx = df_plot.index
    peak1_series = pd.Series(np.nan, index=idx)
    peak2_series = pd.Series(np.nan, index=idx)
    trough_series = pd.Series(np.nan, index=idx)
    confirm_series = pd.Series(np.nan, index=idx)

    # Fill marker positions from dt_events
    for _, row in events_plot.iterrows():
        # Peaks plotted slightly above the high for visibility
        for col, series in [("peak1_date", peak1_series),
                            ("peak2_date", peak2_series)]:
            d = row[col]
            if d in df_plot.index:
                price = df_plot.loc[d, "High"]
                series.loc[d] = price * 1.01  # 1% above high

        # Trough plotted slightly below the low
        d_trough = row["trough_date"]
        if d_trough in df_plot.index:
            price = df_plot.loc[d_trough, "Low"]
            trough_series.loc[d_trough] = price * 0.99  # 1% below low

        # Confirmation plotted slightly above the close
        d_conf = row["confirm_date"]
        if d_conf in df_plot.index:
            price = df_plot.loc[d_conf, "Close"]
            confirm_series.loc[d_conf] = price * 1.01  # 1% above close

    # Build addplots (big, bold markers so they stand out)
    apds = []

    # Peak 1 markers (bright green ▲ with black outline)
    if not peak1_series.isna().all():
        apds.append(
            mpf.make_addplot(
                peak1_series,
                type="scatter",
                marker="^",
                markersize=addplot_size[0],
                color="lime",
                edgecolors="black",
                linewidths=1.5,
            )
        )

    # Peak 2 markers (dark green ▲)
    if not peak2_series.isna().all():
        apds.append(
            mpf.make_addplot(
                peak2_series,
                type="scatter",
                marker="^",
                markersize=addplot_size[1],
                color="green",
                edgecolors="black",
                linewidths=1.5,
            )
        )

    # Trough markers (large orange ●)
    if not trough_series.isna().all():
        apds.append(
            mpf.make_addplot(
                trough_series,
                type="scatter",
                marker="o",
                markersize=addplot_size[2],
                color="orange",
                edgecolors="black",
                linewidths=1.5,
            )
        )

    # Confirmation markers (huge red ▼ with black outline)
    if not confirm_series.isna().all():
        apds.append(
            mpf.make_addplot(
                confirm_series,
                type="scatter",
                marker="v",
                markersize=addplot_size[3],
                color="red",
                edgecolors="black",
                linewidths=1.5,
            )
        )

    # Final plot call
    mpf.plot(
        df_plot,
        type="candle",
        style="yahoo",
        volume=volume,
        mav=mav if mav is not None else (),
        addplot=apds,
        figsize=fig_size,
        tight_layout=True,
        savefig=save_path,
        warn_too_much_data=1000000000
    )

if __name__ == "__main__":
    df_aapl = pd.read_csv("./sp500/sp500/AAPL.csv", 
                 parse_dates=["Date"], 
                 index_col="Date",
                 dtype={
                    "Open": float,
                    "High": float,
                    "Low": float,
                    "Close": float,
                    "Volume": float
                },
                header = 0,
                skiprows=[1]
    )

    ind_return_summary_df, comp_return_summary_df = pd.DataFrame(), pd.DataFrame()
    dt_events, rand_events, ma_events, summary_df, text_summary, full_summary = run_double_top_pipeline('AAPL',ind_full_summary=ind_return_summary_df, comp_full_summary=comp_return_summary_df)
    plot_candles(df_aapl, dt_events, mav=(20,50), fig_size=(12, 6), save_path="aapl_chart.png", addplot_size = [50,50,45,60])
    plot_candles(df_aapl, dt_events, mav=None, date_range=('2020-Dec-16', '2021-Jun-16'), save_path="aapl_chart_zoomed.png")

    plot_candles(df_aapl, dt_events, mav=(20,50), fig_size=(12, 6), date_range=('2020-Dec-16', '2024-Jun-16'), save_path="aapl_chart_zoomed_ma.png", addplot_size = [0,0,0,0])