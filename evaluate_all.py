import numpy as np
import pandas as pd
from summarize_returns import summarize_returns
from compare_distributions import compare_distributions


def evaluate_all(events_dt, events_rand, events_ma, horizons=(5, 20, 60)):
    """
    Build a summary DataFrame for a single ticker over horizons and event types.

    Parameters
    ----------
    events_dt : pd.DataFrame
        DataFrame of double-top events with forward returns.
    events_rand : pd.DataFrame
        DataFrame of random baseline events with forward returns.
    events_ma : pd.DataFrame
        DataFrame of moving-average crossover baseline events with forward returns.
    horizons : tuple of int
        Forward-return horizons in trading days (default is (5, 20, 60))
    """
    individual_records = []

    # comparing returns for each horizon individually
    for h in horizons:
        col = f'ret_{h}d'
        for label, df_src in [('double_top', events_dt),
                              ('random', events_rand),
                              ('ma_crossover', events_ma)]:
            stats_dict = summarize_returns(df_src[col])
            stats_dict.update({'horizon': h, 'type': label})
            individual_records.append(stats_dict)


    # comparing return performance between event types
    comp_records=[]
    for h in horizons:
        col = f'ret_{h}d'
        dt_returns = events_dt[col].dropna()
        rand_returns = events_rand[col].dropna()
        ma_returns = events_ma[col].dropna()

        dt_ma_comp = compare_distributions(dt_returns, ma_returns)
        dt_ma_comp.update({'horizon': h, 'comparison': 'double_top_vs_ma_crossover'})
        dt_rand_comp = compare_distributions(dt_returns, rand_returns)
        dt_rand_comp.update({'horizon': h, 'comparison': 'double_top_vs_random'})
        ma_rand_comp = compare_distributions(ma_returns, rand_returns)
        ma_rand_comp.update({'horizon': h, 'comparison': 'ma_crossover_vs_random'})
        comp_records.extend([dt_ma_comp, dt_rand_comp, ma_rand_comp])




    ind_summary_df = pd.DataFrame(individual_records)
    comp_summary_df = pd.DataFrame(comp_records)
    return ind_summary_df, comp_summary_df