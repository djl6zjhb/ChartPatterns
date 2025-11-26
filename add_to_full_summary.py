import pandas as pd

def add_to_full_summary(full_summary: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes single ticker summary_df and adds relevant results to full_summary DataFrame.

    Full summary columns:
    'Ticker','5-day success','5-day return','5-day p-value','5-day hit ratio','5-day sharpe','5-day cohen d',
            '20-day success','20-day return','20-day p-value','20-day hit ratio','20-day sharpe','20-day cohen d',
            '60-day success','60-day return','60-day p-value','60-day hit ratio','60-day sharpe','60-day cohen d'

    """
    five_day = summary_df.iloc[0].drop(labels=['horizon','type','n','std','t_stat'])
    twenty_day = summary_df.iloc[3].drop(labels=['horizon','type','n','std','t_stat'])
    sixty_day = summary_df.iloc[6].drop(labels=['horizon','type','n','std','t_stat'])

    five_day_success = (five_day['mean'] < 0) and (five_day['p_value'] < 0.05) and (five_day['sharpe'] > 0.5 or abs(five_day['cohen_d']) > 0.3)
    twenty_day_success = (twenty_day['mean'] < 0) and (twenty_day['p_value'] < 0.05) and (twenty_day['sharpe'] > 0.5 or abs(twenty_day['cohen_d']) > 0.3)
    sixty_day_success = (sixty_day['mean'] < 0) and (sixty_day['p_value'] < 0.05) and (sixty_day['sharpe'] > 0.5 or abs(sixty_day['cohen_d']) > 0.3)
    print(f"5-day success: {five_day_success}, 20-day success: {twenty_day_success}, 60-day success: {sixty_day_success}")
    

    full_summary.loc[five_day['symbol']] = {'5-day success': five_day_success,
                                        '5-day return': five_day['mean'],
                                        '5-day p-value': five_day['p_value'],
                                        '5-day hit ratio': five_day['hit_ratio_neg'],
                                        '5-day sharpe': five_day['sharpe'],
                                        '5-day cohen d': five_day['cohen_d'],
                                        '20-day success': twenty_day_success,
                                        '20-day return': twenty_day['mean'],
                                        '20-day p-value': twenty_day['p_value'],
                                        '20-day hit ratio': twenty_day['hit_ratio_neg'],
                                        '20-day sharpe': twenty_day['sharpe'],
                                        '20-day cohen d': twenty_day['cohen_d'],
                                        '60-day success': sixty_day_success,
                                        '60-day return': sixty_day['mean'],
                                        '60-day p-value': sixty_day['p_value'],
                                        '60-day hit ratio': sixty_day['hit_ratio_neg'],
                                        '60-day sharpe': sixty_day['sharpe'],
                                        '60-day cohen d': sixty_day['cohen_d']
                                        }
    return full_summary