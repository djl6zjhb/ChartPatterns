def summarize_pattern_performance(summary_df, horizon=20):
    row = summary_df[(summary_df['type'] == 'double_top') &
                     (summary_df['horizon'] == horizon)].iloc[0]
    row_rand = summary_df[(summary_df['type'] == 'random') &
                          (summary_df['horizon'] == horizon)].iloc[0]

    
    ticker = row['symbol']
    mean = row['mean']
    p = row['p_value']
    hit = row['hit_ratio_neg']
    sharpe = row['sharpe']
    d = row['cohen_d']

    mean_rand = row_rand['mean']
    hit_rand = row_rand['hit_ratio_neg']

    text = []

    text.append(f"Performance summary for {ticker}:")
    text.append(f"For {horizon}-day returns after confirmed double tops (n={int(row['n'])}), "
                f"the average return is {mean:.3%} with p-value {p:.3f}.")
    text.append(f"The hit ratio (probability of a negative return) is {hit:.2%} "
                f"vs {hit_rand:.2%} for random dates.")
    text.append(f"The Sharpe ratio is {sharpe:.2f} and the effect size (Cohen's d) is {d:.2f}.")
    text.append("By the project’s working rule, the pattern 'works' at this horizon if:\n"
                "- average return < 0 and p < 0.05;\n"
                "- hit ratio > random;\n"
                "- Sharpe > 0.5 or |d| > 0.3.")

    works = (mean < 0) and (p < 0.05) and (hit > hit_rand) and ((sharpe > 0.5) or (abs(d) > 0.3))
    text.append(f"Based on the current sample, the double top pattern "
                f"{'meets' if works else 'does NOT meet'} these criteria at {horizon} days.")

    return "\n".join(text)
