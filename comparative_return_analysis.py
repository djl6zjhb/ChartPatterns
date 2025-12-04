import pandas as pd
import altair as alt

def comparative_return_analysis(filename: str):
    """
    Using the cross-event analysis generated from run_double_top_pipeline,
    analyze and compare the return statistics across event types and horizons.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the comparative return statistics.
    """
    alt.renderers.enable('default')

    data = pd.read_csv(filename, index_col=0)
    fived_df = data[data['horizon'] == 5]
    twentyd_df = data[data['horizon'] == 20]
    sixtyd_df = data[data['horizon'] == 60]

    dfs = [(fived_df,5,'#0000FF'), (twentyd_df,20,'#FF0000'), (sixtyd_df,60,'#008000')]

    plots = []

    for tup in dfs: 
        # plotting p-value distributions
        base = alt.Chart(tup[0]).mark_bar(opacity=0.5, color=tup[2]).encode(
            x=alt.X('mw_p:Q',
                    bin=alt.Bin(maxbins=20),
                    title='Mann-Whitney p-value'),
            y=alt.Y('count()',
                    title='Number of tickers')
        ).properties(
            width=240,
            height=150
        )

        chart = base.facet(
            column=alt.Column('comparison:N', title=None)
        ).properties(
            title=alt.TitleParams(
                text=f'horizon = {tup[1]} day returns',
                offset=5
            )
        )

        plots.append(chart)
    
    combined = alt.vconcat(
        plots[0],
        plots[1],
        plots[2],
        spacing=15
        ).resolve_scale(
            x='shared'
        ).properties(
            title = alt.TitleParams(
                text='Mann-Whitney p-value distributions across comparison groups',
                anchor='middle',
                fontSize=18,
                offset=10
            )
        ).configure_bar(binSpacing = 0)

    combined.save('predicted_return_pval_hist.html')


if __name__ == "__main__":
    comparative_return_analysis('comp_returns_all_horizons.csv')
    # comparative_return_analysis('pred_comp_full_summary.csv', 'pred_comp_return_pvalue_h5.html')