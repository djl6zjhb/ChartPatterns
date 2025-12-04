import pandas as pd
import altair as alt

def comparative_return_analysis(filename: str, image_name: str):
    """
    Using the cross-event analysis generated from run_double_top_pipeline,
    analyze and compare the return statistics across event types and horizons.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the comparative return statistics.
    image_name : str
        Name of the image file to save the plot.
    """
    alt.renderers.enable('default')

    df = pd.read_csv(filename, index_col=0)
    fived_df = df[df['horizon'] == 5]
    twentyd_df = df[df['horizon'] == 20]
    sixtyd_df = df[df['horizon'] == 60]

    print(fived_df.head())

    # plotting p-value distributions
    base = alt.Chart(sixtyd_df).mark_bar(opacity=0.8).encode(
        x=alt.X('mw_p:Q',
                bin=alt.Bin(maxbins=20),
                title='Mann-Whitney p-value'),
        y=alt.Y('count()',
                title='Number of tickers')
    ).properties(
        width=180,
        height=150
    )

    chart = base.facet(
        column=alt.Column('comparison:N', title=None)
    ).properties(
        title=f'Mann-Whitney p-value distributions by comparison (horizon=60)'
    )

    chart.save(image_name)


    # print(df)

if __name__ == "__main__":
    comparative_return_analysis('comp_returns_all_horizons.csv', 'comp_return_pvalue_h60.html')
    # comparative_return_analysis('pred_comp_full_summary.csv', 'pred_comp_return_pvalue_h5.html')