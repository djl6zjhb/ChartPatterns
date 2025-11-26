import pandas as pd
import numpy as np

def evaluate_results(filepath) -> pd.DataFrame:
    """
    Evaluate results from a CSV file containing event summaries.
    """
    summary = pd.read_csv(filepath)
    return summary

if __name__ == "__main__":
    filepath = './double_top_events_all.csv'
    summary_df = evaluate_results(filepath)
    