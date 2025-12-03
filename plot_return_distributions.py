import matplotlib.pyplot as plt

def plot_return_distributions(dt_events, rand_events, ma_events, ticker, horizon=20):
    col = f'ret_{horizon}d'
    plt.figure(figsize=(8, 5))
    plt.hist(rand_events[col].dropna(), bins=30, alpha=0.5, label='Random')
    plt.hist(dt_events[col].dropna(), bins=30, alpha=0.5, label='Double top')
    plt.hist(ma_events[col].dropna(), bins=30, alpha=0.5, label='MA crossover')
    plt.axvline(0, linestyle='--')
    plt.title(f'{horizon}-day return distribution')
    plt.xlabel('Return')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{ticker}_return_distribution.png", dpi=300, bbox_inches='tight')
