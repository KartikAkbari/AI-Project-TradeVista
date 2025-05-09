import os
import numpy as np
from Strategies.optimize_portfolio import load_return_data

def compare_stocks(ticker_list):
    data_path = os.path.join("Data", "Processed")
    df_returns = load_return_data(data_path, ticker_list)

    summary = {}
    for ticker in ticker_list:
        returns = df_returns[ticker]
        mean_ret = np.mean(returns)
        vol = np.std(returns)
        sharpe = (mean_ret - 0.01) / vol if vol != 0 else 0
        summary[ticker] = {
            "avg_return": mean_ret * 100,
            "volatility": vol * 100,
            "sharpe_ratio": sharpe
        }
    return summary