import os
import numpy as np
from Strategies.optimize_portfolio import load_return_data, optimize_portfolio

def advise_portfolio(user_holdings: dict):
    tickers = list(user_holdings.keys())
    total = sum(user_holdings.values())
    user_weights = [v / total for v in user_holdings.values()]

    df_returns = load_return_data(os.path.join("Data", "Processed"), tickers)
    optimal = optimize_portfolio(df_returns)

    suggested_weights = optimal["weights"]
    advice = {}
    for i, t in enumerate(tickers):
        curr = user_weights[i] * 100
        suggested = suggested_weights[i] * 100
        change = round(suggested - curr, 2)
        advice[t] = {
            "current_weight": round(curr, 2),
            "suggested_weight": round(suggested, 2),
            "change": change
        }

    current_sharpe = (np.dot(user_weights, df_returns.mean()) - 0.01) / np.std(df_returns.values) if np.std(df_returns.values) != 0 else 0

    return {
        "advice": advice,
        "sharpe_ratio_current": current_sharpe,
        "sharpe_ratio_optimal": optimal["sharpe_ratio"]
    }