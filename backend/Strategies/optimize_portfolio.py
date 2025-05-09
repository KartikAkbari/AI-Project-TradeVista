import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def load_return_data(data_dir, tickers, window_days=252):
    returns = {}
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}.csv")
        df = pd.read_csv(path)
        df['Return'] = df['close'].pct_change()
        returns[ticker] = df['Return'].dropna().values[-window_days:]
    return pd.DataFrame(returns)

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_return, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_vol

def optimize_portfolio(returns_df, risk_free_rate=0.01):
    num_assets = len(returns_df.columns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    init_guess = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    result = minimize(negative_sharpe_ratio, init_guess,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return {
        "weights": result.x,
        "expected_return": portfolio_performance(result.x, mean_returns, cov_matrix)[0],
        "volatility": portfolio_performance(result.x, mean_returns, cov_matrix)[1],
        "sharpe_ratio": -(result.fun)
    }

if __name__ == "__main__":
    data_path = os.path.join("Data", "Processed")
    selected_tickers = ["RELIANCE", "INFY", "TCS", "HDFC", "ITC"]
    
    df_returns = load_return_data(data_path, selected_tickers)
    result = optimize_portfolio(df_returns)

    print("\n📊 Optimized Portfolio:")
    for ticker, weight in zip(selected_tickers, result['weights']):
        print(f"{ticker}: {weight:.2%}")
    
    print(f"\nExpected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
