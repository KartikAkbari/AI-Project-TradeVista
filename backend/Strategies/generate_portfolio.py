import os
from Strategies.optimize_portfolio import optimize_portfolio, load_return_data

def risk_score_to_allocation(risk):
    if risk == "low":
        return 0.05
    elif risk == "medium":
        return 0.10
    elif risk == "high":
        return 0.20

def build_user_portfolio(investment_amt, years, risk):
    tickers = ["RELIANCE", "HDFCBANK", "INFY", "ITC", "TCS", "LT", "NESTLEIND", "AXISBANK", "SBIN", "BHARTIARTL"]
    data_path = os.path.join("Data", "Processed")
    df_returns = load_return_data(data_path, tickers)

    optimized = optimize_portfolio(df_returns, risk_free_rate=0.06)
    weights = optimized["weights"]

    allocation = {}
    for ticker, weight in zip(tickers, weights):
        amt = round(weight * investment_amt, 2)
        allocation[ticker] = amt

    return {
        "allocation": allocation,
        "expected_return": optimized["expected_return"] * 100,
        "volatility": optimized["volatility"] * 100,
        "sharpe_ratio": optimized["sharpe_ratio"]
    }
