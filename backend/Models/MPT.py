import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as sco

# 📌 Load multiple stock price CSV files
input_folder = "D:\AI Project\AI-Driven Hedge Fund Optimization System\Data\Processed"
stock_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Read stock data and merge into a single DataFrame
df_list = []
for file in stock_files:
    stock_df = pd.read_csv(os.path.join(input_folder, file), usecols=['datetime', 'close'])
    stock_df.rename(columns={'close': file.replace('.csv', '')}, inplace=True)
    stock_df.set_index('datetime', inplace=True)
    df_list.append(stock_df)

# Merge all stock data into one DataFrame
portfolio_df = pd.concat(df_list, axis=1).dropna()
portfolio_df.index = pd.to_datetime(portfolio_df.index)

# 📌 Compute Daily Returns
returns = portfolio_df.pct_change().dropna()

# 📌 Define Portfolio Optimization Functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)  # Expected return
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility
    sharpe = returns / risk  # Sharpe Ratio
    return returns, risk, sharpe

def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]  # Minimize negative Sharpe

# 📌 Compute Expected Returns & Covariance Matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# 📌 Set Optimization Constraints
num_assets = len(mean_returns)
weights_init = np.ones(num_assets) / num_assets  # Equal weight start
bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights must be 1

# 📌 Optimize Portfolio (Find Best Weights)
opt_result = sco.minimize(negative_sharpe_ratio, weights_init, args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x

# 📌 Print Optimal Portfolio Weights
optimal_portfolio = pd.DataFrame({'Stock': portfolio_df.columns, 'Weight': optimal_weights})
print("✅ Optimal Portfolio Allocation:\n", optimal_portfolio)

# 📌 Efficient Frontier Plot
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    random_weights = np.random.random(num_assets)
    random_weights /= np.sum(random_weights)
    ret, risk, sharpe = portfolio_performance(random_weights, mean_returns, cov_matrix)
    results[0, i] = ret
    results[1, i] = risk
    results[2, i] = sharpe

# 📌 Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier with Optimal Portfolio")
plt.scatter(*portfolio_performance(optimal_weights, mean_returns, cov_matrix)[1::-1], c='red', marker='*', s=200, label="Optimal Portfolio")
plt.legend()
plt.show()
