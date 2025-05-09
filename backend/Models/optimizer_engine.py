import numpy as np
import pandas as pd

class OptimizerEngine:
    def __init__(self, risk_free_rate=0.01):
        """
        Initializes the optimizer with given parameters.
        
        Args:
            risk_free_rate: The risk-free rate to use in the Sharpe ratio calculation.
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, data, investment, risk_level, top_n):
        """
        Perform portfolio optimization based on the selected stocks.

        Args:
            data: Dictionary with stock data (processed stock data for each symbol)
            investment: Total amount of money to invest
            risk_level: Risk preference ('low', 'medium', 'high')
            top_n: Number of stocks to select for the portfolio

        Returns:
            A dictionary with the optimized portfolio and performance metrics
        """
        try:
            # Data preparation: Extract relevant stock features (e.g., returns, volatility, etc.)
            stock_data = pd.DataFrame()

            # Gather returns data from the available stocks
            for symbol, df in data.items():
                stock_data[symbol] = df['returns']  # You may want to consider additional features here

            # Check if there's enough data to compute correlation
            if stock_data.empty or stock_data.isna().any().any():
                raise ValueError("Stock data contains missing values, cannot compute correlation matrix")

            # Calculate the correlation matrix of stock returns
            correlation_matrix = stock_data.corr()

            # Optimize portfolio using some method (e.g., Mean-Variance Optimization)
            weights = self.mean_variance_optimization(correlation_matrix, risk_level, top_n)

            # Calculate portfolio performance
            expected_return, volatility, sharpe_ratio = self.calculate_portfolio_performance(weights, data)

            # Build portfolio dictionary to return
            portfolio = {
                "weights": weights,
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            }

            return portfolio

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return {"error": f"Optimization failed: {str(e)}"}


    def mean_variance_optimization(self, correlation_matrix, risk_level, top_n):
        """
        Mean-Variance Optimization for portfolio allocation.

        Args:
            correlation_matrix: Correlation matrix of stock returns
            risk_level: Risk preference ('low', 'medium', 'high')
            top_n: Number of stocks to select for the portfolio
            
        Returns:
            Optimized portfolio weights
        """
        num_stocks = len(correlation_matrix)

        # Placeholder: Generate random weights (replace this with actual optimization)
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)  # Normalize the weights to sum to 1

        return weights

    def calculate_portfolio_performance(self, weights, data):
        """
        Calculate the expected return, volatility, and Sharpe ratio of the portfolio.

        Args:
            weights: Portfolio weights
            data: Stock data
        
        Returns:
            Expected return, volatility, and Sharpe ratio
        """
        returns = []
        volatility = []
        
        for i, (symbol, df) in enumerate(data.items()):
            # Calculate expected return (mean return)
            expected_return = df['returns'].mean() * 252  # Annualized return (assuming 252 trading days)
            returns.append(expected_return)
            
            # Calculate volatility (annualized standard deviation)
            stock_volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
            volatility.append(stock_volatility)
        
        # Convert to numpy arrays for vectorized operations
        returns = np.array(returns)
        volatility = np.array(volatility)
        
        # Calculate expected portfolio return
        portfolio_return = np.dot(weights, returns)
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(correlation_matrix, weights)))
        
        # Calculate Sharpe ratio (assuming the risk-free rate is provided)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
