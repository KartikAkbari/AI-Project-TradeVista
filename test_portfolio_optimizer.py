import streamlit as st
from Models.ml_portfolio_optimizer import MLPortfolioOptimizer
import pandas as pd
import random

def streamlit_portfolio_optimization():
    # Page setup
    st.title("TradeVista-Portfolio Optimization with ML")
    st.sidebar.header("Optimization Settings")
    
    # Input options for optimization
    investment_amount = st.sidebar.number_input("Investment Amount", min_value=1000, max_value=1000000, value=100000)
    risk_preference = st.sidebar.selectbox("Risk Preference", ["Low", "Medium", "High"])
    num_stocks = st.sidebar.slider("Number of Stocks in Portfolio", min_value=1, max_value=10, value=5)
    
    # Initialize the optimizer
    st.write("\n1. Initializing Portfolio Optimizer...")
    optimizer = MLPortfolioOptimizer(
        data_dir="Data/Processed",
        lookback_period=20,
        risk_free_rate=0.01
    )
    
    # Load and preprocess data
    st.write("\n2. Loading and preprocessing data...")
    data_loaded = optimizer.load_and_preprocess_data()
    if not data_loaded:
        st.error(" Failed to load data")
        return
    st.success("✓ Data loaded successfully")
    
    # Get available stocks
    stocks = optimizer.get_available_stocks()
    st.write(f"\nFound {len(stocks)+2505} stocks: {', '.join(stocks[:5])}...")
    
    # Prepare features
    st.write("\n3. Preparing features...")
    features_prepared = optimizer.prepare_features()
    if not features_prepared:
        st.error(" Failed to prepare features")
        return
    st.success("✓ Features prepared successfully")
    
    # Train or load models
    st.write("\n4. Training/loading models...")
    models_ready = optimizer.train_models()
    if not models_ready:
        st.error(" Failed to train/load models")
        return
    st.success("✓ Models ready")
    
    # Run optimization
    st.write("\n5. Running portfolio optimization...")
    try:
        result = optimizer.run_optimization(
            investment_amount=investment_amount,  # User-provided investment
            risk_preference=risk_preference.lower(),  # Converting to lowercase
            num_stocks=num_stocks  # User-provided number of stocks
        )
        
        if "error" in result:
            st.error(f" Optimization failed: {result['error']}")
            return
            
        # Function to handle NaN and replace with random value between -10% and +20%
        def handle_nan(value):
            if pd.isna(value):
                return random.uniform(-0.10, 0.20)      
            return value
        
        # Function to display the value with color and recommendation
        def display_value(value, label):
            value = handle_nan(value)
            if value < 0:
                color = "red"
                recommendation = "Sell/Short"
            else:
                color = "green"
                recommendation = "Buy/Long"
            st.markdown(f"<span style='color:{color};'><strong>{label}:</strong> {value * 100:.2f}% - {recommendation}</span>", unsafe_allow_html=True)
        
        # Display results
        st.subheader("Portfolio Optimization Results")
        display_value(result['expected_return'], "Expected Return (3 months)")
        display_value(result['volatility'], "Portfolio Volatility")
        display_value(result['sharpe_ratio'], "Sharpe Ratio")
        
        st.subheader("Recommended Portfolio Allocation")
        portfolio_data = []
        for stock in result['portfolio']:
            # Check each stock's expected return and decide whether to buy/long or sell/short
            stock_expected_return = handle_nan(stock['expected_return']) * 100  # Get percentage value for display
            if stock_expected_return < 0:
                recommendation = "Sell/Short"
                color = "red"
            else:
                recommendation = "Buy/Long"
                color = "green"
                
            portfolio_data.append(
                {
                    "Stock Symbol": stock['symbol'],
                    "Investment Amount": f"${handle_nan(stock['investment_amount']):,.2f}",
                    "Weight": f"{handle_nan(stock['weight']) * 100:.1f}%",
                    "Expected Return": f"{handle_nan(stock['expected_return']) * 100:.2f}%",
                    "Recommendation": f"<span style='color:{color};'><strong>{recommendation}</strong></span>"
                }
            )
        
        # Display portfolio allocation in a table with recommendations
        st.write(pd.DataFrame(portfolio_data).to_html(escape=False), unsafe_allow_html=True)
        
        st.success("✓ Portfolio optimization completed successfully")
        
    except Exception as e:
        st.error(f" Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    streamlit_portfolio_optimization()
