import os
import pandas as pd
from Models.ml_portfolio_optimizer import MLPortfolioOptimizer
import json

def run_batch_optimization():
    # Initialize the optimizer
    optimizer = MLPortfolioOptimizer(
        data_dir="Data/Processed",
        lookback_period=20,
        risk_free_rate=0.01
    )

    # Load and preprocess data
    data_loaded = optimizer.load_and_preprocess_data()
    if not data_loaded:
        print("❌ Failed to load data")
        return

    # Get available stocks (we assume you have a list of 2500 stocks in your dataset)
    all_stocks = optimizer.get_available_stocks()
    total_stocks = len(all_stocks)
    
    results = []

    for i in range(0, total_stocks, 5):  # Process in batches of 5 stocks
        selected_stocks = all_stocks[i:i + 5]
        print(f"\n🚀 Running optimization for batch {i // 5 + 1}: {selected_stocks}")

        try:
            # Manually set the selected stocks in the optimizer
            optimizer.selected_stocks = selected_stocks

            # Now run the optimization
            result = optimizer.run_optimization(
                investment_amount=100000,  # $100,000 investment
                risk_preference="medium",  # medium risk tolerance
                num_stocks=5  # Include top 5 stocks in each batch
            )

            # If the result is valid, store it
            if "error" not in result:
                results.append(result)
            else:
                print(f"⚠️ Batch {i//5+1} failed: {result['error']}")

        except Exception as e:
            print(f"❌ Error in batch {i//5+1}: {e}")
            continue

    # Save results to a file (CSV or JSON)
    with open('portfolio_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n🎉 All batches processed. Results saved to 'portfolio_optimization_results.json'.")

if __name__ == "__main__":
    run_batch_optimization()
