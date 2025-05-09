from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
from Strategies.generate_portfolio import build_user_portfolio
from Strategies.compare_stocks import compare_stocks
from Strategies.portfolio_advisor import advise_portfolio
from Strategies.optimize_portfolio import optimize_portfolio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from Models.ml_portfolio_optimizer import MLPortfolioOptimizer


app = FastAPI(title="AI Hedge Fund Optimizer")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML Portfolio Optimizer with the correct data directory
ml_optimizer = MLPortfolioOptimizer(data_dir="Data/Processed")

class UserRequest(BaseModel):
    amount: float
    duration_months: int
    risk: str  # low, medium, high
    
# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

class PortfolioRequest(BaseModel):
    investment_amount: float
    investment_time: int  # in months
    risk_preference: str  # "low", "medium", "high"
    current_portfolio: Optional[List[dict]] = None

class StockComparisonRequest(BaseModel):
    stock_symbols: List[str]

class PortfolioAnalysisRequest(BaseModel):
    portfolio: List[Dict[str, Any]]
    risk_preference: str

@app.get("/")
async def root():
    return {"message": "Welcome to AI Hedge Fund Optimizer API"}

@app.get("/api/available-stocks")
async def get_available_stocks():
    """
    Get a list of available stocks from the data directory.
    """
    try:
        stocks = ml_optimizer.get_available_stocks()
        return {"stocks": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize-portfolio")
async def optimize_portfolio(request: PortfolioRequest):
    """
    Optimize portfolio using ML-based approach.
    """
    try:
        # Validate inputs
        if request.investment_amount <= 0:
            raise HTTPException(status_code=400, detail="Investment amount must be positive")
        
        if request.investment_time <= 0:
            raise HTTPException(status_code=400, detail="Investment time must be positive")
        
        if request.risk_preference not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="Risk preference must be low, medium, or high")
        
        # Run ML-based portfolio optimization
        result = ml_optimizer.run_optimization(
            investment_amount=request.investment_amount,
            risk_preference=request.risk_preference,
            num_stocks=10  # Default to 10 stocks
        )
        
        # Check if there was an error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-stocks")
async def compare_stocks(request: StockComparisonRequest):
    """
    Compare stocks using ML-based predictions.
    """
    try:
        # Validate inputs
        if not request.stock_symbols:
            raise HTTPException(status_code=400, detail="At least one stock symbol is required")
        
        # Load and preprocess data
        data_loaded = ml_optimizer.load_and_preprocess_data()
        
        if not data_loaded:
            raise HTTPException(
                status_code=500, 
                detail="No stock data available. Please check your data directory."
            )
        
        # Prepare features
        features_prepared = ml_optimizer.prepare_features()
        
        if not features_prepared:
            raise HTTPException(
                status_code=500, 
                detail="Failed to prepare features for ML models."
            )
        
        # Train models
        models_trained = ml_optimizer.train_models()
        
        if not models_trained:
            raise HTTPException(
                status_code=500, 
                detail="Failed to train ML models."
            )
        
        # Get predictions for requested stocks
        predictions = ml_optimizer.predict_returns()
        
        # Filter predictions for requested stocks
        stock_comparisons = {}
        for symbol in request.stock_symbols:
            if symbol in predictions:
                stock_comparisons[symbol] = predictions[symbol]
            else:
                stock_comparisons[symbol] = {
                    "predicted_return": 0,
                    "volatility": 0,
                    "sharpe": 0,
                    "error": "Stock data not available"
                }
        
        return {
            "comparisons": stock_comparisons
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-portfolio")
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    Analyze an existing portfolio.
    """
    try:
        # Validate inputs
        if not request.portfolio:
            raise HTTPException(status_code=400, detail="Portfolio is required")
        
        if request.risk_preference not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="Risk preference must be low, medium, or high")
        
        # Load and preprocess data
        data_loaded = ml_optimizer.load_and_preprocess_data()
        
        if not data_loaded:
            raise HTTPException(
                status_code=500, 
                detail="No stock data available. Please check your data directory."
            )
        
        # Prepare features
        features_prepared = ml_optimizer.prepare_features()
        
        if not features_prepared:
            raise HTTPException(
                status_code=500, 
                detail="Failed to prepare features for ML models."
            )
        
        # Train models
        models_trained = ml_optimizer.train_models()
        
        if not models_trained:
            raise HTTPException(
                status_code=500, 
                detail="Failed to train ML models."
            )
        
        # Get predictions for all stocks
        predictions = ml_optimizer.predict_returns()
        
        # Analyze portfolio
        portfolio_analysis = []
        total_investment = sum(stock.get('investment_amount', 0) for stock in request.portfolio)
        
        for stock in request.portfolio:
            symbol = stock.get('symbol')
            investment_amount = stock.get('investment_amount', 0)
            weight = investment_amount / total_investment if total_investment > 0 else 0
            
            if symbol in predictions:
                pred = predictions[symbol]
                portfolio_analysis.append({
                    'symbol': symbol,
                    'investment_amount': investment_amount,
                    'weight': weight,
                    'predicted_return': pred['predicted_return'],
                    'volatility': pred['volatility'],
                    'sharpe': pred['sharpe'],
                    'recommendation': 'HOLD'  # Default recommendation
                })
            else:
                portfolio_analysis.append({
                    'symbol': symbol,
                    'investment_amount': investment_amount,
                    'weight': weight,
                    'predicted_return': 0,
                    'volatility': 0,
                    'sharpe': 0,
                    'recommendation': 'UNKNOWN',
                    'error': 'Stock data not available'
                })
        
        # Calculate portfolio metrics
        portfolio_return = sum(stock['weight'] * stock['predicted_return'] for stock in portfolio_analysis)
        portfolio_volatility = np.sqrt(sum(stock['weight']**2 * stock['volatility']**2 for stock in portfolio_analysis))
        portfolio_sharpe = (portfolio_return - ml_optimizer.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        
        # Generate recommendations
        for stock in portfolio_analysis:
            if stock['predicted_return'] > 0.02:  # 2% threshold
                stock['recommendation'] = 'BUY'
            elif stock['predicted_return'] < -0.02:  # -2% threshold
                stock['recommendation'] = 'SELL'
        
        return {
            'portfolio_analysis': portfolio_analysis,
            'total_investment': total_investment,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'risk_level': request.risk_preference
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)