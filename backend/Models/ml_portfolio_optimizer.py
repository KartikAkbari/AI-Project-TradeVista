import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Models.predictor import ReturnPredictor
from Models.Model import TradeVistaModel

class MLPortfolioOptimizer:
    
    def __init__(self, data_dir="Data/Processed", lookback_period=20, risk_free_rate=0.01):
    
        self.data_dir = data_dir
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.return_predictor = None
        self.trade_vista_model = None
        self.scaler = StandardScaler()
        self.stock_data = {}
        self.features = {}
        self.available_stocks = []
        self.models_dir = "Models/Saved"
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess stock data from CSV files.
        """
        print("Loading and preprocessing stock data...")
        self.stock_data = {}
        self.available_stocks = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Data directory '{self.data_dir}' does not exist")
            return False
            
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in '{self.data_dir}'")
            return False
            
        for file in csv_files:
            try:
                symbol = file.split('.')[0]
                df = pd.read_csv(os.path.join(self.data_dir, file))
                
                # Check if dataframe is empty or has missing values
                if df.empty or df['close'].isna().any():
                    print(f"Skipping {file}: Empty dataframe or missing values")
                    continue
                
                # Ensure we have enough data points
                if len(df) < self.lookback_period:
                    print(f"Skipping {file}: Insufficient data points")
                    continue
                
                # Calculate returns
                df['returns'] = df['close'].pct_change()
                df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
                df['returns'] = df['returns'].fillna(0)
                
                # Calculate volatility
                df['volatility'] = df['returns'].rolling(window=20).std()
                df['volatility'] = df['volatility'].fillna(0)
                
                # Calculate moving averages
                df['ma5'] = df['close'].rolling(window=5).mean()
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['ma50'] = df['close'].rolling(window=50).mean()
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
                df['rsi'] = df['rsi'].fillna(50)
                
                # Drop NaN values
                df = df.dropna()
                
                # Store the data
                self.stock_data[symbol] = df
                self.available_stocks.append(symbol)
                
                print(f"Processed {symbol}: {len(df)} data points")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        print(f"Loaded {len(self.stock_data)} stocks")
        return len(self.stock_data) > 0
        
    def prepare_features(self):
        """
        Prepare features for the ML models using all available indicators.
        """
        print("Preparing features for ML models...")
        
        if not self.stock_data:
            print("No stock data available. Please load data first.")
            return False
            
        try:
            for symbol, df in self.stock_data.items():
                # Create feature matrix
                features = []
                targets = []
                
                # Ensure we have enough data
                if len(df) <= self.lookback_period:
                    print(f"Not enough data for {symbol}, skipping...")
                    continue
                
                # Get all available technical indicators
                indicator_columns = [
                    'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI_14', 
                    'MACD', 'Signal_Line', 'SMA_20', 'BB_Upper', 'BB_Lower',
                    'TR', 'ATR_14', 'L14', 'H14', '%K', '%D', 'TP', 
                    'MF', 'Positive_MF', 'Negative_MF', 'MFR', 'MFI_14', 'OBV'
                ]
                
                # Check which indicators are available in the dataframe
                available_indicators = [col for col in indicator_columns if col in df.columns]
                
                if not available_indicators:
                    print(f"No technical indicators found for {symbol}, skipping...")
                    continue
                
                print(f"Using indicators for {symbol}: {', '.join(available_indicators)}")
                
                # Fill NaN values in the dataframe
                for col in df.columns:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                for i in range(self.lookback_period, len(df)):
                    try:
                        # Features: price data and all available indicators
                        feature = []
                        
                        # Add price data
                        feature.extend([
                            df['close'].iloc[i-self.lookback_period:i].values,
                            df['open'].iloc[i-self.lookback_period:i].values,
                            df['high'].iloc[i-self.lookback_period:i].values,
                            df['low'].iloc[i-self.lookback_period:i].values,
                            df['volume'].iloc[i-self.lookback_period:i].values,
                            df['change (%)'].iloc[i-self.lookback_period:i].values
                        ])
                        
                        # Add all available technical indicators
                        for indicator in available_indicators:
                            if indicator in df.columns:
                                indicator_values = df[indicator].iloc[i-self.lookback_period:i].values
                                # Fill NaN values with 0
                                indicator_values = np.nan_to_num(indicator_values, 0)
                                feature.append(indicator_values)
                        
                        # Flatten features
                        feature = np.concatenate(feature)
                        
                        # Target: next 3 months return (approximately 63 trading days)
                        if i + 63 < len(df):
                            future_price = df['close'].iloc[i + 63]
                            current_price = df['close'].iloc[i]
                            target = (future_price - current_price) / current_price
                        else:
                            # If we don't have 3 months of future data, use next day return
                            target = df['change (%)'].iloc[i] / 100
                        
                        # Skip if target is NaN
                        if np.isnan(target):
                            continue
                            
                        features.append(feature)
                        targets.append(target)
                            
                    except Exception as e:
                        print(f"Error processing features for {symbol} at index {i}: {str(e)}")
                        continue
                
                if not features:
                    print(f"No valid features generated for {symbol}, skipping...")
                    continue
                
                # Convert to numpy arrays
                features = np.array(features)
                targets = np.array(targets)
                
                # Scale features
                try:
                    features_scaled = self.scaler.fit_transform(features)
                    
                    # Store features and targets
                    self.features[symbol] = {
                        'features': features_scaled,
                        'targets': targets,
                        'feature_shape': features_scaled.shape,
                        'available_indicators': available_indicators
                    }
                    
                    print(f"Prepared features for {symbol}: {len(features)} samples with shape {features_scaled.shape}")
                except Exception as e:
                    print(f"Error scaling features for {symbol}: {str(e)}")
                    continue
            
            return len(self.features) > 0
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def save_models(self):
        """
        Save trained models to disk.
        """
        if not self.return_predictor or not self.trade_vista_model:
            print("No models to save. Please train models first.")
            return False
            
        try:
            # Save return predictor
            return_path = os.path.join(self.models_dir, "return_predictor.pth")
            torch.save(self.return_predictor.state_dict(), return_path)
            
            # Save trade vista model
            vista_path = os.path.join(self.models_dir, "trade_vista_model.pth")
            torch.save(self.trade_vista_model.state_dict(), vista_path)
            
            # Save scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print(f"Models saved to {self.models_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False
            
    def load_models(self):
        """
        Load trained models from disk.
        """
        try:
            # Check if models exist
            return_path = os.path.join(self.models_dir, "return_predictor.pth")
            vista_path = os.path.join(self.models_dir, "trade_vista_model.pth")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            
            if not (os.path.exists(return_path) and os.path.exists(vista_path) and os.path.exists(scaler_path)):
                print("No saved models found.")
                return False
                
            # Load scaler
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            # Determine the correct input size from the first stock's features
            if not self.features:
                print("No features available. Please prepare features first.")
                return False
                
            first_symbol = next(iter(self.features))
            first_features = self.features[first_symbol]['features']
            input_size = first_features.shape[1]  # Number of features per sample
            
            # Initialize models with the correct input size
            self.return_predictor = ReturnPredictor(input_size)
            self.trade_vista_model = TradeVistaModel(input_size)
            
            # Load model weights
            self.return_predictor.load_state_dict(torch.load(return_path))
            self.trade_vista_model.load_state_dict(torch.load(vista_path))
            
            print(f"Models loaded from {self.models_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
            
    def train_models(self):
        """
        Train ML models for each stock using the prepared features.
        """
        print("Training ML models...")
        
        if not self.features:
            print("No features available. Please prepare features first.")
            return False
            
        try:
            self.models = {}
            
            for symbol, data in self.features.items():
                print(f"\nTraining model for {symbol}...")
                
                X = torch.FloatTensor(data['features'])
                y = torch.FloatTensor(data['targets']).reshape(-1, 1)
                
                # Split data into train and validation sets
                train_size = int(0.8 * len(X))
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
                
                # Define model architecture
                class ReturnPredictor(torch.nn.Module):
                    def __init__(self, input_size):
                        super().__init__()
                        self.layers = torch.nn.Sequential(
                            torch.nn.Linear(input_size, 256),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.3),
                            torch.nn.Linear(256, 128),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.2),
                            torch.nn.Linear(128, 64),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.1),
                            torch.nn.Linear(64, 32),
                            torch.nn.ReLU(),
                            torch.nn.Linear(32, 1),
                            torch.nn.Tanh()  # Bounded output
                        )
                        
                    def forward(self, x):
                        return self.layers(x)
                
                # Initialize model
                model = ReturnPredictor(X.shape[1])
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                
                # Training loop
                best_val_loss = float('inf')
                patience = 10
                patience_counter = 0
                
                for epoch in range(100):
                    # Training
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_loss = criterion(val_outputs, y_val)
                        val_mae = torch.mean(torch.abs(val_outputs - y_val))
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Val MAE = {val_mae.item():.4f}")
                
                # Store model and metrics
                self.models[symbol] = {
                    'model': model,
                    'val_mae': val_mae.item(),
                    'feature_shape': data['feature_shape'],
                    'available_indicators': data['available_indicators']
                }
                
                # Save model
                model_path = os.path.join(self.models_dir, f"{symbol}_model.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"Error in train_models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_returns(self):
        """
        Predict returns for each stock using the trained models.
        """
        print("Predicting returns...")
        
        if not self.models:
            print("No trained models available. Please train models first.")
            return None
            
        if not self.features:
            print("No features available. Please prepare features first.")
            return None
            
        predictions = {}
        
        try:
            for symbol, model_data in self.models.items():
                if symbol not in self.features:
                    print(f"No features available for {symbol}, skipping...")
                    continue
                    
                print(f"\nPredicting returns for {symbol}...")
                
                # Get the latest features
                features = self.features[symbol]['features'][-1:]  # Get only the latest data point
                
                # Make prediction
                model = model_data['model']
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features)
                    predicted_return = model(features_tensor).item()
                
                # Calculate volatility from historical data
                historical_returns = self.features[symbol]['targets']
                volatility = np.std(historical_returns) if len(historical_returns) > 0 else 0.1
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
                risk_free_rate = 0.02
                sharpe_ratio = (predicted_return - risk_free_rate) / volatility if volatility > 0 else 0
                
                # Store predictions
                predictions[symbol] = {
                    'predicted_return': predicted_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'confidence': model_data['val_mae']  # Using validation MAE as confidence metric
                }
                
                print(f"Predicted return: {predicted_return:.2%}")
                print(f"Volatility: {volatility:.2%}")
                print(f"Sharpe ratio: {sharpe_ratio:.2f}")
                print(f"Model confidence (MAE): {model_data['val_mae']:.4f}")
            
            return predictions
            
        except Exception as e:
            print(f"Error in predict_returns: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_portfolio(self, investment_amount, risk_preference="medium", num_stocks=10):
        """
        Optimize portfolio weights based on ML predictions.
        
        Args:
            investment_amount: Total investment amount
            risk_preference: Risk preference (low, medium, high)
            num_stocks: Number of stocks to include in the portfolio
            
        Returns:
            Dictionary containing portfolio allocation and metrics
        """
        print("Optimizing portfolio...")
        
        try:
            # Predict returns
            predictions = self.predict_returns()
            
            if not predictions:
                print("No predictions available. Please run predict_returns first.")
                return {
                    "portfolio": [],
                    "total_investment": investment_amount,
                    "risk_level": risk_preference,
                    "expected_return": 0,
                    "volatility": 0,
                    "sharpe_ratio": 0,
                    "error": "No stock data available for optimization"
                }
            
            # Convert to list of tuples for sorting
            stock_metrics = [(symbol, metrics) for symbol, metrics in predictions.items()]
            
            # Sort stocks based on risk preference
            if risk_preference == "low":
                # Sort by volatility (ascending)
                stock_metrics.sort(key=lambda x: x[1]['volatility'])
            elif risk_preference == "high":
                # Sort by predicted return (descending)
                stock_metrics.sort(key=lambda x: x[1]['predicted_return'], reverse=True)
            else:  # medium risk
                # Sort by Sharpe ratio (descending)
                stock_metrics.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)
            
            # Select top stocks
            selected_stocks = stock_metrics[:num_stocks]
            
            # Calculate portfolio weights
            total_investment = investment_amount
            weights = [1/len(selected_stocks)] * len(selected_stocks)  # Equal weight distribution
            
            # Create portfolio
            portfolio = []
            for (symbol, metrics), weight in zip(selected_stocks, weights):
                investment = total_investment * weight
                portfolio.append({
                    'symbol': symbol,
                    'investment_amount': investment,
                    'weight': weight,
                    'expected_return': metrics['predicted_return'],
                    'volatility': metrics['volatility']
                })
            
            # Calculate portfolio metrics
            portfolio_return = sum(stock['weight'] * stock['expected_return'] for stock in portfolio)
            portfolio_volatility = np.sqrt(sum(stock['weight']**2 * stock['volatility']**2 for stock in portfolio))
            portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
            
            return {
                "portfolio": portfolio,
                "total_investment": total_investment,
                "risk_level": risk_preference,
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": portfolio_sharpe
            }
            
        except Exception as e:
            print(f"Error in optimize_portfolio: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "portfolio": [],
                "total_investment": investment_amount,
                "risk_level": risk_preference,
                "expected_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "error": str(e)
            }
    
    def run_optimization(self, investment_amount, risk_preference="medium", num_stocks=10):
        """
        Run the complete ML portfolio optimization process.
        
        Args:
            investment_amount: Total investment amount
            risk_preference: Risk preference (low, medium, high)
            num_stocks: Number of stocks to include in the portfolio
            
        Returns:
            Dictionary containing portfolio allocation and metrics
        """
        # Load and preprocess data
        data_loaded = self.load_and_preprocess_data()
        
        if not data_loaded:
            return {
                "portfolio": [],
                "total_investment": investment_amount,
                "risk_level": risk_preference,
                "expected_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "error": "No stock data available. Please check your data directory."
            }
        
        # Prepare features
        features_prepared = self.prepare_features()
        
        if not features_prepared:
            return {
                "portfolio": [],
                "total_investment": investment_amount,
                "risk_level": risk_preference,
                "expected_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "error": "Failed to prepare features for ML models."
            }
        
        # Train models
        models_trained = self.train_models()
        
        if not models_trained:
            return {
                "portfolio": [],
                "total_investment": investment_amount,
                "risk_level": risk_preference,
                "expected_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "error": "Failed to train ML models."
            }
        
        # Optimize portfolio
        result = self.optimize_portfolio(investment_amount, risk_preference, num_stocks)
        
        return result
        
    def get_available_stocks(self):
        """
        Get a list of available stocks.
        
        Returns:
            List of stock symbols
        """
        if not self.available_stocks:
            self.load_and_preprocess_data()
        
        return self.available_stocks 