# TradeVista

TradeVista is a portfolio optimization and analysis tool that leverages modern portfolio theory (MPT) and machine learning techniques to help users make informed investment decisions.

## Features

### 1. Portfolio Optimization
- Implements **Modern Portfolio Theory (MPT)** to optimize asset allocation for maximum returns at a given risk level.
- Supports advanced optimization algorithms, including:
  - Mean-Variance Optimization
  - Risk Parity
  - Markowitz Portfolio Theory
  
### 2. Machine Learning Models
- Utilizes machine learning models to predict portfolio performance and market trends.
- Includes:
  - Regression models for return prediction.
  - Classification models for risk assessment.
  - Neural networks for advanced forecasting.

### 3. Interactive Frontend
- Provides a user-friendly interface for:
  - Visualizing portfolio performance metrics (e.g., Sharpe ratio, volatility).
  - Comparing different portfolio strategies.
  - Interactive charts and graphs for better insights.

### 4. Backend Utilities
- Comprehensive backend for:
  - Data preprocessing (e.g., cleaning, normalization).
  - Portfolio optimization and simulation.
  - Integration with APIs for real-time market data.

### 5. Reporting and Analysis
- Generates detailed reports on portfolio performance.
- Exports results in JSON and PDF formats for further analysis.

### 6. Testing and Validation
- Includes Jupyter notebooks for testing and validating models.
- Provides unit tests for ensuring the reliability of optimization algorithms.

## Project Structure

```
TradeVista/
├── backend/
│   ├── Models/
│   │   ├── ml_portfolio_optimizer.py
│   │   ├── Model.py
│   │   ├── MPT.py
│   │   ├── optimizer_engine.py
│   │   ├── predictor.py
│   │   └── Saved/
│   ├── Notebooks/
│   │   ├── compare_activation.py
│   │   └── test_activation.py
│   └── Strategies/
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── public/
│   └── src/
├── Data Set Link.txt
├── main.py
├── portfolio_optimization_results.json
├── README.md
├── Report.pdf
├── requirements.txt
├── script.py
└── test_portfolio_optimizer.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TradeVista.git
   cd TradeVista
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

- **Backend**: Handles data processing, optimization, and prediction tasks.
- **Frontend**: Allows users to interact with the tool and visualize results.
- **Notebooks**: Provides scripts for testing and validating models.

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any inquiries or feedback, please contact akbarikartik0811@gmail.com.
