import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Grid,
  CircularProgress,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import axios from 'axios';

function PortfolioAnalysis() {
  const [portfolio, setPortfolio] = useState([{ symbol: '', weight: '' }]);
  const [riskPreference, setRiskPreference] = useState('medium');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);

  const handlePortfolioChange = (index, field, value) => {
    const newPortfolio = [...portfolio];
    newPortfolio[index] = { ...newPortfolio[index], [field]: value };
    setPortfolio(newPortfolio);
  };

  const addStock = () => {
    setPortfolio([...portfolio, { symbol: '', weight: '' }]);
  };

  const removeStock = (index) => {
    const newPortfolio = portfolio.filter((_, i) => i !== index);
    setPortfolio(newPortfolio);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setDebugInfo(null);
    setResult(null);

    // Validate inputs
    const validPortfolio = portfolio.filter(stock => 
      stock.symbol.trim() !== '' && 
      !isNaN(parseFloat(stock.weight)) && 
      parseFloat(stock.weight) > 0
    );

    if (validPortfolio.length === 0) {
      setError("Please enter at least one valid stock with a weight");
      setLoading(false);
      return;
    }

    // Check if weights sum to 100%
    const totalWeight = validPortfolio.reduce((sum, stock) => sum + parseFloat(stock.weight), 0);
    if (Math.abs(totalWeight - 100) > 0.01) {
      setError(`Total weight must be 100%. Current total: ${totalWeight.toFixed(2)}%`);
      setLoading(false);
      return;
    }

    try {
      console.log("Sending request with portfolio:", validPortfolio);
      const response = await axios.post('http://localhost:8000/analyze-portfolio', {
        current_portfolio: validPortfolio.map(stock => ({
          symbol: stock.symbol,
          weight: parseFloat(stock.weight) / 100,
        })),
        risk_preference: riskPreference,
      });

      console.log("Response received:", response.data);
      
      if (!response.data.analysis || response.data.analysis.length === 0) {
        setError("No analysis could be generated for the provided portfolio. Please check your stock symbols.");
        setDebugInfo({
          requestedPortfolio: validPortfolio,
          responseData: response.data
        });
      } else {
        setResult(response.data);
      }
    } catch (err) {
      console.error("Error in portfolio analysis:", err);
      setError(err.response?.data?.detail || 'An error occurred while analyzing your portfolio');
      setDebugInfo({
        error: err.message,
        response: err.response?.data,
        requestedPortfolio: validPortfolio
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Portfolio Analysis
      </Typography>
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Risk Preference</InputLabel>
                  <Select
                    value={riskPreference}
                    onChange={(e) => setRiskPreference(e.target.value)}
                    label="Risk Preference"
                  >
                    <MenuItem value="low">Low Risk</MenuItem>
                    <MenuItem value="medium">Medium Risk</MenuItem>
                    <MenuItem value="high">High Risk</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              {portfolio.map((stock, index) => (
                <Grid item xs={12} key={index}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                      fullWidth
                      label="Stock Symbol"
                      value={stock.symbol}
                      onChange={(e) => handlePortfolioChange(index, 'symbol', e.target.value)}
                      required
                      helperText="Enter the stock symbol (e.g., RELIANCE, TCS)"
                    />
                    <TextField
                      fullWidth
                      label="Weight (%)"
                      type="number"
                      value={stock.weight}
                      onChange={(e) => handlePortfolioChange(index, 'weight', e.target.value)}
                      required
                      helperText="Enter weight as percentage"
                    />
                    <IconButton
                      color="error"
                      onClick={() => removeStock(index)}
                      disabled={portfolio.length === 1}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Grid>
              ))}
              <Grid item xs={12}>
                <Button
                  startIcon={<AddIcon />}
                  onClick={addStock}
                  sx={{ mr: 2 }}
                >
                  Add Stock
                </Button>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Analyze Portfolio'}
                </Button>
              </Grid>
            </Grid>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {debugInfo && (
        <Card sx={{ mb: 2, bgcolor: '#1e1e1e' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Debug Information</Typography>
            <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(debugInfo, null, 2)}
            </Typography>
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Portfolio Analysis Results
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              Total Stocks: {result.total_stocks}
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              Risk Level: {result.risk_level}
            </Typography>

            <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
              Stock Analysis:
            </Typography>
            <Grid container spacing={2}>
              {result.analysis.map((stock, index) => (
                <Grid item xs={12} md={6} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6">{stock.symbol}</Typography>
                      <Typography>
                        Current Price: ${stock.current_price.toLocaleString()}
                      </Typography>
                      <Typography>
                        Average Return: {(stock.avg_return * 100).toFixed(2)}%
                      </Typography>
                      <Typography>
                        Volatility: {(stock.volatility * 100).toFixed(2)}%
                      </Typography>
                      <Typography>
                        Recommendation: {stock.recommendation}
                      </Typography>
                      <Typography>
                        Suggested Weight: {(stock.suggested_weight * 100).toFixed(2)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default PortfolioAnalysis; 