import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  Alert,
  CircularProgress,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import { compareStocks, getAvailableStocks } from '../api';

const StockComparison = () => {
  const [stockSymbols, setStockSymbols] = useState(['']);
  const [availableStocks, setAvailableStocks] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);

  // Fetch available stocks on component mount
  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const response = await getAvailableStocks();
        setAvailableStocks(response.stocks || []);
      } catch (err) {
        console.error('Error fetching available stocks:', err);
        setError('Failed to load available stocks. Please try again later.');
      }
    };
    
    fetchStocks();
  }, []);

  const handleAddStock = () => {
    setStockSymbols([...stockSymbols, '']);
  };

  const handleRemoveStock = (index) => {
    const newSymbols = stockSymbols.filter((_, i) => i !== index);
    setStockSymbols(newSymbols.length ? newSymbols : ['']);
  };

  const handleSymbolChange = (index, value) => {
    const newSymbols = [...stockSymbols];
    newSymbols[index] = value;
    setStockSymbols(newSymbols);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setDebugInfo(null);
    setResult(null);

    try {
      // Filter out empty symbols
      const validSymbols = stockSymbols.filter(symbol => symbol.trim());
      
      if (validSymbols.length === 0) {
        throw new Error('Please enter at least one stock symbol');
      }

      console.log('Sending request:', { stock_symbols: validSymbols });
      const response = await compareStocks({ stock_symbols: validSymbols });
      console.log('Received response:', response);
      
      setResult(response);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
      setDebugInfo({
        stockSymbols,
        error: err.toString()
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Stock Comparison
      </Typography>
      
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={2}>
              {stockSymbols.map((symbol, index) => (
                <Grid item xs={12} key={index}>
                  <Grid container spacing={1} alignItems="center">
                    <Grid item xs>
                      {availableStocks.length > 0 ? (
                        <FormControl fullWidth>
                          <InputLabel>Stock Symbol {index + 1}</InputLabel>
                          <Select
                            value={symbol}
                            onChange={(e) => handleSymbolChange(index, e.target.value)}
                            label={`Stock Symbol ${index + 1}`}
                          >
                            <MenuItem value="">
                              <em>Select a stock</em>
                            </MenuItem>
                            {availableStocks.map((stock) => (
                              <MenuItem key={stock} value={stock}>
                                {stock}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      ) : (
                        <TextField
                          fullWidth
                          label={`Stock Symbol ${index + 1}`}
                          value={symbol}
                          onChange={(e) => handleSymbolChange(index, e.target.value)}
                          helperText="Enter stock symbol (e.g., RELIANCE, TCS)"
                        />
                      )}
                    </Grid>
                    <Grid item>
                      <IconButton
                        onClick={() => handleRemoveStock(index)}
                        disabled={stockSymbols.length === 1}
                        color="error"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Grid>
                  </Grid>
                </Grid>
              ))}
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    startIcon={<AddIcon />}
                    onClick={handleAddStock}
                  >
                    Add Stock
                  </Button>
                  <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    disabled={loading}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Compare Stocks'}
                  </Button>
                </Box>
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
        <Card sx={{ mb: 2, bgcolor: '#f5f5f5' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Debug Information
            </Typography>
            <pre style={{ whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(debugInfo, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Stock Comparison Results
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(result.comparisons).map(([symbol, data]) => (
                <Grid item xs={12} md={6} key={symbol}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {symbol}
                      </Typography>
                      {data.error ? (
                        <Alert severity="error">
                          {data.error}
                        </Alert>
                      ) : (
                        <>
                          <Typography variant="body1">
                            Predicted Return: {(data.predicted_return * 100).toFixed(2)}%
                          </Typography>
                          <Typography variant="body1">
                            Volatility: {(data.volatility * 100).toFixed(2)}%
                          </Typography>
                          <Typography variant="body1">
                            Sharpe Ratio: {data.sharpe.toFixed(2)}
                          </Typography>
                        </>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default StockComparison; 