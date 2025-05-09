import React, { useState } from 'react';
import {
    Container,
    Typography,
    TextField,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Card,
    CardContent,
    Grid,
    Alert,
    CircularProgress
} from '@mui/material';
import { optimizePortfolio } from '../api';

const PortfolioOptimizer = () => {
    const [formData, setFormData] = useState({
        investment_amount: '',
        investment_time: '',
        risk_preference: 'medium'
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [debugInfo, setDebugInfo] = useState(null);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setDebugInfo(null);
        setResult(null);

        try {
            // Validate inputs
            if (!formData.investment_amount || formData.investment_amount <= 0) {
                throw new Error('Please enter a valid investment amount');
            }
            if (!formData.investment_time || formData.investment_time <= 0) {
                throw new Error('Please enter a valid investment time');
            }

            console.log('Sending request:', formData);
            const response = await optimizePortfolio(formData);
            console.log('Received response:', response);
            
            setResult(response);
        } catch (err) {
            console.error('Error:', err);
            setError(err.message);
            setDebugInfo({
                formData,
                error: err.toString()
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h4" component="h1" gutterBottom>
                Portfolio Optimizer
            </Typography>
            
            <Card sx={{ mb: 4 }}>
                <CardContent>
                    <form onSubmit={handleSubmit}>
                        <Grid container spacing={3}>
                            <Grid item xs={12} md={4}>
                                <TextField
                                    fullWidth
                                    label="Investment Amount"
                                    name="investment_amount"
                                    type="number"
                                    value={formData.investment_amount}
                                    onChange={handleChange}
                                    helperText="Enter amount in INR"
                                    required
                                />
                            </Grid>
                            <Grid item xs={12} md={4}>
                                <TextField
                                    fullWidth
                                    label="Investment Time (months)"
                                    name="investment_time"
                                    type="number"
                                    value={formData.investment_time}
                                    onChange={handleChange}
                                    helperText="Enter time in months"
                                    required
                                />
                            </Grid>
                            <Grid item xs={12} md={4}>
                                <FormControl fullWidth>
                                    <InputLabel>Risk Preference</InputLabel>
                                    <Select
                                        name="risk_preference"
                                        value={formData.risk_preference}
                                        onChange={handleChange}
                                        label="Risk Preference"
                                    >
                                        <MenuItem value="low">Low Risk</MenuItem>
                                        <MenuItem value="medium">Medium Risk</MenuItem>
                                        <MenuItem value="high">High Risk</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12}>
                                <Button
                                    type="submit"
                                    variant="contained"
                                    color="primary"
                                    disabled={loading}
                                    fullWidth
                                >
                                    {loading ? <CircularProgress size={24} /> : 'Optimize Portfolio'}
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
                            Optimized Portfolio
                        </Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <Typography variant="subtitle1">
                                    Total Investment: ₹{result.total_investment.toLocaleString()}
                                </Typography>
                                <Typography variant="subtitle1">
                                    Risk Level: {result.risk_level}
                                </Typography>
                                <Typography variant="subtitle1">
                                    Expected Return: {(result.expected_return * 100).toFixed(2)}%
                                </Typography>
                                <Typography variant="subtitle1">
                                    Portfolio Volatility: {(result.volatility * 100).toFixed(2)}%
                                </Typography>
                                <Typography variant="subtitle1">
                                    Sharpe Ratio: {result.sharpe_ratio.toFixed(2)}
                                </Typography>
                            </Grid>
                            <Grid item xs={12}>
                                <Typography variant="h6" gutterBottom>
                                    Portfolio Allocation
                                </Typography>
                                {result.portfolio.map((stock, index) => (
                                    <Card key={index} sx={{ mb: 1 }}>
                                        <CardContent>
                                            <Typography variant="subtitle1">
                                                {stock.symbol}
                                            </Typography>
                                            <Typography variant="body2">
                                                Investment: ₹{stock.investment_amount.toLocaleString()}
                                            </Typography>
                                            <Typography variant="body2">
                                                Weight: {(stock.weight * 100).toFixed(2)}%
                                            </Typography>
                                            <Typography variant="body2">
                                                Expected Return: {(stock.expected_return * 100).toFixed(2)}%
                                            </Typography>
                                            <Typography variant="body2">
                                                Volatility: {(stock.volatility * 100).toFixed(2)}%
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                ))}
                            </Grid>
                        </Grid>
                    </CardContent>
                </Card>
            )}
        </Container>
    );
};

export default PortfolioOptimizer; 