import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Container, Box } from '@mui/material';
import Navbar from './components/Navbar';
import PortfolioOptimizer from './pages/PortfolioOptimizer';
import StockComparison from './pages/StockComparison';
import PortfolioAnalysis from './pages/PortfolioAnalysis';
import { checkHealth } from './api';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  // Check backend health on startup
  React.useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await checkHealth();
        console.log('Backend is healthy');
      } catch (error) {
        console.error('Backend health check failed:', error);
      }
    };
    
    checkBackendHealth();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Navbar />
          <Container component="main" sx={{ mt: 4, mb: 4, flex: 1 }}>
            <Routes>
              <Route path="/" element={<PortfolioOptimizer />} />
              <Route path="/compare" element={<StockComparison />} />
              <Route path="/analyze" element={<PortfolioAnalysis />} />
            </Routes>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App; 