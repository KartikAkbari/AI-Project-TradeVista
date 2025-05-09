import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
} from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <ShowChartIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          AI Hedge Fund Optimizer
        </Typography>
        <Box>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
            startIcon={<ShowChartIcon />}
          >
            Portfolio Optimizer
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/compare"
            startIcon={<CompareArrowsIcon />}
          >
            Stock Comparison
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 