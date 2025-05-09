const API_BASE_URL = 'http://localhost:8000/api';

export const getAvailableStocks = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/available-stocks`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get available stocks');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Get available stocks error:', error);
        throw error;
    }
};

export const optimizePortfolio = async (data) => {
    try {
        const response = await fetch(`${API_BASE_URL}/optimize-portfolio`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to optimize portfolio');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Portfolio optimization error:', error);
        throw error;
    }
};

export const compareStocks = async (data) => {
    try {
        const response = await fetch(`${API_BASE_URL}/compare-stocks`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to compare stocks');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Stock comparison error:', error);
        throw error;
    }
};

export const analyzePortfolio = async (data) => {
    try {
        const response = await fetch(`${API_BASE_URL}/analyze-portfolio`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze portfolio');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Portfolio analysis error:', error);
        throw error;
    }
};

export const checkHealth = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('Backend service is not healthy');
        }
        return await response.json();
    } catch (error) {
        console.error('Health check error:', error);
        throw error;
    }
}; 