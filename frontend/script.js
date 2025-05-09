async function recommendPortfolio() {
    const amount = parseFloat(document.getElementById("amount").value);
    const duration = parseInt(document.getElementById("duration").value);
    const risk = document.getElementById("risk").value;
  
    const response = await fetch("/recommend-portfolio", {
      method: "POST",
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ amount, duration_months: duration, risk })
    });
  
    const data = await response.json();
    document.getElementById("recommendationResult").textContent = JSON.stringify(data, null, 2);
  }
  
  async function compareStocks() {
    const stocks = document.getElementById("compareStocks").value.split(",");
    const response = await fetch(`/compare-stocks?${stocks.map(s => `stock_list=${s.trim()}`).join("&")}`);
    const data = await response.json();
    document.getElementById("comparisonResult").textContent = JSON.stringify(data, null, 2);
  }
  
  async function advisePortfolio() {
    const stocks = document.getElementById("adviseStocks").value.split(",");
    const weights = document.getElementById("adviseWeights").value.split(",").map(w => parseFloat(w));
  
    const response = await fetch("/advise-portfolio", {
      method: "POST",
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ stocks, weights })
    });
  
    const data = await response.json();
    document.getElementById("adviceResult").textContent = JSON.stringify(data, null, 2);
  }
  
  async function optimizePortfolio() {
    const stocks = document.getElementById("optimizeStocks").value.split(",");
    const response = await fetch(`/get-optimized?${stocks.map(s => `stocks=${s.trim()}`).join("&")}`);
    const data = await response.json();
    document.getElementById("optimizeResult").textContent = JSON.stringify(data, null, 2);
  }
  