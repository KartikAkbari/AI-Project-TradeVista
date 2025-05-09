import pandas as pd
import torch
import matplotlib.pyplot as plt
import os 
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Utils.AF import HyperVolatilityAct


# === Load one CSV to test ===
df = pd.read_csv('Data/Processed/AAKASH.csv')  # Change file name to any from your set
df = df.dropna()

# === Feature Engineering ===
df['Return'] = df['close'].pct_change()
df = df.dropna()

# === Convert to Tensor ===
returns_tensor = torch.tensor(df['Return'].values, dtype=torch.float32)

# === Apply Custom Activation ===
activation = HyperVolatilityAct(alpha=0.5)
activated_output = activation(returns_tensor)

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'].values, returns_tensor.numpy(), label='Daily Returns', alpha=0.5)
plt.plot(df['datetime'].values, activated_output.detach().numpy(), label='Activated Output', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Custom Activation Function on Market Returns')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
