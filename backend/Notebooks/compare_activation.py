import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import ReLU, Tanh

# Setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Utils.AF import HyperVolatilityAct

# Load sample data
df = pd.read_csv(os.path.join(project_root, 'Data', 'Processed', 'AAKASH.csv'))
df = df.dropna()
df['Return'] = df['close'].pct_change()
df = df.dropna()
returns = torch.tensor(df['Return'].values, dtype=torch.float32)

# Activations
custom_act = HyperVolatilityAct(alpha=0.5)
relu_act = ReLU()
tanh_act = Tanh()

# Apply
custom_output = custom_act(returns)
relu_output = relu_act(returns)
tanh_output = tanh_act(returns)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(returns.numpy(), label='Original Returns', alpha=0.5)
plt.plot(custom_output.detach().numpy(), label='Custom Activation', linewidth=2)
plt.plot(relu_output.detach().numpy(), label='ReLU', linestyle='--')
plt.plot(tanh_output.detach().numpy(), label='Tanh', linestyle=':')
plt.legend()
plt.title("Comparison of Activation Functions on Real Returns")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
