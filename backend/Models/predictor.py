import torch
import torch.nn as nn
from Utils.AF import HyperVolatilityAct

class ReturnPredictor(nn.Module):
    def __init__(self, input_size):
        super(ReturnPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            HyperVolatilityAct(alpha=0.5),  # Custom Activation
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
