import torch
import torch.nn as nn
from Utils.AF import HyperVolatilityAct

class TradeVistaModel(nn.Module):
    def __init__(self, input_size=10, alpha=0.5):
        super(TradeVistaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.act1 = HyperVolatilityAct(alpha=alpha)
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.output(x)
