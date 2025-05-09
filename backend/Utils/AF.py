import torch
import torch.nn as nn

class HyperVolatilityAct(nn.Module):
    """
    Custom activation function:
    f(x) = x * sigmoid(x) * (1 + alpha * |x|^2)
    Reacts strongly to market volatility.
    """
    def __init__(self, alpha=0.5):
        super(HyperVolatilityAct, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.sigmoid(x) * (1 + self.alpha * torch.abs(x) ** 2)
