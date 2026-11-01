"""
Stage 3: Characteristic Frequency Input Network (CFIN) for Inertia Identification.

Maps 15 characteristic frequency parameters from attitude tumbling FFT
to the target spacecraft's principal inertia ratios:
    y = [Jxx/Jzz, Jyy/Jzz]^T in (0, 1]

References:
- Ma et al., 'Robust attitude estimation of rotating space debris based on
  virtual observations of neural network', 2021. Section 3.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


class SinhActivation(nn.Module):
    """
    Hyperbolic sine activation function with clamp to prevent numerical overflow:
        f(x) = sinh(clamp(x, -5, 5))
    Per Ma et al. Section 3.2.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, -5.0, 5.0)
        return torch.sinh(x_clamped)


class CFINNet(nn.Module):
    """
    Characteristic Frequency Input Network (CFIN).
    Architecture:
        Input: 15 normalized characteristic frequencies
        Hidden Layer: 100 nodes with sinh / tanh activation
        Output Layer: 2 nodes predicting inertia ratios [Jxx/Jzz, Jyy/Jzz]
    """
    def __init__(self, input_dim: int = 15, hidden_dim: int = 100, output_dim: int = 2, activation: str = "tanh"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if activation == "sinh":
            self.act = SinhActivation()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Bounded sigmoid to ensure 0 < Jxx/Jzz < Jyy/Jzz <= 1
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 15) normalized characteristic frequency vector
        Returns:
            y_pred: (B, 2) [Jxx/Jzz, Jyy/Jzz] where 0 < y1 <= y2 <= 1
        """
        h = self.fc1(x)
        h = self.act(h)
        raw_out = self.fc2(h)
        # Map to (0, 1)
        y_sig = self.out_act(raw_out)

        # Enforce physical ordering: Jxx <= Jyy
        # Let y1 = y_sig[:, 0], and y2 = y1 + (1 - y1) * y_sig[:, 1]
        y1 = y_sig[:, 0:1] * 0.95 + 0.05
        y2 = y1 + (0.98 - y1) * y_sig[:, 1:2]

        return torch.cat([y1, y2], dim=-1)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error loss (Eq. 11 in Ma et al.).
        """
        return F.mse_loss(y_pred, y_true)
