"""Residual feature adapters (spec Section 3). Every adapter is the identity map at init."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, z):
        return z

    def reg(self):
        return torch.zeros(())


class LinearAdapter(nn.Module):
    """g(z) = z + W z, W = 0 at init (spec 3.1)."""

    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, d))

    def forward(self, z):
        return z + z @ self.W.T

    def reg(self):
        return (self.W ** 2).sum()


class LowRankAdapter(nn.Module):
    """g(z) = z + U V^T z; U small random, V = 0 (contingency, spec 3.1)."""

    def __init__(self, d: int, r: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d, r) * 0.01)
        self.V = nn.Parameter(torch.zeros(d, r))

    def forward(self, z):
        return z + (z @ self.V) @ self.U.T

    def reg(self):
        # ||U V^T||_F^2 = sum((U^T U) * (V^T V)), cheap for small r
        return ((self.U.T @ self.U) * (self.V.T @ self.V)).sum()


class MLPAdapter(nn.Module):
    """g(z) = z + W2 gelu(W1 z + b1) + b2; W1 Kaiming, W2 = b1 = b2 = 0 (spec 3.2, Prop 3.3)."""

    def __init__(self, d: int, h: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        return z + self.fc2(F.gelu(self.fc1(z)))

    def reg(self):
        return (self.fc2.weight ** 2).sum()


def make_adapter(name: str, d: int, cfg: dict) -> nn.Module:
    if name in ("identity", "baseline", "fixed_bank"):
        return Identity()
    if name == "linear":
        return LinearAdapter(d)
    if name == "lowrank":
        return LowRankAdapter(d, cfg.get("lowrank_r", 32))
    if name == "mlp":
        return MLPAdapter(d, cfg.get("mlp_hidden", 96))
    raise ValueError(f"unknown adapter {name}")
