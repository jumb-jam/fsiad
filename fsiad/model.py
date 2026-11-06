"""Scoring model: adapter + (soft-)min kNN + bounded-bonus fusion (spec Sections 3-5, 8).

One class covers every method:
  baseline       Identity adapter, hard 1-NN, no defect bank       (= reproduced AnomalyDINO)
  fixed_bank     Identity adapter, hard 1-NN, fixed lambda / tau_D (spec 5.3)
  linear / mlp   learned adapter, top-k soft-min with learned tau, learned lambda / tau_D
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .scoring import softmin, unit


class FSModel(nn.Module):
    def __init__(self, adapter: nn.Module, k: int = 10, hard: bool = False,
                 tau: float = 0.05, lam: float = 0.0, tau_d: float = 0.05):
        super().__init__()
        self.adapter = adapter
        self.k = 1 if hard else k
        self.hard = hard
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.lam = nn.Parameter(torch.tensor(float(lam)))
        self.log_tau_d = nn.Parameter(torch.tensor(math.log(tau_d)))

    # ------------------------------------------------------------------ parameters
    @property
    def tau(self):
        return self.log_tau.exp().clamp_min(1e-3)

    @property
    def tau_d(self):
        return self.log_tau_d.exp().clamp_min(1e-3)

    def project(self):
        """Keep lambda >= 0 after each optimiser step (spec 5.2)."""
        with torch.no_grad():
            self.lam.clamp_(min=0.0)

    def reg(self):
        return self.adapter.reg()

    # ------------------------------------------------------------------ scoring
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt then unit-normalise; x (..., D) in any float dtype."""
        return unit(self.adapter(x.float()))

    def bank_distances(self, q: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """Distances (P, k) from unit queries q (P, D) to their k nearest unit bank rows.

        Neighbour selection runs without gradient (it has none anyway); distances to the
        selected neighbours are then recomputed with gradient. Same values as a full
        matrix, a fraction of the memory.
        """
        k = min(self.k, bank.shape[0])
        with torch.no_grad():
            idx = (q.detach() @ bank.detach().T).topk(k, dim=1).indices      # (P, k)
        d = 1.0 - (q.unsqueeze(1) * bank[idx]).sum(-1)
        return d.clamp_min(0.0)

    def reduce(self, d: torch.Tensor, tau) -> torch.Tensor:
        return d[:, 0] if self.hard else softmin(d, tau)

    def distance_maps(self, q_raw, normal_raw, defect_raw=None):
        """Per-image d_N and d_D, each (N, P). d_D is None when there is no defect bank."""
        bank = self.embed(normal_raw)
        dbank = self.embed(defect_raw) if defect_raw is not None and len(defect_raw) else None
        dn, dd = [], []
        for i in range(q_raw.shape[0]):
            q = self.embed(q_raw[i])
            dn.append(self.reduce(self.bank_distances(q, bank), self.tau))
            if dbank is not None:
                dd.append(self.reduce(self.bank_distances(q, dbank), self.tau))
        return torch.stack(dn), (torch.stack(dd) if dd else None)

    def combine(self, dn, dd, fg, lam=None, tau_d=None):
        """s = d_N + lam * exp(-d_D / tau_D); background patches score 0 (AnomalyDINO)."""
        lam = self.lam if lam is None else lam
        tau_d = self.tau_d if tau_d is None else tau_d
        s = dn if dd is None else dn + lam * torch.exp(-dd / tau_d)
        return torch.where(fg, s, torch.zeros_like(s))

    def patch_scores(self, q_raw, fg, normal_raw, defect_raw=None):
        dn, dd = self.distance_maps(q_raw, normal_raw, defect_raw)
        return self.combine(dn, dd, fg)


def make_model(method: str, d: int, cfg: dict, tau: float = 0.05,
               lam: float = 0.0, tau_d: float = 0.05) -> FSModel:
    from .adapters import make_adapter
    hard = method in ("baseline", "fixed_bank", "random_control")
    adapter = make_adapter("identity" if hard else method, d, cfg)
    return FSModel(adapter, k=cfg.get("softmin_k", 10), hard=hard, tau=tau, lam=lam, tau_d=tau_d)
