"""Distances, (soft) nearest neighbours, fusion, image scores and pixel maps (spec 2, 4, 5, 6)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def unit(x: torch.Tensor) -> torch.Tensor:
    """Unit-normalise the last dimension in float32."""
    return F.normalize(x.float(), dim=-1)


def knn_distances(query: torch.Tensor, bank: torch.Tensor, k: int = 1,
                  chunk: int = 65536) -> torch.Tensor:
    """Cosine distances from each query to its k nearest bank vectors, ascending.

    query (P, D) and bank (M, D) must already be unit-normalised.
    Returns (P, k). Distance = 1 - cosine (spec Fact 1), clamped at 0 for rounding.
    """
    if bank.shape[0] == 0:
        raise ValueError("empty bank")
    k = min(k, bank.shape[0])
    best = None
    for s in range(0, bank.shape[0], chunk):
        sim = query @ bank[s:s + chunk].T                       # (P, m)
        top = sim.topk(min(k, sim.shape[1]), dim=1).values
        best = top if best is None else torch.cat([best, top], 1).topk(k, dim=1).values
    return (1.0 - best).clamp_min(0.0)


def softmin(d: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    """m_tau(d) = -tau * log sum_i exp(-d_i / tau), over the last dimension (spec 4.1)."""
    return -tau * torch.logsumexp(-d / tau, dim=-1)


def fused_score(d_normal: torch.Tensor, d_defect: torch.Tensor | None,
                lam: torch.Tensor | float, tau_d: torch.Tensor | float) -> torch.Tensor:
    """Bounded-bonus fusion s = d_N + lam * exp(-d_D / tau_D) (spec 5.2).

    `d_defect=None` (empty defect bank, k_d = 0) gives the baseline score d_N.
    """
    if d_defect is None:
        return d_normal
    return d_normal + lam * torch.exp(-d_defect / tau_d)


def image_score(patch_scores: np.ndarray | torch.Tensor, fraction: float = 0.01):
    """Mean of the top floor(fraction * P) scores; the max if that count is 0 (AnomalyDINO mean_top1p)."""
    if isinstance(patch_scores, torch.Tensor):
        flat = patch_scores.reshape(-1)
        n = int(flat.numel() * fraction)
        return flat.max() if n == 0 else flat.topk(n).values.mean()
    flat = np.asarray(patch_scores, dtype=np.float64).reshape(-1)
    n = int(flat.size * fraction)
    return float(flat.max()) if n == 0 else float(np.sort(flat)[::-1][:n].mean())


def anomaly_map(patch_scores: np.ndarray, grid: tuple[int, int], out_hw: tuple[int, int],
                sigma: float = 4.0) -> np.ndarray:
    """Bilinear upsampling to the original image size, then Gaussian blur (AnomalyDINO dists2map)."""
    grid_map = np.asarray(patch_scores, dtype=np.float64).reshape(grid)
    up = cv2.resize(grid_map, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(up, sigma=sigma).astype(np.float32)


def baseline_patch_scores(test_feats: torch.Tensor, test_fg: torch.Tensor,
                          bank: torch.Tensor) -> torch.Tensor:
    """AnomalyDINO scoring for a batch of images.

    test_feats (N, P, D), test_fg (N, P) bool, bank (M, D) unit-normalised, all on one device.
    Returns (N, P): 1-NN cosine distance, 0 for masked background patches.
    """
    out = torch.zeros(test_feats.shape[:2], device=test_feats.device)
    for i in range(test_feats.shape[0]):
        d = knn_distances(unit(test_feats[i]), bank, k=1)[:, 0]
        out[i] = torch.where(test_fg[i], d, torch.zeros_like(d))
    return out
