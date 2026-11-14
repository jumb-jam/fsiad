"""Position-aware kNN and a support-alignment statistic (diagnostics for the transistor/cable gap).

Global kNN (AnomalyDINO) compares a test patch with every support patch, wherever it is.
A missing part exposes board that looks like board elsewhere in the support, so the
distance stays small. Positional kNN only compares with support patches near the SAME grid
location, so "board where a transistor leg should be" no longer finds a close match.
"""
from __future__ import annotations

import torch

from .scoring import knn_distances, unit


def _windows(h: int, w: int, radius: int):
    """For each offset (dy, dx): test-grid slices and the matching support-grid slices."""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dy) >= h or abs(dx) >= w:
                continue
            ty = slice(max(0, -dy), h - max(0, dy))       # test rows whose y + dy is on the grid
            tx = slice(max(0, -dx), w - max(0, dx))
            sy = slice(max(0, dy), h + min(0, dy))        # the support rows they compare with
            sx = slice(max(0, dx), w + min(0, dx))
            yield ty, tx, sy, sx


@torch.no_grad()
def positional_distances(test_feats: torch.Tensor, support: torch.Tensor, grid: tuple[int, int],
                         radius: int, chunk: int = 16) -> torch.Tensor:
    """Cosine distance from each test patch to its nearest support patch within +/- radius cells.

    test_feats (N, P, D) and support (k, P, D), raw features (any float dtype), same device.
    Returns (N, P). With radius >= the grid size this equals global 1-NN on the same support.
    """
    h, w = grid
    n, p, d = test_feats.shape
    s = unit(support).reshape(-1, h, w, d)                 # (k, h, w, D)
    out = torch.empty(n, h, w, device=test_feats.device)
    for c in range(0, n, chunk):
        q = unit(test_feats[c:c + chunk]).reshape(-1, h, w, d)
        best = torch.full(q.shape[:3], -2.0, device=q.device)
        for ty, tx, sy, sx in _windows(h, w, radius):
            sim = torch.einsum("nyxd,kyxd->nkyx", q[:, ty, tx], s[:, sy, sx]).amax(1)
            best[:, ty, tx] = torch.maximum(best[:, ty, tx], sim)
        out[c:c + chunk] = (1.0 - best).clamp_min(0.0)
    return out.reshape(n, p)


def positional_patch_scores(test_feats, test_fg, support, grid, radius: int) -> torch.Tensor:
    """Positional distances with background patches set to 0, like baseline_patch_scores."""
    d = positional_distances(test_feats, support, grid, radius)
    return torch.where(test_fg, d, torch.zeros_like(d))


@torch.no_grad()
def alignment_ratio(support: torch.Tensor, fg: torch.Tensor, grid: tuple[int, int],
                    radius: int = 1) -> float:
    """How well the support images line up: mean positional / mean global distance, image vs image.

    support (k, P, D) unrotated support features, fg (k, P) foreground. Near 1: objects sit in
    the same place, so positional matching loses nothing on normal patches. Clearly above 1:
    poses vary (screw, textures), so positional matching would raise false alarms.
    Needs k >= 2; returns nan otherwise.
    """
    k = support.shape[0]
    ratios = []
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            m = fg[a]
            if not bool(m.any()):
                continue
            d_pos = positional_distances(support[a:a + 1], support[b:b + 1], grid, radius)[0]
            d_glob = knn_distances(unit(support[a]), unit(support[b]), k=1)[:, 0]
            ratios.append(float(d_pos[m].mean() / d_glob[m].mean().clamp_min(1e-8)))
    return float(sum(ratios) / len(ratios)) if ratios else float("nan")
