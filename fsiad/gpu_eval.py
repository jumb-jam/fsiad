"""Full-resolution maps and exact pixel metrics on the GPU (spec Sections 6, 11).

Same definitions as fsiad.metrics (and so as AnomalyDINO / the MVTec PRO code), but the
upsampling, blur, single sort and cumulative sums run on the GPU. A test checks agreement
with the CPU implementation.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from .data import load_mask
from .losses import image_scores
from .metrics import image_metrics


# ---------------------------------------------------------------- maps
def _gaussian_kernel(sigma: float, truncate: float = 4.0, device="cpu"):
    """Same 1-D kernel as scipy.ndimage.gaussian_filter."""
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).to(device), r


def _sym_pad(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """scipy 'reflect' padding (d c b a | a b c d | d c b a)."""
    left = x.narrow(dim, 0, r).flip(dim)
    right = x.narrow(dim, x.shape[dim] - r, r).flip(dim)
    return torch.cat([left, x, right], dim=dim)


def make_maps(patch_scores: torch.Tensor, grid, hw, sigma: float = 4.0, chunk: int = 16):
    """(N, P) patch scores -> (N, H, W) bilinear-upsampled, Gaussian-blurred maps."""
    n = patch_scores.shape[0]
    h, w = hw
    k, r = _gaussian_kernel(sigma, device=patch_scores.device)
    out = torch.empty(n, h, w, device=patch_scores.device)
    for s in range(0, n, chunk):
        g = patch_scores[s:s + chunk].double().reshape(-1, 1, *grid)
        up = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        up = F.conv2d(_sym_pad(up, r, 2), k.view(1, 1, -1, 1))
        up = F.conv2d(_sym_pad(up, r, 3), k.view(1, 1, 1, -1))
        out[s:s + chunk] = up[:, 0].float()
    return out


# ---------------------------------------------------------------- metrics
def _trapezoid_limited(x: torch.Tensor, y: torch.Tensor, x_max: float) -> float:
    """Area under y(x) for x in [0, x_max], interpolating at x_max (MVTec `trapezoid`)."""
    x, y = x.double(), y.double()
    correction = 0.0
    if not bool((x == x_max).any()):
        ins = int(torch.searchsorted(x, torch.tensor([x_max], dtype=x.dtype, device=x.device),
                                     right=True))
        y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
        correction = float(0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1]))
    keep = x <= x_max
    x, y = x[keep], y[keep]
    return float((0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum()) + correction


def pixel_metrics_gpu(maps: torch.Tensor, gt: torch.Tensor, pro_w: torch.Tensor,
                      num_regions: int, fpr_limit: float = 0.3, compute_ap: bool = True) -> dict:
    """maps (N,H,W) float, gt (N,H,W) bool, pro_w (N,H,W) = 1/region size on defect pixels."""
    pred = maps.reshape(-1)
    s, order = torch.sort(pred, descending=True, stable=True)
    last = torch.ones_like(s, dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]
    del s
    y = gt.reshape(-1)[order]
    tps = torch.cumsum(y.double(), 0)[last]
    fps = torch.cumsum((~y).double(), 0)[last]
    del y
    pro = torch.cumsum(pro_w.reshape(-1)[order].double(), 0)[last]
    del order, last

    zero, one = tps.new_zeros(1), tps.new_ones(1)
    n_pos, n_neg = tps[-1], fps[-1]
    tpr = torch.cat([zero, tps / n_pos])
    fpr = torch.cat([zero, fps / n_neg])
    out = {}
    pros = torch.cat([zero, (pro / num_regions).clamp(max=1.0), one])
    out["px_aupro"] = _trapezoid_limited(torch.cat([fpr, one]), pros, fpr_limit) / fpr_limit
    out["px_auroc"] = float(torch.trapezoid(tpr, fpr))
    precision = tps / (tps + fps)
    recall = tps / n_pos
    f1 = 2 * precision * recall / (precision + recall)
    out["px_f1max"] = float(f1[torch.isfinite(f1)].max())
    if compute_ap:
        out["px_ap"] = float((torch.diff(torch.cat([zero, recall])) * precision).sum())
    return out


# ---------------------------------------------------------------- per-category evaluator
def relocate(path: str, data_root: str, category: str) -> str:
    """Map a cached absolute path onto the current data_root (the dataset may move)."""
    i = path.rfind(f"/{category}/")
    return os.path.join(data_root, path[i + 1:]) if i >= 0 else path


class CategoryEvaluator:
    """Holds one category's ground truth on the GPU and scores any subset of test images."""

    def __init__(self, cd, data_root: str, cfg: dict):
        self.cd, self.cfg = cd, cfg
        dev = cd.test.device
        shapes = set(cd.test_shapes)
        if len(shapes) != 1:
            raise ValueError(f"{cd.name}: mixed image sizes {shapes}")
        self.hw = shapes.pop()
        structure = np.ones((3, 3), dtype=int)
        gts, ws, regions = [], [], []
        for m, hw in zip(cd.test_masks, cd.test_shapes):
            g = load_mask(None if m is None else relocate(m, data_root, cd.name), hw)
            lab, n = label(g, structure)
            if n:
                sizes = np.bincount(lab.ravel()).astype(np.float32)
                w = np.where(lab > 0, 1.0 / sizes[lab], 0.0).astype(np.float32)
            else:
                w = np.zeros(hw, np.float32)
            gts.append(torch.from_numpy(g))
            ws.append(torch.from_numpy(w))
            regions.append(n)
        self.gt = torch.stack(gts).to(dev)
        self.pro_w = torch.stack(ws).to(dev)
        self.regions = np.array(regions)

    def evaluate(self, patch_scores: torch.Tensor, subsets: dict[str, list[int]],
                 pixel: bool = True) -> dict[str, dict]:
        """patch_scores (Ntest, P) for every test image; subsets name -> test indices."""
        cfg = self.cfg
        img = image_scores(patch_scores, cfg["top_fraction"]).detach().cpu().numpy()
        maps = make_maps(patch_scores.detach(), self.cd.grid, self.hw, cfg["blur_sigma"]) if pixel else None
        out = {}
        for name, idx in subsets.items():
            labels = self.cd.image_labels[idx]
            if labels.min() == labels.max():          # needs both normal and defective images
                continue
            res = image_metrics(labels, img[idx])
            if pixel:
                t = torch.as_tensor(idx, device=maps.device)
                res.update(pixel_metrics_gpu(maps[t], self.gt[t], self.pro_w[t],
                                             int(self.regions[idx].sum()), cfg["pro_fpr_limit"],
                                             cfg["compute_pixel_ap"]))
            out[name] = res
        del maps
        return out
