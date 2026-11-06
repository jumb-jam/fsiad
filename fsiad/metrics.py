"""Image- and pixel-level metrics (spec Section 11).

`compute_pro` and `trapezoid` are ported from AnomalyDINO's post_eval.py, which builds on
the official MVTec AD evaluation code, so AU-PRO numbers are comparable to the paper.
"""
from __future__ import annotations

from bisect import bisect

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(labels, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
    return float(np.max(f1[np.isfinite(f1)]))


def image_metrics(labels, scores) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    return {
        "img_auroc": float(roc_auc_score(labels, scores)),
        "img_ap": float(average_precision_score(labels, scores)),
        "img_f1max": f1_max(labels, scores),
    }


def trapezoid(x, y, x_max=None) -> float:
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        keep = x <= x_max
        x, y = x[keep], y[keep]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def compute_pro(anomaly_maps: list[np.ndarray], gt_maps: list[np.ndarray]):
    """PRO curve over a set of maps. Returns (fprs, pros), each starting at 0 and ending at 1."""
    structure = np.ones((3, 3), dtype=int)
    n = len(anomaly_maps)
    h, w = anomaly_maps[0].shape
    fp_changes = np.zeros((n, h, w), dtype=np.uint32)
    assert n * h * w < np.iinfo(np.uint32).max
    pro_changes = np.zeros((n, h, w), dtype=np.float64)
    num_ok, num_regions = 0, 0
    for i, gt in enumerate(gt_maps):
        labeled, n_comp = label(gt, structure)
        num_regions += n_comp
        ok = labeled == 0
        num_ok += int(ok.sum())
        fp_changes[i][ok] = 1
        for c in range(n_comp):
            region = labeled == (c + 1)
            pro_changes[i][region] = 1.0 / region.sum()

    scores = np.stack(anomaly_maps).ravel()
    order = np.argsort(scores, kind="stable")[::-1]
    scores = scores[order]
    fprs = np.cumsum(fp_changes.ravel()[order]).astype(np.float64) / num_ok
    pros = np.cumsum(pro_changes.ravel()[order]) / num_regions
    del order, fp_changes, pro_changes

    keep = np.append(np.diff(scores) != 0, True)          # last point per threshold
    fprs, pros = np.clip(fprs[keep], None, 1.0), np.clip(pros[keep], None, 1.0)
    return np.concatenate([[0.0], fprs, [1.0]]), np.concatenate([[0.0], pros, [1.0]])


def pixel_metrics(anomaly_maps: list[np.ndarray], gt_maps: list[np.ndarray],
                  fpr_limit: float = 0.3, compute_ap: bool = True) -> dict:
    """Pooled over all pixels of all test images of one category."""
    fprs, pros = compute_pro(anomaly_maps, gt_maps)
    out = {"px_aupro": trapezoid(fprs, pros, x_max=fpr_limit) / fpr_limit}
    gt = np.concatenate([g.ravel() for g in gt_maps]).astype(np.uint8)
    pred = np.concatenate([a.ravel() for a in anomaly_maps])
    out["px_auroc"] = float(roc_auc_score(gt, pred))
    out["px_f1max"] = f1_max(gt, pred)
    if compute_ap:
        out["px_ap"] = float(average_precision_score(gt, pred))
    return out
