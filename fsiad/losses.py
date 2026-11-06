"""Pairwise logistic loss, an upper bound on 1 - AUROC (spec Section 7)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def pair_loss(margin: torch.Tensor, a: float) -> torch.Tensor:
    """log2(1 + exp(-a * margin)), averaged."""
    return (F.softplus(-a * margin) / LN2).mean()


def patch_pair_loss(scores, labels, fg, a: float, neg_per_pos: int = 64,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Defective vs normal foreground patches, pooled over the episode's query images.

    Each positive is paired with `neg_per_pos` random negatives (unbiased estimate of the
    full pairwise loss). Ignored (-1) and background patches are excluded.
    """
    pos = scores[(labels == 1) & fg]
    neg = scores[(labels == 0) & fg]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    j = torch.randint(neg.numel(), (pos.numel(), neg_per_pos), device=scores.device,
                      generator=generator)
    return pair_loss(pos[:, None] - neg[j], a)


def image_scores(scores: torch.Tensor, fraction: float = 0.01) -> torch.Tensor:
    """Mean of the top floor(fraction * P) patch scores per image, differentiable. (N, P) -> (N,)"""
    n = max(int(scores.shape[1] * fraction), 1)
    return scores.topk(n, dim=1).values.mean(dim=1)


def image_pair_loss(img_scores, img_labels, a: float) -> torch.Tensor:
    pos, neg = img_scores[img_labels == 1], img_scores[img_labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return img_scores.sum() * 0.0
    return pair_loss(pos[:, None] - neg[None, :], a)
