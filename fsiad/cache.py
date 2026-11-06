"""Run DINOv2 once and store everything later stages need (spec Section 10).

One file per category, `<cache_dir>/<category>.pt`, holding:
  train_feats  (N_train, P, D) float16   every train/good image, unrotated
  pool_feats   (N_pool, R, P, D) float16 first `pool_size` train images x R rotations
                                         (rotation 0 = the unrotated image)
  test_feats   (N_test, P, D) float16    every test image, unrotated
  train_fg / test_fg  (N, P) bool        foreground masks (all True if category unmasked)
  test_labels  (N_test, P) int8          patch labels 1 / 0 / -1 (ignored)
  plus paths, defect types, image shapes and the patch grid.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from .backbone import DINOv2Extractor
from .preprocess import background_mask, rotate_image
from .data import CategoryIndex, load_mask, load_rgb, patch_labels


def _extract_images(extractor: DINOv2Extractor, images: list[np.ndarray], batch_size: int):
    feats, grid = [], None
    for i in range(0, len(images), batch_size):
        prepared = [extractor.prepare(im) for im in images[i:i + batch_size]]
        grids = {g for _, g in prepared}
        if len(grids) != 1:
            raise ValueError(f"images in a batch gave different grids: {grids}")
        g = grids.pop()
        if grid is None:
            grid = g
        elif g != grid:
            raise ValueError(f"grid changed within a category: {grid} vs {g}")
        feats.append(extractor.extract([t for t, _ in prepared]))
    return torch.cat(feats), grid


def _extract_paths(extractor, paths, batch_size, desc):
    """Load + extract in chunks so we never hold all full-resolution images at once."""
    feats, grid, shapes = [], None, []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, leave=False):
        imgs = [load_rgb(p) for p in paths[i:i + batch_size]]
        shapes += [im.shape[:2] for im in imgs]
        f, g = _extract_images(extractor, imgs, batch_size)
        feats.append(f)
        grid = grid or g
    return torch.cat(feats), grid, shapes


def _fg_masks(feats: torch.Tensor, grid, masked: bool, cfg: dict) -> torch.Tensor:
    if not masked:
        return torch.ones(feats.shape[:2], dtype=torch.bool)
    return torch.from_numpy(np.stack([
        background_mask(f.numpy(), grid, cfg["mask_threshold"], cfg["mask_kernel"], cfg["mask_border"])
        for f in feats
    ]))


def cache_category(extractor: DINOv2Extractor, idx: CategoryIndex, cfg: dict) -> str:
    out = os.path.join(cfg["cache_dir"], f"{idx.name}.pt")
    if os.path.exists(out):
        return out
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    bs = cfg["batch_size"]
    masked = idx.name in cfg["masked_categories"]

    train_feats, grid, _ = _extract_paths(extractor, idx.train, bs, f"{idx.name} train")
    test_feats, g2, test_shapes = _extract_paths(extractor, idx.test, bs, f"{idx.name} test")
    assert g2 == grid, (grid, g2)

    # Rotated support pool: rotate the full-resolution image first, then preprocess.
    angles = cfg["rotation_angles"]
    pool = []
    for p in tqdm(idx.pool, desc=f"{idx.name} pool rotations", leave=False):
        img = load_rgb(p)
        f, g = _extract_images(extractor, [rotate_image(img, a) for a in angles], len(angles))
        assert g == grid
        pool.append(f)
    pool_feats = torch.stack(pool)                                   # (N_pool, R, P, D)

    test_labels = np.stack([
        patch_labels(load_mask(m, s), grid, cfg["label_rho"])
        for m, s in zip(idx.test_masks, test_shapes)
    ])

    torch.save({
        "category": idx.name,
        "grid": grid,
        "angles": angles,
        "train_paths": idx.train,
        "pool_paths": idx.pool,
        "test_paths": idx.test,
        "test_types": idx.test_types,
        "test_masks": idx.test_masks,
        "test_shapes": test_shapes,
        "defect_types": idx.defect_types,
        "train_feats": train_feats.half(),
        "pool_feats": pool_feats.half(),
        "test_feats": test_feats.half(),
        "train_fg": _fg_masks(train_feats, grid, masked, cfg),
        "test_fg": _fg_masks(test_feats, grid, masked, cfg),
        "test_labels": torch.from_numpy(test_labels),
    }, out)
    return out


def load_category(cache_dir: str, name: str) -> dict:
    return torch.load(os.path.join(cache_dir, f"{name}.pt"), weights_only=False)
