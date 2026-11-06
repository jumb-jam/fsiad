"""MVTec AD indexing, image/mask loading and patch labels (spec Section 1)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load YAML config; `overrides` like ["cache_dir=/kaggle/input/x/cache", "steps=100"]."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        cfg[key.strip()] = yaml.safe_load(value)
    return cfg


@dataclass
class CategoryIndex:
    """All file paths for one MVTec category, in a fixed (sorted) order."""
    name: str
    train: list[str]                 # train/good images, sorted by filename
    pool: list[str]                  # first `pool_size` train images: the support pool
    test: list[str]                  # all test images
    test_types: list[str]            # 'good' or the defect type, one per test image
    test_masks: list[str | None]     # ground-truth mask path, None for good images
    defect_types: list[str] = field(default_factory=list)

    @property
    def test_labels(self) -> np.ndarray:
        """Image-level labels: 1 = defective, 0 = good."""
        return np.array([t != "good" for t in self.test_types], dtype=np.int64)


def _pngs(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))


def index_category(root: str, name: str, pool_size: int = 16) -> CategoryIndex:
    cat_dir = os.path.join(root, name)
    train_dir = os.path.join(cat_dir, "train", "good")
    # AnomalyDINO sorts filenames inside train/good; we do the same.
    train = [os.path.join(train_dir, f) for f in _pngs(train_dir)]

    test, types, masks = [], [], []
    test_dir = os.path.join(cat_dir, "test")
    defect_types = sorted(d for d in os.listdir(test_dir)
                          if d != "good" and os.path.isdir(os.path.join(test_dir, d)))
    for t in ["good"] + defect_types:
        for f in _pngs(os.path.join(test_dir, t)):
            test.append(os.path.join(test_dir, t, f))
            types.append(t)
            if t == "good":
                masks.append(None)
            else:
                m = os.path.join(cat_dir, "ground_truth", t, os.path.splitext(f)[0] + "_mask.png")
                if not os.path.exists(m):
                    raise FileNotFoundError(f"missing mask for {test[-1]}: {m}")
                masks.append(m)

    return CategoryIndex(name=name, train=train, pool=train[:pool_size], test=test,
                         test_types=types, test_masks=masks, defect_types=defect_types)


def index_dataset(root: str, categories: list[str], pool_size: int = 16) -> dict[str, CategoryIndex]:
    return {c: index_category(root, c, pool_size) for c in categories}


def load_rgb(path: str) -> np.ndarray:
    """Load an image as RGB uint8 (H, W, 3), exactly as AnomalyDINO does."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    """Binary ground-truth mask (H, W) as bool. Good images get an all-False mask."""
    if path is None:
        return np.zeros(shape, dtype=bool)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != tuple(shape):
        raise ValueError(f"mask {path} has shape {m.shape}, image has {shape}")
    return m > 0


def patch_coverage(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Fraction of each patch's pixels inside the mask, shape (gh * gw,).

    Assumes the patch grid covers the whole image (true for square MVTec images
    resized to 448: 448 = 32 * 14, so nothing is trimmed).
    """
    gh, gw = grid
    cov = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return cov.reshape(-1)


def patch_labels(mask: np.ndarray, grid: tuple[int, int], rho: float = 0.5) -> np.ndarray:
    """Spec Section 1: 1 if coverage >= rho, 0 if coverage == 0, -1 (ignored) otherwise."""
    cov = patch_coverage(mask, grid)
    lab = np.full(cov.shape, -1, dtype=np.int8)
    lab[cov >= rho] = 1
    lab[cov < 1e-6] = 0
    return lab
