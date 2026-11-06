"""Sanity checks that need no GPU or PyTorch: data indexing, patch labels, masking, metrics.

Run:  python -m pytest tests -q
"""
import os
import sys

import cv2
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.data import index_category, load_mask, patch_labels   # noqa: E402
from fsiad.metrics import compute_pro, image_metrics, pixel_metrics, trapezoid  # noqa: E402


# ---------------------------------------------------------------- data
def _fake_mvtec(root, name="widget"):
    d = os.path.join(root, name)
    for sub in ["train/good", "test/good", "test/scratch", "ground_truth/scratch"]:
        os.makedirs(os.path.join(d, sub))
    img = np.zeros((64, 64, 3), np.uint8)
    for i in [3, 1, 2, 0]:                                 # written out of order on purpose
        cv2.imwrite(os.path.join(d, "train/good", f"{i:03d}.png"), img)
    cv2.imwrite(os.path.join(d, "test/good", "000.png"), img)
    cv2.imwrite(os.path.join(d, "test/scratch", "000.png"), img)
    m = np.zeros((64, 64), np.uint8)
    m[:32, :32] = 255
    cv2.imwrite(os.path.join(d, "ground_truth/scratch", "000_mask.png"), m)
    return name


def test_index_category(tmp_path):
    name = _fake_mvtec(str(tmp_path))
    idx = index_category(str(tmp_path), name, pool_size=2)
    assert [os.path.basename(p) for p in idx.train] == ["000.png", "001.png", "002.png", "003.png"]
    assert [os.path.basename(p) for p in idx.pool] == ["000.png", "001.png"]
    assert idx.test_types == ["good", "scratch"]
    assert idx.test_masks[0] is None and idx.test_masks[1].endswith("000_mask.png")
    assert idx.defect_types == ["scratch"]
    assert list(idx.test_labels) == [0, 1]
    m = load_mask(idx.test_masks[1], (64, 64))
    assert m.sum() == 32 * 32


def test_patch_labels():
    mask = np.zeros((64, 64), bool)
    mask[:16, :16] = True        # patch (0,0) of a 4x4 grid fully covered
    mask[16:20, 16:32] = True    # patch (1,1) 25% covered -> ignored
    lab = patch_labels(mask, (4, 4), rho=0.5).reshape(4, 4)
    assert lab[0, 0] == 1 and lab[1, 1] == -1 and lab[3, 3] == 0
    assert (lab == 1).sum() == 1 and (lab == -1).sum() == 1


# ---------------------------------------------------------------- masking
def _anomalydino_mask(img_features, grid_size, threshold=10, kernel_size=3, border=0.2):
    """Verbatim logic of AnomalyDINO compute_background_mask (masking_type=True)."""
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    first_pc = pca.fit_transform(img_features.astype(np.float32))
    mask = first_pc > threshold
    m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1 - border)),
                                int(grid_size[1] * border):int(grid_size[1] * (1 - border))]
    if m.sum() <= m.size * 0.35:
        mask = - first_pc > threshold
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
    return mask.squeeze()


def test_background_mask_matches_anomalydino():
    try:
        from fsiad.preprocess import background_mask
    except ImportError:
        pytest.skip("preprocess import failed")
    rng = np.random.default_rng(0)
    g = (32, 32)
    yy, xx = np.mgrid[:32, :32]
    obj = ((yy - 16) ** 2 + (xx - 16) ** 2 < 100).reshape(-1)       # round object in the centre
    feats = rng.normal(size=(1024, 64)).astype(np.float32)
    feats[obj, 0] += 40.0
    ours = background_mask(feats, g)
    theirs = _anomalydino_mask(feats, g)
    assert ours.shape == (1024,)
    assert np.array_equal(ours, theirs)


# ---------------------------------------------------------------- metrics
def test_image_metrics_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert image_metrics(y, [0.1, 0.2, 0.8, 0.9])["img_auroc"] == 1.0
    assert image_metrics(y, [0.9, 0.8, 0.2, 0.1])["img_auroc"] == 0.0
    assert image_metrics(y, [0.1, 0.2, 0.8, 0.9])["img_f1max"] == 1.0


def test_trapezoid_limit():
    x, y = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])
    assert abs(trapezoid(x, y, x_max=0.3) - 0.045) < 1e-12        # area under y=x up to 0.3


def test_pixel_metrics_perfect_map():
    gt = [np.zeros((40, 40), bool), np.zeros((40, 40), bool)]
    gt[1][10:20, 10:20] = True
    maps = [g.astype(np.float32) + 0.01 * np.random.default_rng(i).random(g.shape).astype(np.float32)
            for i, g in enumerate(gt)]
    m = pixel_metrics(maps, gt, fpr_limit=0.3)
    assert m["px_auroc"] == 1.0 and abs(m["px_aupro"] - 1.0) < 1e-6 and m["px_f1max"] == 1.0
    assert abs(m["px_ap"] - 1.0) < 1e-9


def test_pixel_auroc_matches_sklearn_on_random_maps():
    rng = np.random.default_rng(1)
    gt = [rng.random((30, 30)) > 0.9 for _ in range(3)]
    maps = [rng.random((30, 30)).astype(np.float32) + 0.3 * g for g in gt]
    m = pixel_metrics(maps, gt)
    ref = roc_auc_score(np.concatenate([g.ravel() for g in gt]), np.concatenate([a.ravel() for a in maps]))
    assert abs(m["px_auroc"] - ref) < 1e-12
    fprs, pros = compute_pro(maps, gt)
    assert fprs[0] == 0 and fprs[-1] == 1 and np.all(np.diff(fprs) >= -1e-12)
    assert 0.0 <= m["px_aupro"] <= 1.0
