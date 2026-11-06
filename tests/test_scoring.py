"""Sanity checks for scoring (spec Section 13, checks 1-2). Needs PyTorch; skipped without it."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.scoring import (anomaly_map, baseline_patch_scores, fused_score, image_score,  # noqa: E402
                           knn_distances, softmin, unit)

torch.manual_seed(0)


def test_knn_matches_brute_force_and_chunking():
    q, b = unit(torch.randn(50, 16)), unit(torch.randn(300, 16))
    brute = (1 - q @ b.T).sort(dim=1).values[:, :5].clamp_min(0)
    assert torch.allclose(knn_distances(q, b, k=5), brute, atol=1e-6)
    assert torch.allclose(knn_distances(q, b, k=5, chunk=37), brute, atol=1e-6)


def test_cosine_equals_half_squared_euclidean():          # spec Fact 1
    u, v = unit(torch.randn(10, 8)), unit(torch.randn(10, 8))
    assert torch.allclose(1 - (u * v).sum(1), 0.5 * ((u - v) ** 2).sum(1), atol=1e-6)


def test_softmin_bounds_and_tau_derivative():               # spec Prop 4.1, Section 8
    d = torch.rand(200, 10, dtype=torch.float64)
    for tau in [0.01, 0.1, 1.0]:
        m = softmin(d, tau)
        dmin = d.min(dim=1).values
        assert torch.all(m <= dmin + 1e-12)
        assert torch.all(m >= dmin - tau * np.log(10) - 1e-12)
    t = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    softmin(d, t).sum().backward()
    assert t.grad <= 0
    assert torch.allclose(softmin(d, 1e-4), d.min(dim=1).values, atol=1e-3)


def test_softmin_gradcheck():                               # spec Section 13, check 3
    d = torch.rand(4, 6, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda d, t: softmin(d, t), (d, t))


def test_fusion_lambda_zero_is_baseline():
    dn, dd = torch.rand(100), torch.rand(100)
    assert torch.equal(fused_score(dn, dd, 0.0, 0.1), dn)
    assert torch.equal(fused_score(dn, None, 5.0, 0.1), dn)
    s = fused_score(dn, dd, 0.5, 0.1)
    assert torch.all(s >= dn) and torch.all(s - dn <= 0.5 + 1e-7)   # spec Prop 5.2


def test_image_score_top10_of_1024():
    x = np.zeros(1024)
    x[:10] = 1.0
    x[10:20] = 0.5
    assert image_score(x) == 1.0
    assert float(image_score(torch.tensor(x))) == 1.0
    assert image_score(np.array([0.3, 0.9])) == 0.9          # count 0 -> max


def test_baseline_masked_patches_score_zero():
    feats = torch.randn(2, 16, 8)
    fg = torch.ones(2, 16, dtype=torch.bool)
    fg[0, :4] = False
    bank = unit(torch.randn(30, 8))
    s = baseline_patch_scores(feats, fg, bank)
    assert torch.all(s[0, :4] == 0) and torch.all(s[0, 4:] > 0)


def test_anomaly_map_shape():
    m = anomaly_map(np.random.rand(1024), (32, 32), (900, 900), sigma=4.0)
    assert m.shape == (900, 900) and m.dtype == np.float32
