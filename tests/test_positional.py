"""Positional kNN: matches brute force, reduces to global 1-NN at large radius. Needs PyTorch."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.positional import alignment_ratio, positional_distances, positional_patch_scores  # noqa: E402
from fsiad.scoring import knn_distances, unit                                              # noqa: E402

torch.manual_seed(0)


def _brute(q, s, h, w, r):
    qn, sn = unit(q).reshape(-1, h, w, q.shape[-1]), unit(s).reshape(-1, h, w, s.shape[-1])
    out = torch.empty(qn.shape[:3])
    for n in range(qn.shape[0]):
        for y in range(h):
            for x in range(w):
                win = sn[:, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, sn.shape[-1])
                out[n, y, x] = 1 - (win @ qn[n, y, x]).max()
    return out.clamp_min(0).reshape(qn.shape[0], -1)


@pytest.mark.parametrize("h,w,r", [(5, 4, 1), (6, 6, 2), (4, 7, 0)])
def test_positional_matches_brute_force(h, w, r):
    q, s = torch.randn(3, h * w, 8), torch.randn(2, h * w, 8)
    assert torch.allclose(positional_distances(q, s, (h, w), r, chunk=2), _brute(q, s, h, w, r), atol=1e-5)


def test_large_radius_equals_global_knn():
    q, s = torch.randn(2, 20, 8), torch.randn(3, 20, 8)
    pos = positional_distances(q, s, (4, 5), radius=10)
    glob = torch.stack([knn_distances(unit(q[i]), unit(s.reshape(-1, 8)), k=1)[:, 0] for i in range(2)])
    assert torch.allclose(pos, glob, atol=1e-5)


def test_background_zero_and_alignment_ratio_at_least_one():
    q, s = torch.randn(2, 16, 8), torch.randn(2, 16, 8)
    fg = torch.ones(2, 16, dtype=torch.bool)
    fg[0, :3] = False
    sc = positional_patch_scores(q, fg, s, (4, 4), 1)
    assert torch.all(sc[0, :3] == 0) and torch.all(sc[0, 3:] > 0)
    assert alignment_ratio(s, torch.ones(2, 16, dtype=torch.bool), (4, 4), 1) >= 1.0 - 1e-6
