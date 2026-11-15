"""Positional kNN: matches brute force, reduces to global 1-NN at large radius. Needs PyTorch."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.positional import (alignment_ratio, margin_scores, positional_distances,  # noqa: E402
                              positional_patch_scores, support_gaps)
from fsiad.scoring import knn_distances, unit                                              # noqa: E402

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def test_margin_scores_equal_baseline_below_margin():
    g = torch.tensor([[0.1, 0.2, 0.3]])
    p = torch.tensor([[0.15, 0.6, 0.3]])                     # gaps 0.05, 0.4, 0
    s = margin_scores(g, p, margin=0.1, weight=1.0)
    assert torch.allclose(s, torch.tensor([[0.1, 0.5, 0.3]]))
    assert torch.equal(margin_scores(g, p, margin=1.0, weight=1.0), g)


def test_support_gaps_nonnegative_and_zero_for_identical_supports():
    pool = torch.randn(2, 3, 16, 8)
    fg = torch.ones(2, 16, dtype=torch.bool)
    gaps = support_gaps(pool, fg, (4, 4), 1)
    assert gaps.shape == (32,) and torch.all(gaps >= 0)
    same = pool[:1].repeat(2, 1, 1, 1)                        # two identical images: no gap
    assert torch.allclose(support_gaps(same, fg, (4, 4), 1), torch.zeros(32), atol=1e-5)


def _fake_cd():
    from fsiad.episodes import CategoryData
    g = torch.Generator().manual_seed(1)
    P, D = 64, 8
    types = ["good"] * 4 + ["scratch"] * 4
    labels = torch.zeros(8, P, dtype=torch.int8)
    labels[4:, :5] = 1
    return CategoryData({
        "category": "fake", "grid": (8, 8),
        "pool_feats": torch.randn(4, 2, P, D, generator=g),
        "train_feats": torch.randn(6, P, D, generator=g),
        "test_feats": torch.randn(8, P, D, generator=g),
        "train_fg": torch.ones(6, P, dtype=torch.bool),
        "test_fg": torch.ones(8, P, dtype=torch.bool),
        "test_labels": labels, "test_types": types,
        "test_shapes": [(32, 32)] * 8, "test_masks": [None] * 8,
        "defect_types": ["scratch"],
    }, DEV, load_train=False)


class _StubEvaluator:
    """Records a score summary instead of computing pixel metrics."""

    def evaluate(self, scores, subsets, pixel=True):
        return {k: {"px_aupro": float(scores[v].sum())} for k, v in subsets.items()}


def test_positional_parts_and_term():
    from fsiad.positional import positional_parts, positional_term
    cd = _fake_cd()
    base, gap, sgap = positional_parts(cd.test, cd.test_fg, cd.pool[[0, 1]], cd.pool_fg[[0, 1]], cd.grid, 2)
    assert base.shape == gap.shape == (8, 64) and torch.all(gap >= 0) and sgap.numel() == 128
    assert torch.equal(positional_term(gap, sgap, 90, 0.0), torch.zeros_like(gap))
    t = positional_term(gap, sgap, 50, 1.0)
    assert torch.all(t >= 0) and torch.all(t <= gap)


def test_evaluate_methods_adds_pos_rows():
    from fsiad.model import make_model
    from fsiad.runner import evaluate_methods
    cd = _fake_cd()
    cfg = {"softmin_k": 5}
    methods = {"baseline": make_model("baseline", 8, cfg).to(DEV).eval(),
               "fixed_bank": make_model("fixed_bank", 8, cfg, lam=0.2, tau_d=0.1).to(DEV).eval()}
    methods["random_control"] = methods["fixed_bank"]
    rows = evaluate_methods(methods, cd, _StubEvaluator(), [(2, 1), (1, 0)], [0],
                            positional={"quantile": 50, "weight": 1.0, "radius": 2})
    names = {(r["k0"], r["method"]) for r in rows}
    assert (2, "baseline+pos") in names and (2, "fixed_bank+pos") in names
    assert not any(m == "random_control+pos" for _, m in names)
    assert not any(k == 1 and m.endswith("+pos") for k, m in names)   # margin needs k0 >= 2
    off = evaluate_methods(methods, cd, _StubEvaluator(), [(2, 0)], [0],
                           positional={"quantile": 50, "weight": 0.0, "radius": 2})
    v = {r["method"]: r["px_aupro"] for r in off if r["subset"] == "all"}
    assert v["baseline+pos"] == v["baseline"]                        # weight 0 = identical
