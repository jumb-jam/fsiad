"""Sanity checks for adapters, model, episodes, losses and GPU metrics (spec Section 13).

Needs PyTorch (skipped without it). Runs on CPU or GPU; on Kaggle it uses the GPU if present.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.adapters import LinearAdapter, LowRankAdapter, MLPAdapter      # noqa: E402
from fsiad.episodes import CategoryData, build_banks, sample_episode, sample_support  # noqa: E402
from fsiad.gpu_eval import make_maps, pixel_metrics_gpu                    # noqa: E402
from fsiad.losses import image_pair_loss, image_scores, patch_pair_loss    # noqa: E402
from fsiad.metrics import pixel_metrics                                    # noqa: E402
from fsiad.model import make_model                                         # noqa: E402
from fsiad.scoring import anomaly_map, baseline_patch_scores, unit         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, P = 16, 64                                       # tiny sizes: 8x8 grid


def fake_category(n_pool=6, n_train=10, n_test=12, rot=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    types = ["good"] * 4 + ["scratch"] * 4 + ["dent"] * 4
    labels = torch.zeros(n_test, P, dtype=torch.int8)
    for i in range(4, n_test):
        labels[i, (i * 3) % P:(i * 3) % P + 4] = 1
    test = torch.randn(n_test, P, D, generator=g)
    test[labels == 1] += 3.0
    return {
        "category": "fake", "grid": (8, 8),
        "pool_feats": torch.randn(n_pool, rot, P, D, generator=g).half(),
        "train_feats": torch.randn(n_train, P, D, generator=g).half(),
        "test_feats": test.half(),
        "train_fg": torch.ones(n_train, P, dtype=torch.bool),
        "test_fg": torch.ones(n_test, P, dtype=torch.bool),
        "test_labels": labels, "test_types": types,
        "test_shapes": [(32, 32)] * n_test, "test_masks": [None] * n_test,
        "defect_types": ["dent", "scratch"],
    }


CFG = {"k0_choices": [1, 2], "kd_choices": [0, 1, 2], "n_query_normal": 2, "n_query_seen": 1,
       "n_query_heldout": 1, "rotation": True, "softmin_k": 5}


# ---------------------------------------------------------------- spec 13 check 1
def test_untrained_adapters_equal_baseline():
    """Every adapter at init, with lambda = 0 and k = 1, reproduces the baseline scores."""
    torch.manual_seed(0)
    q = torch.randn(3, P, D, device=DEV)
    fg = torch.ones(3, P, dtype=torch.bool, device=DEV)
    fg[0, :5] = False
    bank = torch.randn(40, D, device=DEV)
    ref = baseline_patch_scores(q, fg, unit(bank))
    for m in ["baseline", "linear", "mlp", "lowrank"]:
        model = make_model(m, D, {"softmin_k": 1}).to(DEV)
        model.k = 1
        model.hard = True
        with torch.no_grad():
            s = model.patch_scores(q, fg, bank)
        assert torch.allclose(s, ref, atol=1e-5), m


# ---------------------------------------------------------------- spec 13 check 4
def test_mlp_init_trains_but_all_zero_init_is_frozen():
    z = torch.randn(32, D)
    good = MLPAdapter(D, 8)
    (good(z) ** 3).sum().backward()
    assert good.fc2.weight.grad.abs().sum() > 0

    frozen = MLPAdapter(D, 8)
    with torch.no_grad():
        frozen.fc1.weight.zero_()
    (frozen(z) ** 3).sum().backward()
    assert frozen.fc2.weight.grad.abs().sum() == 0 and frozen.fc1.weight.grad.abs().sum() == 0


def test_adapters_start_as_identity():
    z = torch.randn(10, D)
    for a in [LinearAdapter(D), MLPAdapter(D, 8), LowRankAdapter(D, 4)]:
        assert torch.allclose(a(z), z)


# ---------------------------------------------------------------- episodes and banks
def test_episode_shapes_and_no_support_in_queries():
    cd = CategoryData(fake_category(), DEV)
    rng = np.random.default_rng(0)
    for _ in range(20):
        ep = sample_episode(cd, rng, CFG)
        n = ep["q"].shape[0]
        assert ep["fg"].shape == (n, P) and ep["labels"].shape == (n, P)
        assert ep["normal_bank"].shape[1] == D
        assert ep["normal_bank"].shape[0] == ep["k0"] * 2 * P        # rot = 2
        assert (ep["img_labels"] == 1).sum() >= 1 and (ep["img_labels"] == 0).sum() == 2


def test_support_defects_not_in_test_subsets():
    from fsiad.runner import test_subsets
    cd = CategoryData(fake_category(), DEV)
    sup = sample_support(cd, np.random.default_rng(1), 2, 2, "scratch")
    subs = test_subsets(cd, sup, "scratch")
    for s in subs.values():
        assert not set(sup.defect_idx) & set(s)
    assert all(cd.test_types[i] in ("good", "scratch") for i in subs["seen"])
    normal, defect = build_banks(cd, sup)
    assert defect.shape[0] == 8                       # 2 images x 4 labelled patches


# ---------------------------------------------------------------- spec 13 check 5
def test_overfit_one_episode():
    torch.manual_seed(0)
    cd = CategoryData(fake_category(), DEV)
    ep = sample_episode(cd, np.random.default_rng(3), {**CFG, "kd_choices": [1]})
    model = make_model("linear", D, {"softmin_k": 5}, tau=0.05, tau_d=0.05).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for _ in range(150):
        s = model.patch_scores(ep["q"], ep["fg"], ep["normal_bank"], ep["defect_bank"])
        loss = patch_pair_loss(s, ep["labels"], ep["fg"], a=20.0, neg_per_pos=32)
        loss = loss + 0.5 * image_pair_loss(image_scores(s), ep["img_labels"], a=20.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.project()
        losses.append(float(loss))
    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])
    assert float(model.lam) >= 0


# ---------------------------------------------------------------- GPU metrics vs CPU
def test_gpu_maps_and_metrics_match_cpu():
    rng = np.random.default_rng(0)
    n, grid, hw = 4, (8, 8), (56, 56)
    patch = rng.random((n, 64)).astype(np.float32)
    gts = []
    for i in range(n):
        g = np.zeros(hw, bool)
        if i % 2:
            g[10:20, 10:30] = True
            g[40:45, 5:9] = True
        gts.append(g)
    cpu_maps = [anomaly_map(p, grid, hw, sigma=4.0) for p in patch]
    gpu_maps = make_maps(torch.tensor(patch, device=DEV), grid, hw, sigma=4.0)
    assert np.abs(gpu_maps.cpu().numpy() - np.stack(cpu_maps)).max() < 1e-4

    from scipy.ndimage import label
    ws, regions = [], 0
    for g in gts:
        lab, k = label(g, np.ones((3, 3), int))
        sizes = np.bincount(lab.ravel()).astype(np.float32)
        ws.append(np.where(lab > 0, 1.0 / sizes[lab], 0.0).astype(np.float32))
        regions += k
    cpu = pixel_metrics(cpu_maps, gts)
    gpu = pixel_metrics_gpu(torch.tensor(np.stack(cpu_maps), device=DEV),
                            torch.tensor(np.stack(gts), device=DEV),
                            torch.tensor(np.stack(ws), device=DEV), regions)
    for k in cpu:
        assert abs(cpu[k] - gpu[k]) < 1e-6, (k, cpu[k], gpu[k])


def test_softmin_gradient_flows_to_adapter():
    model = make_model("linear", D, {"softmin_k": 5}).to(DEV)
    q = torch.randn(2, P, D, device=DEV)
    s = model.patch_scores(q, torch.ones(2, P, dtype=torch.bool, device=DEV),
                           torch.randn(50, D, device=DEV), torch.randn(10, D, device=DEV))
    s.sum().backward()
    assert model.adapter.W.grad.abs().sum() > 0
    assert model.lam.grad is not None and model.log_tau.grad is not None
