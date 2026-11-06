"""Cached category data on the GPU, support sets, banks and training episodes (spec Section 9)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class CategoryData:
    """One category's cache, moved to `device` once."""

    def __init__(self, c: dict, device: str):
        self.name = c["category"]
        self.grid = tuple(c["grid"])
        self.pool = c["pool_feats"].to(device)            # (Npool, R, P, D) half
        self.train = c["train_feats"].to(device)          # (Ntrain, P, D) half
        self.test = c["test_feats"].to(device)            # (Ntest, P, D) half
        self.train_fg = c["train_fg"].to(device)
        self.test_fg = c["test_fg"].to(device)
        self.test_labels = c["test_labels"].to(device)    # (Ntest, P) int8: 1 / 0 / -1
        self.test_types = list(c["test_types"])
        self.test_shapes = [tuple(s) for s in c["test_shapes"]]
        self.test_masks = list(c["test_masks"])
        self.defect_types = list(c["defect_types"])
        self.n_pool = self.pool.shape[0]
        self.dim = self.test.shape[-1]
        self.good_idx = [i for i, t in enumerate(self.test_types) if t == "good"]
        self.type_idx = {t: [i for i, x in enumerate(self.test_types) if x == t]
                         for t in self.defect_types}
        self.image_labels = np.array([t != "good" for t in self.test_types], dtype=np.int64)


@dataclass
class Support:
    normal_idx: list[int]      # indices into the support pool (= first train images)
    defect_idx: list[int]      # indices into the test set (images of `seen_type`)
    seen_type: str | None


def sample_support(cd: CategoryData, rng: np.random.Generator, k0: int, kd: int,
                   seen_type: str | None) -> Support:
    normal = sorted(rng.choice(cd.n_pool, size=k0, replace=False).tolist())
    defect = []
    if kd > 0:
        pool = cd.type_idx[seen_type]
        defect = sorted(rng.choice(pool, size=min(kd, len(pool) - 1), replace=False).tolist())
    return Support(normal, defect, seen_type)


def build_banks(cd: CategoryData, s: Support, rotation: bool = True):
    """Raw (un-adapted) normal bank (M, D) and defect bank (Md, D) or None."""
    ref = cd.pool[s.normal_idx]                           # (k0, R, P, D)
    if not rotation:
        ref = ref[:, :1]
    normal = ref.reshape(-1, ref.shape[-1])
    if not s.defect_idx:
        return normal, None
    feats = cd.test[s.defect_idx]                         # (kd, P, D)
    labels = cd.test_labels[s.defect_idx]
    defect = feats[labels == 1]
    if defect.shape[0] == 0:                              # tiny defects: fall back to partial patches
        defect = feats[labels != 0]
    return normal, defect


def random_control_bank(normal: torch.Tensor, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Same number of patches as the defect bank, drawn from the normal support (spec 9.3)."""
    idx = torch.as_tensor(rng.choice(normal.shape[0], size=n, replace=False), device=normal.device)
    return normal[idx]


def sample_episode(cd: CategoryData, rng: np.random.Generator, cfg: dict) -> dict:
    """One training episode (spec 9.1)."""
    k0 = int(rng.choice(cfg["k0_choices"]))
    kd = int(rng.choice(cfg["kd_choices"]))
    t = str(rng.choice(cd.defect_types))
    n_seen, n_held, n_norm = cfg["n_query_seen"], cfg["n_query_heldout"], cfg["n_query_normal"]

    normal_idx = sorted(rng.choice(cd.n_pool, size=k0, replace=False).tolist())
    perm = rng.permutation(cd.type_idx[t]).tolist()
    kd = min(kd, max(len(perm) - n_seen, 0))
    defect_idx = perm[:kd]
    if cfg.get("defect_query_mode", "balanced") == "natural":
        # every remaining defect image equally likely, so the seen type's share of the
        # queries matches its share of the test set (about 1 / number of types)
        pool = [i for u in cd.defect_types for i in cd.type_idx[u] if i not in defect_idx]
        defect_q = rng.choice(pool, size=min(n_seen + n_held, len(pool)), replace=False).tolist()
    else:
        seen_q = perm[kd:kd + n_seen]
        others = [i for u in cd.defect_types if u != t for i in cd.type_idx[u]]
        if others:
            held_q = rng.choice(others, size=min(n_held, len(others)), replace=False).tolist()
        else:                                             # single-type category (toothbrush)
            held_q = perm[kd + n_seen:kd + n_seen + n_held]
        defect_q = seen_q + held_q

    if cfg.get("normal_query_source", "train") == "test_good":
        # same acquisition split as the defective queries, as at evaluation time
        normal_q = rng.choice(cd.good_idx, size=min(n_norm, len(cd.good_idx)), replace=False).tolist()
        nq_feats, nq_fg = cd.test[normal_q], cd.test_fg[normal_q]
    else:
        free = np.setdiff1d(np.arange(cd.train.shape[0]), normal_idx)   # pool image j = train image j
        normal_q = rng.choice(free, size=n_norm, replace=False).tolist()
        nq_feats, nq_fg = cd.train[normal_q], cd.train_fg[normal_q]

    support = Support(normal_idx, sorted(defect_idx), t)
    normal_bank, defect_bank = build_banks(cd, support, rotation=cfg.get("rotation", True))
    dev = cd.test.device
    q = torch.cat([nq_feats, cd.test[defect_q]])
    fg = torch.cat([nq_fg, cd.test_fg[defect_q]])
    labels = torch.cat([torch.zeros(len(normal_q), q.shape[1], dtype=torch.int8, device=dev),
                        cd.test_labels[defect_q]])
    img_labels = torch.tensor([0] * len(normal_q) + [1] * len(defect_q), device=dev)
    return {"q": q, "fg": fg, "labels": labels, "img_labels": img_labels,
            "normal_bank": normal_bank, "defect_bank": defect_bank, "k0": k0, "kd": len(defect_idx)}
