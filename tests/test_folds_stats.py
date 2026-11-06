"""Folds (leakage audit, spec 13 check 6) and statistics. No PyTorch needed."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.folds import OBJECTS, TEXTURES, fold_split, make_folds   # noqa: E402
from fsiad.stats import category_table, compare, holm                 # noqa: E402


def test_folds_cover_every_category_once():
    folds = make_folds(0)
    flat = [c for f in folds for c in f]
    assert sorted(flat) == sorted(TEXTURES + OBJECTS)
    for f in folds:
        assert sum(c in TEXTURES for c in f) == 1 and len(f) == 3


def test_split_has_no_leakage():
    for k in range(5):
        s = fold_split(k, 0)
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert len(tr) == 10 and len(va) == 2 and len(te) == 3
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert sum(c in TEXTURES for c in va) == 1
    assert fold_split(2, 0) == fold_split(2, 0)          # deterministic


def test_holm_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    adj = holm(p)
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->max(0.09)=0.09, 0.2*1=0.2
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_category_table_and_compare():
    rows = []
    for c in range(8):
        for seed in range(3):
            for m, v in [("baseline", 0.80), ("linear", 0.82)]:
                rows.append({"category": f"c{c}", "k0": 2, "kd": 1, "subset": "all", "seed": seed,
                             "seen_type": "t", "method": m, "px_aupro": v + 0.01 * c})
    tab = category_table(pd.DataFrame(rows), "px_aupro", 2, 1)
    assert tab.shape == (8, 2)
    r = compare(tab, "linear", "baseline", n_boot=500)
    assert abs(r["mean_delta"] - 0.02) < 1e-9 and r["wins"] == 8
    assert r["p_value"] < 0.01 and r["ci_low"] - 1e-9 <= 0.02 <= r["ci_high"] + 1e-9
