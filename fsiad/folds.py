"""Leave-categories-out folds (spec Section 10).

5 folds x 3 test categories, each fold = 1 texture + 2 objects. Within a fold, 2 of the
remaining 12 categories (1 texture, 1 object) are validation, the other 10 are training.
The assignment is fixed by `seed` and must not change once experiments start.
"""
from __future__ import annotations

import numpy as np

TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECTS = ["bottle", "cable", "capsule", "hazelnut", "metal_nut",
           "pill", "screw", "toothbrush", "transistor", "zipper"]


def make_folds(seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    tex = list(rng.permutation(TEXTURES))
    obj = list(rng.permutation(OBJECTS))
    return [[str(tex[i]), str(obj[2 * i]), str(obj[2 * i + 1])] for i in range(5)]


def fold_split(fold: int, seed: int = 0) -> dict[str, list[str]]:
    """Return {'train': 10 cats, 'val': 2 cats, 'test': 3 cats} for one fold."""
    folds = make_folds(seed)
    test = folds[fold]
    nxt = folds[(fold + 1) % 5]
    val = [nxt[0], nxt[1]]                      # next fold's texture + first object
    train = [c for f in folds for c in f if c not in test and c not in val]
    return {"train": train, "val": val, "test": test}
