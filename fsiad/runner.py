"""Shared helpers for the training and evaluation scripts."""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch

from .cache import load_category
from .episodes import CategoryData, build_banks, random_control_bank, sample_support
from .gpu_eval import CategoryEvaluator
from .model import FSModel, make_model
from .positional import positional_parts, positional_term

DEFECT_METHODS = ("fixed_bank", "random_control")


def device_of(cfg: dict) -> str:
    return cfg["device"] if torch.cuda.is_available() else "cpu"


def load_cd(name: str, cfg: dict, device: str, load_train: bool = False) -> CategoryData:
    return CategoryData(load_category(cfg["cache_dir"], name), device, load_train=load_train)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def support_rng(seed: int, cat: str, seen_type: str | None, k0: int, kd: int) -> np.random.Generator:
    """Deterministic RNG shared by every method, so comparisons are paired (spec 9.2)."""
    key = [seed, k0, kd] + [ord(ch) for ch in f"{cat}|{seen_type}"]
    return np.random.default_rng(key)


def test_subsets(cd: CategoryData, support, seen_type: str | None) -> dict[str, list[int]]:
    """Test images minus the support defects: all / seen-type / held-out-type subsets."""
    used = set(support.defect_idx)
    remaining = [i for i in range(len(cd.test_types)) if i not in used]
    subsets = {"all": remaining}
    if seen_type is not None:
        subsets["seen"] = [i for i in remaining if cd.test_types[i] in ("good", seen_type)]
        held = [i for i in remaining if cd.test_types[i] != seen_type]
        if any(cd.test_types[i] != "good" for i in held):
            subsets["heldout"] = held
    return subsets


@torch.no_grad()
def score_support(model: FSModel, cd: CategoryData, normal, defect):
    return model.patch_scores(cd.test, cd.test_fg, normal, defect)


def evaluate_methods(methods: dict[str, FSModel], cd: CategoryData, ev: CategoryEvaluator,
                     settings, seeds, max_types: int | None = None, pixel: bool = True,
                     extra: dict | None = None, positional: dict | None = None):
    """Score every method on identical supports. Returns a list of result rows.

    positional: {"quantile", "weight", "radius"} or None. When given (and k0 >= 2), every
    method except random_control is also scored with the positional add-on, as "<name>+pos".
    """
    rows = []
    for k0, kd in settings:
        types = cd.defect_types[:max_types] if kd > 0 else [None]
        for t in types:
            for r in seeds:
                rng = support_rng(r, cd.name, t, k0, kd)
                sup = sample_support(cd, rng, k0, kd, t)
                normal, defect = build_banks(cd, sup)
                subsets = test_subsets(cd, sup, t)
                term = None
                if positional is not None and k0 >= 2:
                    _, gap, sgap = positional_parts(cd.test, cd.test_fg, cd.pool[sup.normal_idx],
                                                    cd.pool_fg[sup.normal_idx], cd.grid,
                                                    positional.get("radius", 2))
                    term = positional_term(gap, sgap, positional["quantile"], positional["weight"])
                    del gap
                for name, model in methods.items():
                    if name in DEFECT_METHODS and defect is None:
                        continue
                    bank = defect
                    if name == "random_control" and defect is not None:
                        bank = random_control_bank(normal, defect.shape[0], rng)
                    scores = score_support(model, cd, normal, bank)
                    variants = [(name, scores)]
                    if term is not None and name != "random_control":
                        variants.append((f"{name}+pos", scores + term))
                    for vname, vscores in variants:
                        for subset, res in ev.evaluate(vscores, subsets, pixel=pixel).items():
                            rows.append({**(extra or {}), "category": cd.name, "k0": k0, "kd": kd,
                                         "seen_type": t or "-", "seed": r, "method": vname,
                                         "subset": subset, **res})
    return rows


def checkpoint_path(cfg: dict, fold: int, method: str) -> str:
    d = os.path.join(cfg["results_dir"], "checkpoints")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"fold{fold}_{method}.pt")


def load_trained(cfg: dict, fold: int, method: str, dim: int, device: str) -> FSModel | None:
    path = checkpoint_path(cfg, fold, method)
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    model = make_model(method, dim, cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def load_json(cfg: dict, fold: int, method: str) -> dict | None:
    path = checkpoint_path(cfg, fold, method).replace(".pt", ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_positional(cfg: dict, fold: int) -> dict | None:
    """The fold's positional add-on settings, chosen on its training categories."""
    p = load_json(cfg, fold, "positional")
    if p is None:
        return None
    return {"quantile": p["quantile"], "weight": p["weight"], "radius": cfg.get("pos_radius", 2)}


def load_fixed_bank(cfg: dict, fold: int, dim: int, device: str):
    path = checkpoint_path(cfg, fold, "fixed_bank").replace(".pt", ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        p = json.load(f)
    return make_model("fixed_bank", dim, cfg, lam=p["lam"], tau_d=p["tau_d"]).to(device).eval()
