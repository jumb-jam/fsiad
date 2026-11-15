"""Step 6: free diagnostics before the next training run. No training; cached features only.

Part A (CPU, seconds): print lambda, tau and tau_D from trained checkpoints, next to the fixed
bank's grid choice, to confirm whether the learned fusion ever switched the defect bank on.

Part B (GPU): normal-only scoring variants on identical supports (k0 = 2, the same support
seeds as 04_evaluate.py, so `rot` equals the evaluation's baseline at k_d = 0):
  rot           global 1-NN, 8 rotations in the bank          (current baseline)
  norot         global 1-NN, unrotated support only
  pos_r1/pos_r2 positional 1-NN: only support patches within +/-1 or +/-2 grid cells
  avg_rot_pos2  0.5 * (rot + pos_r2), a preview of fusing a positional branch
  mQ_gW         rot + W * max(0, (pos_r2 - rot) - m_Q): the positional gap counts only above a
                margin m_Q = Q-th percentile of the gaps between the support images themselves
                (leave-one-out, no labels). Rotated or jittery categories get a large margin,
                so the score falls back to the baseline on its own.
Metrics on all test images and per defect type (good images + that type), plus a
support-alignment ratio per category (near 1 = objects line up, positional matching is safe).

Variant sets (--variants): "margin" (default) = rot, pos_r2, avg_rot_pos2 and five margin
variants; "v1" = the first five above.

Usage:
  python scripts/06_diagnostics.py --ckpt-only                         # Part A only
  python scripts/06_diagnostics.py --categories transistor cable       # quick look, ~3 min
  python scripts/06_diagnostics.py                                     # all 15 categories
  python scripts/06_diagnostics.py --k0 4 --out diagnostics_k4.csv      # 4 normal shots
Writes <results_dir>/diagnostics.csv (or --out) and <results_dir>/diagnostics_params.csv.
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import torch                                                         # noqa: E402

from fsiad.cache import load_category                                # noqa: E402
from fsiad.data import load_config                                   # noqa: E402
from fsiad.episodes import CategoryData, sample_support              # noqa: E402
from fsiad.gpu_eval import CategoryEvaluator                         # noqa: E402
from fsiad.positional import (alignment_ratio, margin_scores, positional_patch_scores,  # noqa: E402
                              support_gaps)
from fsiad.runner import device_of, free_gpu, support_rng            # noqa: E402
from fsiad.scoring import baseline_patch_scores, unit                # noqa: E402

VARIANT_SETS = {
    "v1": ["rot", "norot", "pos_r1", "pos_r2", "avg_rot_pos2"],
    "margin": ["rot", "pos_r2", "avg_rot_pos2",
               "m50_g0.5", "m50_g1", "m90_g0.5", "m90_g1", "m99_g1"],
}
QUANTILES = [50, 90, 99]


# ---------------------------------------------------------------- Part A
def find_checkpoint_dir(cfg, given):
    if given:
        return given
    cands = [os.path.join(cfg["results_dir"], "checkpoints")]
    cands += sorted({os.path.dirname(p) for p in
                     glob.glob("/kaggle/input/**/checkpoints/fold*_*.pt", recursive=True)})
    for c in cands:
        if glob.glob(os.path.join(c, "fold*_*.pt")):
            return c
    return None


def checkpoint_params(ckpt_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "fold*_*.pt"))):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        fold = ck.get("fold")
        fb = os.path.join(ckpt_dir, f"fold{fold}_fixed_bank.json")
        fixed = json.load(open(fb)) if os.path.exists(fb) else {}
        rows.append({
            "fold": fold, "method": ck.get("method"),
            "best_step": ck.get("best_step"), "best_val": ck.get("best_val"),
            "lam": float(sd["lam"]), "tau": float(sd["log_tau"].exp()),
            "tau_d": float(sd["log_tau_d"].exp()), "tau0_init": ck.get("tau0"),
            "fixed_lam": fixed.get("lam"), "fixed_tau_d": fixed.get("tau_d"),
        })
    return pd.DataFrame(rows)


def part_a(cfg, ckpt_dir):
    ckpt_dir = find_checkpoint_dir(cfg, ckpt_dir)
    if ckpt_dir is None:
        print("Part A: no checkpoints found (pass --ckpt-dir). Skipping.\n")
        return
    df = checkpoint_params(ckpt_dir)
    if df.empty:
        print(f"Part A: no fold*_*.pt files in {ckpt_dir}. Skipping.\n")
        return
    os.makedirs(cfg["results_dir"], exist_ok=True)
    df.to_csv(os.path.join(cfg["results_dir"], "diagnostics_params.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"Part A: learned fusion parameters ({ckpt_dir})")
    print(df.round(4).to_string(index=False))
    learned = df[df.method.isin(["linear", "mlp", "fusion"])]
    if len(learned) and learned.fixed_lam.notna().any():
        def bonus(lam, td):                                        # bonus for a patch 0.1 from a defect
            return lam * np.exp(-0.1 / td)
        lb = bonus(learned.lam, learned.tau_d).median()
        fbv = bonus(learned.fixed_lam.astype(float), learned.fixed_tau_d.astype(float)).median()
        print(f"\nmedian bonus at defect distance 0.1:  learned {lb:.4f}   fixed bank {fbv:.4f}"
              f"   (ratio {lb / max(fbv, 1e-12):.3f})")
        print("ratio well below 1 -> the learned models barely used the defect bank "
              "(the warm-start fix is justified)\n")


# ---------------------------------------------------------------- Part B
def type_subsets(cd: CategoryData):
    subsets = {"all": list(range(len(cd.test_types)))}
    for t in cd.defect_types:
        subsets[f"type:{t}"] = cd.good_idx + cd.type_idx[t]
    return subsets


@torch.no_grad()
def variant_scores(cd: CategoryData, normal_idx, variants, margins):
    """Patch scores (N, P) for each requested variant. margins: {quantile: margin value}."""
    ref = cd.pool[normal_idx]                                      # (k0, R, P, D)
    d = ref.shape[-1]
    out = {}
    rot = baseline_patch_scores(cd.test, cd.test_fg, unit(ref.reshape(-1, d)))
    p2 = positional_patch_scores(cd.test, cd.test_fg, ref[:, 0], cd.grid, 2)
    for v in variants:
        if v == "rot":
            out[v] = rot
        elif v == "norot":
            out[v] = baseline_patch_scores(cd.test, cd.test_fg, unit(ref[:, 0].reshape(-1, d)))
        elif v == "pos_r1":
            out[v] = positional_patch_scores(cd.test, cd.test_fg, ref[:, 0], cd.grid, 1)
        elif v == "pos_r2":
            out[v] = p2
        elif v == "avg_rot_pos2":
            out[v] = 0.5 * (rot + p2)
        elif v.startswith("m"):
            q, w = v[1:].split("_g")
            out[v] = margin_scores(rot, p2, margins[int(q)], float(w))
        else:
            raise ValueError(f"unknown variant {v}")
    return out


def part_b(cfg, cats, seeds, k0, pixel, variants, out_name):
    device = device_of(cfg)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], out_name)
    if os.path.exists(out):
        os.remove(out)
    for name in cats:
        t0 = time.time()
        c = load_category(cfg["cache_dir"], name)
        pool_fg = c["train_fg"][:c["pool_feats"].shape[0]].to(device)   # pool image j = train image j
        cd = CategoryData(c, device, load_train=False)
        del c
        ev = CategoryEvaluator(cd, cfg["data_root"], cfg)
        subsets = type_subsets(cd)
        rows = []
        for r in seeds:
            sup = sample_support(cd, support_rng(r, cd.name, None, k0, 0), k0, 0, None)
            align = alignment_ratio(cd.pool[sup.normal_idx, 0], pool_fg[sup.normal_idx], cd.grid, 1)
            gaps = support_gaps(cd.pool[sup.normal_idx], pool_fg[sup.normal_idx], cd.grid, 2)
            if gaps.numel():
                qs = torch.tensor([q / 100 for q in QUANTILES], device=gaps.device)
                margins = dict(zip(QUANTILES, torch.quantile(gaps.float(), qs).tolist()))
            else:                                              # no foreground: positional term off
                margins = {q: float("inf") for q in QUANTILES}
            extra = {f"margin_q{q}": m for q, m in margins.items()}
            for v, scores in variant_scores(cd, sup.normal_idx, variants, margins).items():
                for subset, res in ev.evaluate(scores, subsets, pixel=pixel).items():
                    rows.append({"category": name, "k0": k0, "seed": r, "variant": v,
                                 "subset": subset, "align_ratio": align, **extra, **res})
        df = pd.DataFrame(rows)
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        metric = "px_aupro" if pixel else "img_auroc"
        tab = (df.groupby(["subset", "variant"])[metric].mean() * 100).unstack("variant")[variants]
        ms = "  ".join(f"m{q}={df[f'margin_q{q}'].mean():.3f}" for q in QUANTILES)
        print(f"\n{name}  ({time.time() - t0:.0f}s)  k0={k0}  alignment ratio {df.align_ratio.mean():.3f}"
              f"  margins {ms}\n{metric} x100, mean over {len(seeds)} seeds:")
        print(tab.round(1).to_string(), flush=True)
        del cd, ev
        free_gpu()
    summarise(out, pixel)


def summarise(path, pixel):
    df = pd.read_csv(path)
    metric = "px_aupro" if pixel else "img_auroc"
    variants = list(dict.fromkeys(df.variant))                     # order of first appearance
    others = [v for v in variants if v != "rot"]
    allr = df[df.subset == "all"]
    tab = (allr.groupby(["category", "variant"])[metric].mean() * 100).unstack("variant")[variants]
    tab["align"] = allr.groupby("category").align_ratio.mean()
    delta = tab[others].sub(tab["rot"], axis=0)
    tab = tab.sort_values("align")
    delta = delta.loc[tab.index]
    pd.set_option("display.width", 240)
    print(f"\n=== Summary: {metric} x100 on all test images, sorted by alignment ratio ===")
    print(tab.round(2).to_string())
    print("\nChange vs rot (the baseline):")
    print(delta.round(2).to_string())
    print("\nPer variant, over categories: mean change, categories up/down by > 0.1, worst change")
    summ = pd.DataFrame({"mean": delta.mean(), "up": (delta > 0.1).sum(),
                         "down": (delta < -0.1).sum(), "worst": delta.min()})
    print(summ.round(2).to_string())
    print(f"\nrows in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k0", type=int, default=2)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-only", action="store_true")
    ap.add_argument("--no-pixel", action="store_true", help="image metrics only (fast)")
    ap.add_argument("--variants", default="margin", choices=sorted(VARIANT_SETS))
    ap.add_argument("--out", default="diagnostics.csv", help="file name inside results_dir")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    part_a(cfg, args.ckpt_dir)
    if args.ckpt_only:
        return
    cats = args.categories or cfg["categories"]
    if args.k0 < 2:
        ap.error("--k0 must be >= 2 (the margin and alignment ratio compare support images)")
    part_b(cfg, cats, list(range(args.seeds)), args.k0, not args.no_pixel,
           VARIANT_SETS[args.variants], args.out)


if __name__ == "__main__":
    main()
