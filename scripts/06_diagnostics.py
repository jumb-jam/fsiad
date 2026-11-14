"""Step 6: free diagnostics before the next training run. No training; cached features only.

Part A (CPU, seconds): print lambda, tau and tau_D from trained checkpoints, next to the fixed
bank's grid choice, to confirm whether the learned fusion ever switched the defect bank on.

Part B (GPU): normal-only scoring variants on identical supports (k0 = 2, the same support
seeds as 04_evaluate.py, so `rot` equals the evaluation's baseline at k_d = 0):
  rot           global 1-NN, 8 rotations in the bank          (current baseline)
  norot         global 1-NN, unrotated support only
  pos_r1/pos_r2 positional 1-NN: only support patches within +/-1 or +/-2 grid cells
  avg_rot_pos2  0.5 * (rot + pos_r2), a preview of fusing a positional branch
Metrics on all test images and per defect type (good images + that type), plus a
support-alignment ratio per category (near 1 = objects line up, positional matching is safe).

Usage:
  python scripts/06_diagnostics.py --ckpt-only                         # Part A only
  python scripts/06_diagnostics.py --categories transistor cable       # quick look, ~3 min
  python scripts/06_diagnostics.py                                     # all 15 categories
Writes <results_dir>/diagnostics.csv and <results_dir>/diagnostics_params.csv.
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
from fsiad.positional import alignment_ratio, positional_patch_scores  # noqa: E402
from fsiad.runner import device_of, free_gpu, support_rng            # noqa: E402
from fsiad.scoring import baseline_patch_scores, unit                # noqa: E402

VARIANTS = ["rot", "norot", "pos_r1", "pos_r2", "avg_rot_pos2"]


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
def variant_scores(cd: CategoryData, normal_idx):
    ref = cd.pool[normal_idx]                                      # (k0, R, P, D)
    rot = baseline_patch_scores(cd.test, cd.test_fg, unit(ref.reshape(-1, ref.shape[-1])))
    norot = baseline_patch_scores(cd.test, cd.test_fg, unit(ref[:, 0].reshape(-1, ref.shape[-1])))
    p1 = positional_patch_scores(cd.test, cd.test_fg, ref[:, 0], cd.grid, 1)
    p2 = positional_patch_scores(cd.test, cd.test_fg, ref[:, 0], cd.grid, 2)
    return {"rot": rot, "norot": norot, "pos_r1": p1, "pos_r2": p2, "avg_rot_pos2": 0.5 * (rot + p2)}


def part_b(cfg, cats, seeds, k0, pixel):
    device = device_of(cfg)
    out = os.path.join(cfg["results_dir"], "diagnostics.csv")
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
            for v, scores in variant_scores(cd, sup.normal_idx).items():
                for subset, res in ev.evaluate(scores, subsets, pixel=pixel).items():
                    rows.append({"category": name, "k0": k0, "seed": r, "variant": v,
                                 "subset": subset, "align_ratio": align, **res})
        df = pd.DataFrame(rows)
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        metric = "px_aupro" if pixel else "img_auroc"
        tab = (df.groupby(["subset", "variant"])[metric].mean() * 100).unstack("variant")[VARIANTS]
        print(f"\n{name}  ({time.time() - t0:.0f}s)  alignment ratio {df.align_ratio.mean():.3f}"
              f"   {metric} x100, mean over {len(seeds)} seeds:")
        print(tab.round(1).to_string(), flush=True)
        del cd, ev
        free_gpu()
    summarise(out, pixel)


def summarise(path, pixel):
    df = pd.read_csv(path)
    metric = "px_aupro" if pixel else "img_auroc"
    allr = df[df.subset == "all"]
    tab = (allr.groupby(["category", "variant"])[metric].mean() * 100).unstack("variant")[VARIANTS]
    tab["align"] = allr.groupby("category").align_ratio.mean()
    for v in VARIANTS[1:]:
        tab[f"d_{v}"] = tab[v] - tab["rot"]
    tab = tab.sort_values("align")
    pd.set_option("display.width", 220)
    print(f"\n=== Summary: {metric} x100 on all test images, sorted by alignment ratio ===")
    print(tab.round(2).to_string())
    print("\nMean over categories:")
    print(tab[VARIANTS].mean().round(2).to_string())
    if len(tab) >= 4:
        rho = tab["align"].corr(tab["d_pos_r2"], method="spearman")
        print(f"\nSpearman(alignment ratio, pos_r2 - rot) = {rho:.2f}  "
              "(clearly negative -> the ratio can gate a positional branch)")
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
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    part_a(cfg, args.ckpt_dir)
    if args.ckpt_only:
        return
    cats = args.categories or cfg["categories"]
    part_b(cfg, cats, list(range(args.seeds)), args.k0, not args.no_pixel)


if __name__ == "__main__":
    main()
