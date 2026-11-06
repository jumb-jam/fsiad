"""Step 2: reproduce AnomalyDINO (agnostic: rotation on, test masking on) from the feature cache.

Protocol copied from AnomalyDINO: seed s with k shots uses train images s*k ... (s+1)*k - 1
(sorted filenames), each with 8 rotations in the memory bank; 1-NN cosine distance;
image score = mean of the top 1% of patches; maps = bilinear + Gaussian(sigma=4).

Usage:
  python scripts/02_reproduce_baseline.py                       # image + pixel metrics
  python scripts/02_reproduce_baseline.py --no-pixel            # image metrics only (fast)
  python scripts/02_reproduce_baseline.py --categories bottle --shots 1
Writes <results_dir>/baseline_repro.csv and prints mean +/- std over seeds per shot.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import torch                                         # noqa: E402

from fsiad.cache import load_category                # noqa: E402
from fsiad.data import load_config, load_mask        # noqa: E402
from fsiad.metrics import image_metrics, pixel_metrics  # noqa: E402
from fsiad.scoring import anomaly_map, baseline_patch_scores, image_score, unit  # noqa: E402


def run_category(c: dict, shots, seeds, cfg, device, do_pixel: bool, rotation: bool):
    grid, labels = c["grid"], np.array([t != "good" for t in c["test_types"]], dtype=int)
    test_feats = c["test_feats"].to(device)
    test_fg = c["test_fg"].to(device)
    gts = [load_mask(m, s) for m, s in zip(c["test_masks"], c["test_shapes"])] if do_pixel else None
    rows = []
    for k in shots:
        for s in seeds:
            ref = c["pool_feats"][s * k:(s + 1) * k]                 # (k, R, P, D)
            if ref.shape[0] < k:
                raise ValueError(f"pool too small for shot {k}, seed {s}")
            if not rotation:
                ref = ref[:, :1]
            bank = unit(ref.reshape(-1, ref.shape[-1]).to(device))
            patch = baseline_patch_scores(test_feats, test_fg, bank).cpu().numpy()
            img = np.array([image_score(p, cfg["top_fraction"]) for p in patch])
            row = {"category": c["category"], "shot": k, "seed": s, **image_metrics(labels, img)}
            if do_pixel:
                maps = [anomaly_map(p, grid, hw, cfg["blur_sigma"]) for p, hw in zip(patch, c["test_shapes"])]
                row.update(pixel_metrics(maps, gts, cfg["pro_fpr_limit"], cfg["compute_pixel_ap"]))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--shots", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--no-pixel", action="store_true", help="skip pixel metrics (much faster)")
    ap.add_argument("--no-rotation", action="store_true", help="bank without rotated copies")
    ap.add_argument("--out", default="baseline_repro.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    cats = args.categories or cfg["categories"]
    shots = args.shots or cfg["repro_shots"]
    seeds = args.seeds if args.seeds is not None else cfg["repro_seeds"]

    rows = []
    for name in cats:
        t0 = time.time()
        c = load_category(cfg["cache_dir"], name)
        rows += run_category(c, shots, seeds, cfg, device, not args.no_pixel, not args.no_rotation)
        print(f"{name:12s} done in {time.time() - t0:6.1f}s")
        del c

    df = pd.DataFrame(rows)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], args.out)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    metrics = [m for m in df.columns if m.startswith(("img_", "px_"))]
    # AnomalyDINO reporting: mean over categories per seed, then mean +/- std over seeds.
    per_seed = df.groupby(["shot", "seed"])[metrics].mean()
    summary = per_seed.groupby("shot").agg(["mean", "std"]) * 100
    pd.set_option("display.width", 200)
    print("\nMean over categories, then mean / std over seeds (x100):")
    print(summary.round(1).to_string())
    print("\nPer category, averaged over seeds (x100):")
    print((df.groupby(["shot", "category"])[metrics].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
