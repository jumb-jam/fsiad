"""Step 4: evaluate every method on each fold's unseen test categories (spec 9.2).

For each test category, shot setting, seen defect type and seed, all methods are scored on the
SAME support set (paired comparison). Methods: baseline (normal-only AnomalyDINO), fixed_bank,
random_control, and every trained adapter found in <results_dir>/checkpoints.

Usage:
  python scripts/04_evaluate.py --folds 0                 # one fold
  python scripts/04_evaluate.py                           # all folds that have checkpoints
  python scripts/04_evaluate.py --folds 0 --smoke         # 1 category, 1 type, 1 seed
Appends rows to <results_dir>/eval.csv as each category finishes.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                                   # noqa: E402

from fsiad.data import load_config                                    # noqa: E402
from fsiad.folds import fold_split                                    # noqa: E402
from fsiad.gpu_eval import CategoryEvaluator                          # noqa: E402
from fsiad.model import make_model                                    # noqa: E402
from fsiad.runner import (device_of, evaluate_methods, free_gpu, load_cd,  # noqa: E402
                          load_fixed_bank, load_trained)

ADAPTERS = ["fusion", "linear", "mlp", "lowrank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=None, help="number of support seeds")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="eval.csv")
    ap.add_argument("--append", action="store_true", help="add to an existing results file")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = device_of(cfg)
    settings = [tuple(s) for s in cfg["eval_settings"]]
    seeds = list(range(1 if args.smoke else (args.seeds or cfg["eval_seeds"])))
    out = os.path.join(cfg["results_dir"], args.out)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    if os.path.exists(out) and not args.append:
        os.remove(out)                       # avoid silently mixing old rows into new results

    for fold in args.folds:
        split = fold_split(fold, cfg["fold_seed"])
        cats = split["test"][:1] if args.smoke else split["test"]
        for name in cats:
            t0 = time.time()
            cd = load_cd(name, cfg, device)
            methods = {"baseline": make_model("baseline", cd.dim, cfg).to(device).eval()}
            fb = load_fixed_bank(cfg, fold, cd.dim, device)
            if fb is not None:
                methods["fixed_bank"] = fb
                methods["random_control"] = fb
            for a in ADAPTERS:
                m = load_trained(cfg, fold, a, cd.dim, device)
                if m is not None:
                    methods[a] = m
            ev = CategoryEvaluator(cd, cfg["data_root"], cfg)
            rows = evaluate_methods(methods, cd, ev, settings, seeds,
                                    max_types=1 if args.smoke else None, extra={"fold": fold})
            df = pd.DataFrame(rows)
            df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            summary = (df[df.subset == "all"].groupby(["k0", "kd", "method"])["px_aupro"].mean() * 100)
            print(f"fold {fold} {name:12s} {time.time() - t0:6.0f}s  methods={list(methods)}")
            print(summary.round(1).to_string(), flush=True)
            del cd, ev, methods
            free_gpu()
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
