"""Step 3: train one method on one fold.

  --method linear | mlp | lowrank   episodic meta-training of adapter + learned fusion (spec 9.1)
  --method fusion                   learned lambda / tau / tau_D only, adapter frozen at identity
  --method fixed_bank               grid search of (lambda, tau_D) on training categories (spec 5.3)

Usage:
  python scripts/03_train.py --fold 0 --method fixed_bank
  python scripts/03_train.py --fold 0 --method linear
  python scripts/03_train.py --fold 0 --method linear --smoke      # 20 steps, quick bug check
  add --set cache_dir=/kaggle/input/... to read a cache from an earlier run's output
Writes <results_dir>/checkpoints/fold{F}_{method}.pt (.json for fixed_bank) and a training log.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import torch                                                          # noqa: E402

from fsiad.data import load_config                                    # noqa: E402
from fsiad.episodes import build_banks, sample_episode, sample_support  # noqa: E402
from fsiad.folds import fold_split                                    # noqa: E402
from fsiad.gpu_eval import CategoryEvaluator                          # noqa: E402
from fsiad.losses import image_pair_loss, image_scores, patch_pair_loss  # noqa: E402
from fsiad.model import make_model                                    # noqa: E402
from fsiad.runner import (checkpoint_path, device_of, evaluate_methods, free_gpu, load_cd,  # noqa: E402
                          support_rng, test_subsets)


# ---------------------------------------------------------------- fixed bank (spec 5.3)
def fit_fixed_bank(cfg, fold, split, device, smoke):
    lams, tds = cfg["fixed_lambdas"], cfg["fixed_tau_ds"]
    k0, kd = cfg["primary_setting"]
    cats = split["train"][:2] if smoke else split["train"]
    scores = {(l, t): [] for l in lams for t in tds}
    base = make_model("fixed_bank", 1, cfg).to(device).eval()      # identity adapter: dim unused
    for name in cats:
        t0 = time.time()
        cd = load_cd(name, cfg, device)
        ev = CategoryEvaluator(cd, cfg["data_root"], cfg)
        for t in cd.defect_types[:1 if smoke else cfg["fit_max_types"]]:
            rng = support_rng(0, name, t, k0, kd)
            sup = sample_support(cd, rng, k0, kd, t)
            normal, defect = build_banks(cd, sup)
            subsets = {"all": test_subsets(cd, sup, t)["all"]}
            with torch.no_grad():
                dn, dd = base.distance_maps(cd.test, normal, defect)
                for lam, td in itertools.product(lams, tds):
                    s = base.combine(dn, dd, cd.test_fg, lam=lam, tau_d=td)
                    scores[(lam, td)].append(ev.evaluate(s, subsets)["all"]["px_aupro"])
        print(f"  {name:12s} grid done in {time.time() - t0:5.1f}s", flush=True)
        del cd, ev
        free_gpu()
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    (lam, td), best = max(mean.items(), key=lambda kv: kv[1])
    base_score = mean[(0.0, tds[0])] if 0.0 in lams else float("nan")
    out = {"lam": lam, "tau_d": td, "train_aupro": best, "train_aupro_lambda0": base_score,
           "grid": {f"{l},{t}": v for (l, t), v in mean.items()}}
    path = checkpoint_path(cfg, fold, "fixed_bank").replace(".pt", ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"best lambda={lam}, tau_D={td}: train AU-PRO {best:.4f} (lambda=0: {base_score:.4f})")
    print(f"wrote {path}")


# ---------------------------------------------------------------- adapter training (spec 7-9)
@torch.no_grad()
def calibrate(cds, cfg, rng, device, dim):
    """tau0 = median(d_(k) - d_(1)); a = 1 / median pair gap at init (spec Section 12)."""
    probe = make_model("linear", dim, cfg).to(device)      # W = 0: identical to identity
    gaps, margins = [], []
    for _ in range(cfg["calib_episodes"]):
        cd = cds[rng.integers(len(cds))]
        ep = sample_episode(cd, rng, cfg)
        bank = probe.embed(ep["normal_bank"])
        for i in range(ep["q"].shape[0]):
            d = probe.bank_distances(probe.embed(ep["q"][i]), bank)
            gaps.append((d[:, -1] - d[:, 0])[ep["fg"][i]].cpu())
    tau0 = max(float(torch.cat(gaps).median()), 1e-3)
    probe.log_tau.fill_(np.log(tau0))
    for _ in range(cfg["calib_episodes"]):
        cd = cds[rng.integers(len(cds))]
        ep = sample_episode(cd, rng, cfg)
        s = probe.patch_scores(ep["q"], ep["fg"], ep["normal_bank"], ep["defect_bank"])
        pos = s[(ep["labels"] == 1) & ep["fg"]]
        neg = s[(ep["labels"] == 0) & ep["fg"]]
        if pos.numel() and neg.numel():
            j = torch.randint(neg.numel(), (min(pos.numel(), 512),), device=s.device)
            margins.append((pos[:len(j)] - neg[j]).abs().cpu())
    gap = float(torch.cat(margins).median()) if margins else 0.05
    a = 1.0 / max(gap, 1e-4)
    return tau0, a


def validate(model, val_cds, val_evs, cfg, smoke):
    model.eval()
    rows = []
    for cd, ev in zip(val_cds, val_evs):
        rows += evaluate_methods({"m": model}, cd, ev, [tuple(cfg["primary_setting"])], [0],
                                 max_types=1 if smoke else cfg["val_max_types"])
    model.train()
    free_gpu()
    if not rows:
        return float("nan")
    per_cat = pd.DataFrame(rows).query("subset == 'all'").groupby("category")["px_aupro"].mean()
    return float(per_cat.mean())


def train_adapter(cfg, fold, method, split, device, smoke):
    rng = np.random.default_rng([cfg["train_seed"], fold])
    torch.manual_seed(cfg["train_seed"] + fold)
    train_names = split["train"][:3] if smoke else split["train"]
    print("loading", train_names, "+ val", split["val"], flush=True)
    cds = [load_cd(n, cfg, device) for n in train_names]
    val_cds = [load_cd(n, cfg, device) for n in split["val"]]
    val_evs = [CategoryEvaluator(cd, cfg["data_root"], cfg) for cd in val_cds]
    n_probe = cfg.get("probe_train_categories", 0)
    probe_cds = cds[:n_probe]
    probe_evs = [CategoryEvaluator(cd, cfg["data_root"], cfg) for cd in probe_cds]
    dim = cds[0].dim

    tau0, a = calibrate(cds, cfg, rng, device, dim)
    print(f"calibration: tau0={tau0:.4f}  a={a:.2f}", flush=True)
    model = make_model(method, dim, cfg, tau=tau0, lam=0.0, tau_d=tau0).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    steps = 20 if smoke else cfg["steps"]
    eval_every = 10 if smoke else cfg["eval_every"]
    val0 = validate(model, val_cds, val_evs, cfg, smoke)
    tr0 = validate(model, probe_cds, probe_evs, cfg, smoke) if probe_cds else float("nan")
    print(f"step     0  val AU-PRO {val0:.4f}  train-cat AU-PRO {tr0:.4f}  (= frozen-feature start)",
          flush=True)
    best, best_step, log = val0, 0, []
    ck_path = checkpoint_path(cfg, fold, method)
    meta = {"method": method, "fold": fold, "tau0": tau0, "a": a, "split": split}
    torch.save({"state_dict": model.state_dict(), **meta, "best_val": best, "best_step": 0}, ck_path)
    gen = torch.Generator(device=device).manual_seed(cfg["train_seed"] + fold)

    t0 = time.time()
    for step in range(1, steps + 1):
        lp_sum = li_sum = 0.0
        for _ in range(cfg["episodes_per_step"]):
            cd = cds[rng.integers(len(cds))]
            ep = sample_episode(cd, rng, cfg)
            s = model.patch_scores(ep["q"], ep["fg"], ep["normal_bank"], ep["defect_bank"])
            lp = patch_pair_loss(s, ep["labels"], ep["fg"], a, cfg["neg_per_pos"], gen)
            li = image_pair_loss(image_scores(s, cfg["top_fraction"]), ep["img_labels"], a)
            ((lp + cfg["beta"] * li) / cfg["episodes_per_step"]).backward()
            lp_sum += float(lp.detach())
            li_sum += float(li.detach())
        reg = cfg["eta"] * model.reg()
        if reg.requires_grad:
            reg.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.project()

        if step % cfg["log_every"] == 0 or smoke:
            e = cfg["episodes_per_step"]
            log.append({"step": step, "loss_pix": lp_sum / e, "loss_img": li_sum / e,
                        "lam": float(model.lam), "tau": float(model.tau), "tau_d": float(model.tau_d),
                        "reg": float(model.reg().detach())})
            r = log[-1]
            print(f"step {step:5d}  pix {r['loss_pix']:.4f}  img {r['loss_img']:.4f}  "
                  f"lam {r['lam']:.4f}  tau {r['tau']:.4f}  tau_D {r['tau_d']:.4f}  "
                  f"|W|^2 {r['reg']:.4f}  {time.time() - t0:6.0f}s", flush=True)
        if step % eval_every == 0:
            val = validate(model, val_cds, val_evs, cfg, smoke)
            tr = validate(model, probe_cds, probe_evs, cfg, smoke) if probe_cds else float("nan")
            log.append({"step": step, "val_aupro": val, "train_cat_aupro": tr})
            flag = ""
            if val > best:
                best, best_step, flag = val, step, "  * best"
                torch.save({"state_dict": model.state_dict(), **meta, "best_val": best,
                            "best_step": step}, ck_path)
            print(f"step {step:5d}  val AU-PRO {val:.4f} ({val - val0:+.4f})  "
                  f"train-cat AU-PRO {tr:.4f} ({tr - tr0:+.4f}){flag}", flush=True)
            if step - best_step >= cfg["patience"]:
                print(f"early stop: no improvement for {cfg['patience']} steps")
                break

    pd.DataFrame(log).to_csv(ck_path.replace(".pt", "_log.csv"), index=False)
    print(f"best val AU-PRO {best:.4f} at step {best_step} (start {val0:.4f}); saved {ck_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--method", required=True, choices=["fusion", "linear", "mlp", "lowrank", "fixed_bank"])
    ap.add_argument("--smoke", action="store_true", help="tiny run to check for bugs")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = device_of(cfg)
    split = fold_split(args.fold, cfg["fold_seed"])
    print(f"fold {args.fold}: test {split['test']}  val {split['val']}")
    if args.method == "fixed_bank":
        fit_fixed_bank(cfg, args.fold, split, device, args.smoke)
    else:
        train_adapter(cfg, args.fold, args.method, split, device, args.smoke)


if __name__ == "__main__":
    main()
