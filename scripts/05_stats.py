"""Step 5: category-level statistics and report tables (spec Section 11).

Usage:  python scripts/05_stats.py [--in eval.csv]
Prints and saves <results_dir>/stats.csv and <results_dir>/tables.md.
The primary comparison (fixed before running: spec 11) is reported unadjusted;
every other comparison gets a Holm-corrected p-value.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                                   # noqa: E402

from fsiad.data import load_config                                    # noqa: E402
from fsiad.stats import category_table, compare, holm                 # noqa: E402

def md_table(df, **kw):
    try:
        return df.to_markdown(**kw)
    except ImportError:                       # tabulate not installed
        return "```\n" + df.to_string(**kw) + "\n```"


METRICS = ["px_aupro", "px_auroc", "px_f1max", "px_ap", "img_auroc", "img_ap", "img_f1max"]
PAIRS = [("linear", "fixed_bank"), ("fusion", "fixed_bank"), ("fusion", "baseline"),
         ("linear", "fusion"), ("lowrank", "fixed_bank"), ("mlp", "fixed_bank"), ("linear", "baseline"),
         ("mlp", "baseline"), ("fixed_bank", "baseline"), ("random_control", "baseline"),
         ("mlp", "linear"), ("lowrank", "linear")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--in", dest="inp", default="eval.csv")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    df = pd.read_csv(os.path.join(cfg["results_dir"], args.inp))
    primary = (cfg["primary_metric"], tuple(cfg["primary_setting"]), *cfg["primary_pair"])

    results, md = [], ["# Results\n"]
    for k0, kd in sorted({(int(a), int(b)) for a, b in zip(df.k0, df.kd)}):
        for subset in ["all", "seen", "heldout"]:
            for metric in METRICS:
                tab = category_table(df, metric, k0, kd, subset)
                if tab.empty:
                    continue
                if metric == cfg["primary_metric"]:
                    md.append(f"\n## {metric}, k0={k0}, kd={kd}, subset={subset} (x100, per category)\n")
                    md.append(md_table((tab * 100).round(1)))
                    md.append("\n\nMean: " + ", ".join(f"{m} {v * 100:.1f}" for m, v in tab.mean().items()) + "\n")
                for m, ref in PAIRS:
                    if m in tab and ref in tab:
                        r = compare(tab, m, ref)
                        r.update(metric=metric, k0=k0, kd=kd, subset=subset,
                                 primary=(metric, (k0, kd), m, ref, subset) == (*primary, "all"))
                        results.append(r)

    res = pd.DataFrame(results)
    if res.empty:
        print("no comparisons possible: are the methods in eval.csv?")
        return
    sec = ~res.primary & res.p_value.notna()
    res["p_holm"] = res["p_value"]
    res.loc[sec, "p_holm"] = holm(res.loc[sec, "p_value"].to_numpy())
    cols = ["primary", "metric", "k0", "kd", "subset", "method", "ref", "n_categories", "mean_delta",
            "median_delta", "ci_low", "ci_high", "wins", "losses", "p_value", "p_holm"]
    res = res[cols].sort_values(["primary", "metric", "k0", "kd", "subset"], ascending=[False, True, True, True, True])
    res.to_csv(os.path.join(cfg["results_dir"], "stats.csv"), index=False)

    show = res[res.subset == "all"].copy()
    for c in ["mean_delta", "median_delta", "ci_low", "ci_high"]:
        show[c] = (show[c] * 100).round(2)
    pd.set_option("display.width", 220)
    print("Primary comparison:")
    print(res[res.primary].to_string(index=False))
    print("\nAll comparisons on the full test set (deltas x100):")
    print(show.to_string(index=False))
    md.append("\n## Comparisons (subset = all, deltas x100)\n")
    md.append(md_table(show, index=False))
    with open(os.path.join(cfg["results_dir"], "tables.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote stats.csv and tables.md to {cfg['results_dir']}")


if __name__ == "__main__":
    main()
