"""Category-level paired statistics (spec Section 11).

The independent units are categories: average each method over seen types and seeds within a
category, then compare methods with a Wilcoxon signed-rank test over the categories, a bootstrap
interval for the mean difference (resampling categories), and Holm correction across comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def category_table(df: pd.DataFrame, metric: str, k0: int, kd: int, subset: str = "all") -> pd.DataFrame:
    """Rows = categories, columns = methods, values = metric averaged over seen types and seeds."""
    d = df[(df.k0 == k0) & (df.kd == kd) & (df.subset == subset)]
    return d.groupby(["category", "method"])[metric].mean().unstack("method")


def compare(table: pd.DataFrame, method: str, ref: str, n_boot: int = 10000, seed: int = 0) -> dict:
    pair = table[[method, ref]].dropna()
    delta = (pair[method] - pair[ref]).to_numpy()
    n = len(delta)
    out = {"method": method, "ref": ref, "n_categories": n}
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    boots = delta[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    out.update(mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
               ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)),
               wins=int((delta > 0).sum()), losses=int((delta < 0).sum()))
    if n >= 2 and np.any(delta != 0):
        out["p_value"] = float(wilcoxon(delta, alternative="two-sided").pvalue)
    else:
        out["p_value"] = 1.0
    return out


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
