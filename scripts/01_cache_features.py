"""Step 1: run DINOv2 once over MVTec AD and cache features, masks and patch labels.

Usage:  python scripts/01_cache_features.py --config configs/default.yaml [--categories bottle screw]
Re-running skips categories that are already cached.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsiad.backbone import DINOv2Extractor          # noqa: E402
from fsiad.cache import cache_category              # noqa: E402
from fsiad.data import index_category, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cats = args.categories or cfg["categories"]

    extractor = DINOv2Extractor(cfg["model_name"], cfg["resolution"], cfg["device"])
    for c in cats:
        t0 = time.time()
        idx = index_category(cfg["data_root"], c, cfg["pool_size"])
        path = cache_category(extractor, idx, cfg)
        size_mb = os.path.getsize(path) / 2**20
        print(f"{c:12s} train={len(idx.train):4d} test={len(idx.test):4d} "
              f"types={len(idx.defect_types)}  {size_mb:7.1f} MB  {time.time() - t0:6.1f}s")


if __name__ == "__main__":
    main()
