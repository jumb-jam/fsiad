# FS-IAD: Meta-Learned Feature Adapter for Few-Shot Industrial Anomaly Detection

Week-1 code: feature cache, AnomalyDINO baseline reproduction, metrics and sanity tests.
The math and every design choice are in the project's math spec; section numbers below refer to it.

## Layout

```
configs/default.yaml        all settings (paths set for Kaggle)
fsiad/
  data.py                   MVTec index, image/mask loading, patch labels      (spec 1)
  preprocess.py             rotation + PCA background mask, copied from AnomalyDINO (2.5)
  backbone.py               DINOv2 ViT-S/14, 448 px, last-layer patch tokens  (2.5)
  cache.py                  run DINOv2 once, save features per category      (10)
  scoring.py                cosine kNN, soft-min, fusion, image score, maps   (2, 4, 5, 6)
  metrics.py                image AUROC/AP/F1-max, pixel AU-PRO/AUROC/F1-max/AP (11)
scripts/
  01_cache_features.py      step 1
  02_reproduce_baseline.py  step 2
tests/                      sanity checks (spec 13)
```

Coming next: defect bank baseline, adapters, episodes, losses, training, evaluation, statistics.

## Running on Kaggle

1. Notebook settings: **Accelerator = GPU** (T4 or P100), **Internet = On**.
2. Get the dataset path (already verified: 15 categories, 3,629 train, 1,725 test, 1,258 masks):

```python
import kagglehub
print(kagglehub.dataset_download("ipythonx/mvtec-ad"))   # -> /kaggle/input/datasets/ipythonx/mvtec-ad
```

If the printed path differs, change `data_root` in `configs/default.yaml`.

3. Get the code (replace with your repo URL):

```python
!git clone https://github.com/<your-username>/fsiad.git
%cd fsiad
```

4. Run the sanity tests:

```python
!python -m pytest tests -q
```

5. Cache features (DINOv2 runs once; about 5.5 GB written to `/kaggle/working/cache`):

```python
!python scripts/01_cache_features.py
```

Try one category first to check timing: `--categories bottle`.

6. Reproduce the baseline. Quick image-level pass first, then the full run with pixel metrics:

```python
!python scripts/02_reproduce_baseline.py --no-pixel
!python scripts/02_reproduce_baseline.py
```

Results go to `/kaggle/working/results/baseline_repro.csv`, and a summary table
(mean over categories, then mean / std over seeds 0-2) is printed. Compare it with the
AnomalyDINO paper's MVTec AD results for ViT-S at 448 px, agnostic preprocessing.

Files in `/kaggle/working` disappear when an interactive session ends. If that happens,
re-run step 5 (it skips categories that are already cached).

## Faithfulness notes

- Preprocessing, features, rotation (8 angles), masking (capsule, hazelnut, pill, screw,
  toothbrush; test images only), 1-NN cosine distance, top-1% image score, bilinear + sigma 4
  maps, reference selection per seed and all metric code follow the AnomalyDINO repository.
- AnomalyDINO dilates/closes its background mask on the flattened (P, 1) array, not the
  2-D grid. We keep that quirk on purpose so numbers match; a test checks equality.
- Differences: features are cached in float16 (distances are computed in float32), and the
  mask PCA uses a fixed random seed. Both should change metrics only in the 3rd-4th decimal.

## Running locally or on Colab

Install `requirements.txt`, set `data_root`, `cache_dir`, `results_dir` and `device` in the config.
