# CollagenFeatures

Reproducible collagen-metric extraction for microscopy images, with Python orchestration and optional Fiji/ImageJ integration.

## What the package provides

- `compute_collagen_metrics(...)`
  Processes one image and returns a dictionary of collagen metrics.
- `batch_compute_collagen_metrics(...)`
  Processes many images and saves one CSV with one row per image.
- `run_red_channel_batch.py`
  Dataset-specific helper that scans `/datastorage/li/CollagenRawImages`, computes official OrientationJ metrics for folders ending in `_R`, and merges those with existing official segmented exports.
- `compare_official_results.py`
  Compares generated outputs against official exported CSV and MATLAB results.

## Core metrics

The single-image API returns stable column names for:

- image identifiers and dimensions
- OrientationJ orientation and coherency
- mask area fraction
- fibre length
- lacunarity
- endpoints
- branch points
- hyphal growth unit
- curvature at 30 um and 40 um scales
- box-counting fractal dimension
- fibre width
- length tortuosity

## Public API

```python
from CollagenFeatures import batch_compute_collagen_metrics, compute_collagen_metrics

metrics = compute_collagen_metrics("example.tif")
```

```python
from pathlib import Path
from CollagenFeatures import batch_compute_collagen_metrics

image_paths = sorted(Path("images").glob("*.tif"))
df = batch_compute_collagen_metrics(image_paths, "results/metrics.csv")
```

## Module layout

- [`metrics/`](./metrics): preprocessing, Python-native morphology metrics, and the OrientationJ fallback
- [`backends/`](./backends): external backend integrations such as Fiji/ImageJ
- [`datasets/`](./datasets): readers for official exported metric files
- [`utils/`](./utils): small shared helpers such as image-name normalization
- [`pipelines/`](./pipelines): high-level single-image, batch, and red-channel workflows
- [`main.py`](./main.py): main batch entry point for the red-channel dataset workflow
- [`compare_official_results.py`](./compare_official_results.py): comparison utility for generated versus official results

## Notes

- Fiji is optional for the library API. If OrientationJ is unavailable, the backend falls back to a deterministic Python tensor implementation.
- TWOMBLI execution is not invoked directly from Fiji in this repository. For the local dataset, official exported TWOMBLI/AnaMorf/stat outputs are merged from `/datastorage/li/Collagen/Segmented`.
- Generated CSV files are treated as outputs and are ignored by git.
