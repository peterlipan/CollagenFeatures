# CollagenFeatures

CollagenFeatures computes reproducible collagen-image metrics from TIFF images. The package exposes a simple Python API for single images and batches, and it can run the official Fiji plugins for OrientationJ and TWOMBLI when their jars are installed locally.

## What it does

- Computes one consistent row of metrics per image.
- Saves batch outputs as CSV with image names and source paths.
- Uses official Fiji plugin execution for:
  - OrientationJ orientation and coherency
  - TWOMBLI morphology metrics
- Falls back to deterministic Python implementations if the Fiji runtime is unavailable.

## Main API

```python
from CollagenFeatures import compute_collagen_metrics

metrics = compute_collagen_metrics("example.tif")
```

```python
from pathlib import Path
from CollagenFeatures import batch_compute_collagen_metrics

image_paths = sorted(Path("images").glob("*.tif"))
df = batch_compute_collagen_metrics(image_paths, "results/metrics.csv")
```

The package also includes a dataset-specific entry point:

```bash
python -m CollagenFeatures.main --raw-root /datastorage/li/CollagenRawImages
```

That workflow scans only folders ending with `_R` and processes the TIFF files inside them.

## Metric columns

The single-image pipeline returns stable column names for:

- image identifiers and dimensions
- `orientationj_orientation_deg`
- `orientationj_coherency`
- `mask_area_fraction`
- `mask_area_um2`
- `fibre_length_um`
- `lacunarity`
- `endpoints`
- `branch_points`
- `hyphal_growth_unit_um`
- `curvature_30um_deg`
- `curvature_40um_deg`
- `fractal_dimension_boxcount`
- `high_density_matrix_fraction`
- `twombli_alignment`
- `fibre_width_um`
- `length_tortuosity`

It also records backend provenance in `orientationj_backend` and `twombli_backend`.

## Installation

Install the Python dependencies and download the Fiji plugin jars:

```bash
./scripts/setup_environment.sh
```

Or run the two steps separately:

```bash
python -m pip install -r requirements.txt
python scripts/install_fiji_plugins.py
```

The installer places the required jars under `third_party/twombli/`. Those jars are added to the JVM classpath automatically by the Fiji backend.

## Project layout

- [`metrics/`](./metrics): preprocessing, segmentation, Python surrogate metrics, orientation fallback
- [`backends/`](./backends): Fiji/ImageJ integration, including direct OrientationJ and TWOMBLI execution
- [`pipelines/`](./pipelines): single-image, batch, and red-channel workflows
- [`datasets/`](./datasets): comparison helpers for official exported results
- [`utils/`](./utils): small shared utilities
- [`main.py`](./main.py): main batch entry point for the red-channel dataset
- [`compare_official_results.py`](./compare_official_results.py): script for checking alignment against official exports

## Notes

- Fiji execution is optional, but preferred when you want alignment with the official plugin implementations.
- If TWOMBLI completes partially in headless mode, the package keeps the official written outputs and augments them with OrientationJ-based alignment from the produced mask image.
- Generated CSV files and downloaded jars are treated as outputs and are ignored by git.
