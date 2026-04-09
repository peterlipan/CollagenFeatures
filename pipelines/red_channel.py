"""Red-channel batch workflow for the local dataset layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .batch import batch_compute_collagen_metrics


def collect_red_channel_images(root: Path) -> list[Path]:
    """Collect TIFF images from directories whose names end with `_R`."""
    image_paths = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_R")):
        image_paths.extend(sorted(folder.glob("*.tif")))
    return image_paths


def build_red_channel_metrics(
    raw_root: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Compute metrics directly for red-channel images."""
    image_paths = collect_red_channel_images(raw_root)
    df = batch_compute_collagen_metrics(
        image_paths=image_paths,
        output_csv=output_csv,
        preprocess_config={
            "equalization": "histogram",
            "threshold": "otsu",
            "minimum_line_width": 5,
            "maximum_line_width": 10,
            "minimum_branch_length": 5,
            "minimum_curvature_window": 30,
            "curvature_window_step_size": 10,
            "maximum_curvature_window": 40,
            "maximum_display_hdm": 200,
            "contrast_saturation": 0.35,
            "perform_gap_analysis": True,
            "minimum_gap_diameter": 40,
        },
    )
    return df
