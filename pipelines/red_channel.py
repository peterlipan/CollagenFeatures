"""Red-channel batch workflow for the local dataset layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..datasets.official_metrics import finalize_metric_columns, load_official_metric_lookup
from .core import compute_collagen_metrics


def collect_red_channel_images(root: Path) -> list[Path]:
    """Collect TIFF images from directories whose names end with `_R`."""
    image_paths = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_R")):
        image_paths.extend(sorted(folder.glob("*.tif")))
    return image_paths


def build_red_channel_metrics(
    raw_root: Path,
    segmented_root: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Compute red-channel OrientationJ metrics and merge official morphology exports."""
    image_paths = collect_red_channel_images(raw_root)
    official_lookup = load_official_metric_lookup(segmented_root)

    orientation_rows = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        metrics = compute_collagen_metrics(
            image_path=image_path,
            preprocess=True,
            preprocess_config={"equalization": "histogram", "threshold": "otsu"},
            run_twombli=False,
            compute_custom=False,
        )
        orientation_rows.append(
            {
                "image_name": metrics["image_name"],
                "image_path": metrics["image_path"],
                "folder_name": image_path.parent.name,
                "image_height_px": metrics["image_height_px"],
                "image_width_px": metrics["image_width_px"],
                "orientationj_backend": metrics["orientationj_backend"],
                "orientationj_orientation_deg": metrics["orientationj_orientation_deg"],
                "orientationj_coherency": metrics["orientationj_coherency"],
            }
        )
        if index % 100 == 0 or index == total:
            print(f"Processed OrientationJ for {index}/{total} red-channel images")

    orientation_df = pd.DataFrame(orientation_rows)
    merged = orientation_df.merge(official_lookup, on="image_name", how="left")
    merged = finalize_metric_columns(merged)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged = merged.sort_values(["folder_name", "image_name"]).reset_index(drop=True)
    merged.to_csv(output_csv, index=False)
    return merged
