"""Batch helpers for collagen metric extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .core import compute_collagen_metrics


def batch_compute_collagen_metrics(
    image_paths: Iterable[str | Path],
    output_csv: str | Path,
    per_image_dir: str | Path | None = None,
    preprocess_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute metrics for many images and save a single CSV."""
    rows = []
    output_csv = Path(output_csv)
    if per_image_dir is not None:
        Path(per_image_dir).mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_path = Path(image_path)
        image_output_dir = None
        if per_image_dir is not None:
            image_output_dir = Path(per_image_dir) / image_path.stem
        rows.append(
            compute_collagen_metrics(
                image_path=image_path,
                output_dir=image_output_dir,
                preprocess=True,
                preprocess_config=preprocess_config,
                run_twombli=True,
                run_orientationj=True,
                compute_custom=True,
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty and "image_name" in df.columns:
        df = df.sort_values("image_name").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
