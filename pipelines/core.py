"""Core API for single-image collagen metric extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..backends import FijiBackend
from ..metrics import (
    DEFAULT_PREPROCESS_CONFIG,
    compute_python_metrics,
    load_grayscale_image,
    preprocess_image,
    segment_fibres,
)


def compute_collagen_metrics(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    preprocess: bool = True,
    preprocess_config: dict[str, Any] | None = None,
    run_twombli: bool = True,
    run_orientationj: bool = True,
    compute_custom: bool = True,
) -> dict[str, Any]:
    """
    Compute reproducible collagen metrics for a single image.

    Fiji-dependent metrics are optional. When OrientationJ is unavailable,
    coherency and orientation fall back to the Python tensor implementation.
    """
    image_path = Path(image_path)
    cfg = {**DEFAULT_PREPROCESS_CONFIG, **(preprocess_config or {})}
    raw_image = load_grayscale_image(image_path)
    working_image = preprocess_image(raw_image, cfg) if preprocess else raw_image.astype(np.float32)
    mask = segment_fibres(working_image, cfg)

    result: dict[str, Any] = {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "image_height_px": int(raw_image.shape[0]),
        "image_width_px": int(raw_image.shape[1]),
        "preprocessing_applied": bool(preprocess),
    }

    pixel_size_um = float(cfg.get("pixel_size_um", 1.0))
    backend = FijiBackend(
        fiji_path=cfg.get("fiji_path"),
        mode=str(cfg.get("fiji_mode", "headless")),
        prefer_orientationj=bool(cfg.get("prefer_orientationj", False)),
    )

    if run_orientationj:
        result.update(backend.compute_orientation_metrics(working_image))
    else:
        result["orientationj_backend"] = "disabled"
        result["orientationj_orientation_deg"] = np.nan
        result["orientationj_coherency"] = np.nan

    if run_twombli:
        result.update(backend.compute_twombli_metrics(str(image_path)))
    else:
        result["twombli_backend"] = "disabled"

    if compute_custom or run_twombli:
        result.update(compute_python_metrics(mask, pixel_size_um=pixel_size_um))

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return result
