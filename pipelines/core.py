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
        twombli_result = backend.compute_twombli_metrics(
            image=working_image,
            image_name=image_path.stem,
            output_dir=output_dir,
            config=cfg,
        )
        if twombli_result is None:
            result["twombli_backend"] = "python_surrogate"
            surrogate_metrics = compute_python_metrics(mask, pixel_size_um=pixel_size_um)
            for key in (
                "mask_area_fraction",
                "fibre_length_um",
                "lacunarity",
                "endpoints",
                "branch_points",
                "hyphal_growth_unit_um",
                "curvature_30um_deg",
                "curvature_40um_deg",
                "fractal_dimension_boxcount",
            ):
                result[key] = surrogate_metrics.get(key)
        else:
            result.update(twombli_result)
    else:
        result["twombli_backend"] = "disabled"

    if run_twombli and "mask_area_um2" in result and "mask_area_fraction" not in result:
        total_area = raw_image.shape[0] * raw_image.shape[1] * pixel_size_um * pixel_size_um
        result["mask_area_fraction"] = float(result["mask_area_um2"] / total_area) if total_area > 0 else np.nan

    if compute_custom:
        python_metrics = compute_python_metrics(mask, pixel_size_um=pixel_size_um)
        if run_twombli:
            for key in ("fibre_width_um", "length_tortuosity", "skeleton_pixels", "connected_components"):
                result[key] = python_metrics.get(key)
        else:
            result.update(python_metrics)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return result
