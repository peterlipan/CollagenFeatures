"""Small naming and path normalization helpers."""

from __future__ import annotations

from pathlib import Path


def normalize_image_name(value: object) -> str:
    """Normalize exported image names to raw-image TIFF basenames."""
    text = str(value).strip().strip('"').strip("'")
    if text.lower().endswith(".png"):
        text = f"{Path(text).stem}.tif"
    return Path(text).name
