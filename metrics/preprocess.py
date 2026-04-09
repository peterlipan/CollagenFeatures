"""Image loading, preprocessing, and segmentation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


DEFAULT_PREPROCESS_CONFIG: dict[str, Any] = {
    "scale_percentiles": (1.0, 99.0),
    "denoise": None,
    "denoise_sigma": 1.0,
    "equalization": "histogram",
    "adaptive_radius": 32,
    "threshold": "otsu",
    "min_object_size": 32,
    "closing_iterations": 1,
    "opening_iterations": 0,
    "fill_holes": True,
    "invert_mask": False,
}


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as a 2D float32 grayscale array."""
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32)


def preprocess_image(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Apply intensity scaling, optional denoising, and equalization."""
    cfg = {**DEFAULT_PREPROCESS_CONFIG, **(config or {})}

    processed = intensity_scale(image, *cfg["scale_percentiles"])

    denoise_mode = cfg.get("denoise")
    if denoise_mode == "gaussian":
        processed = ndimage.gaussian_filter(processed, sigma=float(cfg.get("denoise_sigma", 1.0)))
    elif denoise_mode == "median":
        processed = ndimage.median_filter(processed, size=3)

    equalization = cfg.get("equalization")
    if equalization == "histogram":
        processed = histogram_equalize(processed)
    elif equalization == "adaptive":
        processed = adaptive_equalize(processed, radius=int(cfg.get("adaptive_radius", 32)))

    return processed.astype(np.float32, copy=False)


def segment_fibres(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Convert a preprocessed image into a clean binary collagen mask."""
    cfg = {**DEFAULT_PREPROCESS_CONFIG, **(config or {})}

    threshold_mode = cfg.get("threshold", "otsu")
    threshold = otsu_threshold(image) if threshold_mode == "otsu" else float(threshold_mode)

    mask = image >= threshold
    if cfg.get("invert_mask", False):
        mask = ~mask

    structure = ndimage.generate_binary_structure(2, 2)
    if int(cfg.get("closing_iterations", 1)) > 0:
        mask = ndimage.binary_closing(mask, structure=structure, iterations=int(cfg["closing_iterations"]))
    if int(cfg.get("opening_iterations", 0)) > 0:
        mask = ndimage.binary_opening(mask, structure=structure, iterations=int(cfg["opening_iterations"]))
    if cfg.get("fill_holes", True):
        mask = ndimage.binary_fill_holes(mask)

    min_size = int(cfg.get("min_object_size", 32))
    if min_size > 1:
        mask = remove_small_objects(mask, min_size=min_size)

    return mask.astype(bool, copy=False)


def intensity_scale(image: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    """Rescale intensities to [0, 1] using robust percentiles."""
    low, high = np.percentile(image, [lower_pct, upper_pct])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization for a [0, 1] image."""
    flat = np.clip(image.ravel(), 0.0, 1.0)
    hist, bins = np.histogram(flat, bins=256, range=(0.0, 1.0))
    cdf = hist.cumsum()
    if cdf[-1] == 0:
        return image.astype(np.float32)
    cdf = cdf / cdf[-1]
    equalized = np.interp(flat, bins[:-1], cdf)
    return equalized.reshape(image.shape).astype(np.float32)


def adaptive_equalize(image: np.ndarray, radius: int = 32) -> np.ndarray:
    """Local contrast normalization followed by global equalization."""
    radius = max(3, int(radius))
    local_mean = ndimage.uniform_filter(image, size=radius, mode="reflect")
    local_sq_mean = ndimage.uniform_filter(image * image, size=radius, mode="reflect")
    local_var = np.clip(local_sq_mean - local_mean * local_mean, 0.0, None)
    local_std = np.sqrt(local_var)
    normalized = (image - local_mean) / (local_std + 1e-6)
    normalized = intensity_scale(normalized, 1.0, 99.0)
    return histogram_equalize(normalized)


def otsu_threshold(image: np.ndarray) -> float:
    """Compute an Otsu threshold on a [0, 1] grayscale image."""
    hist, bins = np.histogram(np.clip(image, 0.0, 1.0), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    prob = hist / total
    omega = np.cumsum(prob)
    centers = (bins[:-1] + bins[1:]) / 2.0
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b_sq = (mu_t * omega - mu) ** 2 / np.clip(omega * (1.0 - omega), 1e-12, None)
    return float(centers[int(np.argmax(sigma_b_sq))])


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]
