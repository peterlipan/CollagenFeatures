"""Pure-Python fallback for dominant fibre orientation estimation."""

from __future__ import annotations

import numpy as np
from scipy import signal


def dominant_orientation(image: np.ndarray) -> dict[str, float]:
    """
    Estimate dominant orientation and coherence from a grayscale image.

    This mirrors the OrientationJ dominant-direction tensor calculation closely
    enough to serve as a deterministic fallback when Fiji is unavailable.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("dominant_orientation expects a 2D grayscale image")

    height, width = image.shape
    if height < 3 or width < 3:
        return {"Orientation": 0.0, "Coherence": 0.0}

    kernel = np.array([1, -8, 0, 8, -1], dtype=np.float32) / 12.0
    dx = signal.convolve2d(image, kernel.reshape(1, 5), mode="same", boundary="fill", fillvalue=0)
    dy = signal.convolve2d(image, kernel.reshape(5, 1), mode="same", boundary="fill", fillvalue=0)

    dx = dx[1:-1, 1:-1]
    dy = dy[1:-1, 1:-1]

    vxx = np.mean(dx**2)
    vyy = np.mean(dy**2)
    vxy = np.mean(dx * dy)

    orientation = 0.5 * np.arctan2(2 * vxy, vyy - vxx)
    orientation_deg = float(np.degrees(orientation))

    numerator = np.sqrt((vyy - vxx) ** 2 + 4 * vxy**2)
    denominator = vxx + vyy
    coherence = float(numerator / denominator) if denominator > 1 else 0.0

    return {"Orientation": orientation_deg, "Coherence": coherence}
