"""Optional Fiji/ImageJ integration for plugin-based collagen metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scyjava

try:
    import imagej
except Exception:  # pragma: no cover
    imagej = None

from ..metrics.orientation import dominant_orientation


@dataclass
class FijiBackend:
    """Thin wrapper around PyImageJ with safe fallback behavior."""

    fiji_path: str | None = None
    mode: str = "headless"
    prefer_orientationj: bool = False
    orientationj_jar: str | None = None
    ij: Any | None = None
    _orientation_plugin: Any | None = None

    def __post_init__(self) -> None:
        if self.orientationj_jar is None:
            discovered = self._discover_orientationj_jar()
            self.orientationj_jar = str(discovered) if discovered is not None else None

    def initialize(self) -> Any | None:
        if self.ij is not None:
            return self.ij
        if imagej is None:
            return None

        try:
            if self.orientationj_jar and not scyjava.jvm_started():
                scyjava.config.add_classpath(self.orientationj_jar)
            if self.fiji_path:
                self.ij = imagej.init(self.fiji_path, mode=self.mode)
            elif self.prefer_orientationj or self.orientationj_jar:
                self.ij = imagej.init(mode=self.mode)
        except Exception:
            self.ij = None
        return self.ij

    def compute_orientation_metrics(self, image: np.ndarray) -> dict[str, float]:
        """Return orientation/coherency, using OrientationJ when available."""
        ij = self.initialize()
        if ij is not None and self.orientationj_jar:
            plugin_result = self._try_orientationj(image)
            if plugin_result is not None:
                return plugin_result

        result = dominant_orientation(self._to_orientationj_pixels(image))
        return {
            "orientationj_backend": "python_fallback",
            "orientationj_orientation_deg": float(result["Orientation"]),
            "orientationj_coherency": float(result["Coherence"]),
        }

    def compute_twombli_metrics(self, _image_path: str) -> dict[str, float]:
        """Placeholder for plugin-backed TWOMBLI execution."""
        return {"twombli_backend": "python_proxy"}

    def _try_orientationj(self, image: np.ndarray) -> dict[str, float] | None:
        try:
            from jpype import JArray, JFloat

            if self._orientation_plugin is None:
                orientation_class = scyjava.jimport("OrientationJ_Dominant_Direction")
                self._orientation_plugin = orientation_class()
            float_processor = scyjava.jimport("ij.process.FloatProcessor")

            pixels = np.ascontiguousarray(self._to_orientationj_pixels(image), dtype=np.float32)
            java_pixels = JArray(JFloat)(pixels.ravel().tolist())
            processor = float_processor(pixels.shape[1], pixels.shape[0], java_pixels)
            result = self._orientation_plugin.computeSpline(processor)
            return {
                "orientationj_backend": "fiji_orientationj",
                "orientationj_orientation_deg": float(result[0]),
                "orientationj_coherency": float(result[1]),
            }
        except Exception:
            return None

    @staticmethod
    def _discover_orientationj_jar() -> Path | None:
        candidates = (
            Path("/datastorage/li/fiji/plugins/OrientationJ/OrientationJ_.jar"),
            Path("/home/li/fiji/plugins/OrientationJ/OrientationJ_.jar"),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _to_orientationj_pixels(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.0:
            return np.clip(image, 0.0, 1.0) * 255.0
        return np.clip(image, 0.0, 255.0)
