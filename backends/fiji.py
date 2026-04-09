"""Optional Fiji/ImageJ integration for plugin-based collagen metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
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
    twombli_dir: str | None = None
    ij: Any | None = None
    _orientation_plugin: Any | None = None
    _twombli_runner_class: Any | None = None

    def __post_init__(self) -> None:
        if self.orientationj_jar is None:
            discovered = self._discover_orientationj_jar()
            self.orientationj_jar = str(discovered) if discovered is not None else None
        if self.twombli_dir is None:
            discovered_dir = self._discover_twombli_dir()
            self.twombli_dir = str(discovered_dir) if discovered_dir is not None else None

    def initialize(self) -> Any | None:
        if self.ij is not None:
            return self.ij
        if imagej is None:
            return None

        try:
            if not scyjava.jvm_started():
                for jar in self._classpath_jars():
                    scyjava.config.add_classpath(str(jar))
            if self.fiji_path:
                self.ij = imagej.init(self.fiji_path, mode=self.mode)
            elif self.prefer_orientationj or self.orientationj_jar or self.twombli_dir:
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

    def compute_twombli_metrics(
        self,
        image: np.ndarray,
        image_name: str,
        output_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, float] | None:
        """Run the official TWOMBLI pipeline headlessly and parse its outputs."""
        ij = self.initialize()
        if ij is None or self.twombli_dir is None:
            return None

        work_dir: Path | None = None
        temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        if output_dir is None:
            temporary_dir = tempfile.TemporaryDirectory(prefix="twombli_")
            work_dir = Path(temporary_dir.name)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = self._run_twombli(image=image, image_name=image_name, output_dir=work_dir, config=config or {})
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()

        return result

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

    def _run_twombli(
        self,
        image: np.ndarray,
        image_name: str,
        output_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, float] | None:
        from jpype import JArray, JFloat

        if self._twombli_runner_class is None:
            self._twombli_runner_class = scyjava.jimport("uk.ac.franciscrickinstitute.twombli.TWOMBLIRunner")

        float_processor = scyjava.jimport("ij.process.FloatProcessor")
        image_plus = scyjava.jimport("ij.ImagePlus")

        pixels = np.ascontiguousarray(self._to_orientationj_pixels(image), dtype=np.float32)
        java_pixels = JArray(JFloat)(pixels.ravel().tolist())
        processor = float_processor(pixels.shape[1], pixels.shape[0], java_pixels)
        imp = image_plus(f"{image_name}.tif", processor)

        runner = self._twombli_runner_class()
        runner.img = imp
        runner.outputPath = str(output_dir)
        runner.filePrefix = image_name
        runner.minimumLineWidth = int(config.get("minimum_line_width", 5))
        runner.maximumLineWidth = int(config.get("maximum_line_width", 10))
        runner.minimumBranchLength = int(config.get("minimum_branch_length", 5))
        runner.minimumCurvatureWindow = int(config.get("minimum_curvature_window", 30))
        runner.curvatureWindowStepSize = int(config.get("curvature_window_step_size", 10))
        runner.maximumCurvatureWindow = int(config.get("maximum_curvature_window", 40))
        runner.maximumDisplayHDM = int(config.get("maximum_display_hdm", 200))
        runner.contrastSaturation = float(config.get("contrast_saturation", 0.35))
        runner.performGapAnalysis = bool(config.get("perform_gap_analysis", True))
        runner.minimumGapDiameter = int(config.get("minimum_gap_diameter", 40))

        runner_error: Exception | None = None
        try:
            runner.run()
        except Exception as exc:  # pragma: no cover - runtime dependent
            runner_error = exc

        result_csv = output_dir / "masks" / f"{image_name}_results.csv"
        hdm_csv = output_dir / "hdm_csvs" / f"{image_name}_ResultsHDM.csv"
        mask_path = output_dir / "masks" / f"{image_name}_masks.png"

        if not result_csv.exists():
            if runner_error is not None:
                raise runner_error
            return None

        result = self._parse_twombli_outputs(result_csv=result_csv, hdm_csv=hdm_csv, mask_path=mask_path)
        result["twombli_backend"] = "fiji_twombli"
        return result

    def _parse_twombli_outputs(self, result_csv: Path, hdm_csv: Path, mask_path: Path) -> dict[str, float]:
        df = pd.read_csv(result_csv)
        image_col = df["Image"].astype(str)
        df = df[image_col.str.endswith("_masks.png")].copy()
        if not df.empty:
            nonzero_df = df[df["Area (microns^2)"].fillna(0) > 0].copy()
            if not nonzero_df.empty:
                df = nonzero_df
        if df.empty:
            raise ValueError(f"No TWOMBLI result row found in {result_csv}")
        row = df.iloc[-1]

        result = {
            "mask_area_um2": float(row.get("Area (microns^2)", np.nan)),
            "lacunarity": float(row.get("Lacunarity", np.nan)),
            "fibre_length_um": float(row.get("Total Length (microns)", np.nan)),
            "endpoints": float(row.get("Endpoints", np.nan)),
            "branch_points": float(row.get("Branchpoints", np.nan)),
            "hyphal_growth_unit_um": float(row.get("HGU (microns)", np.nan)),
            "fractal_dimension_boxcount": float(row.get("Box-Counting Fractal Dimension", np.nan)),
            "curvature_30um_deg": float(row.get("Curvature_30", np.nan)),
            "curvature_40um_deg": float(row.get("Curvature_40", np.nan)),
        }

        if hdm_csv.exists():
            hdm_df = pd.read_csv(hdm_csv)
            if not hdm_df.empty and "% HDM" in hdm_df.columns:
                result["high_density_matrix_fraction"] = float(hdm_df.iloc[0]["% HDM"])

        return self._attach_alignment_from_mask(result, mask_path)

    def _attach_alignment_from_mask(self, result: dict[str, float], mask_path: Path) -> dict[str, float]:
        from PIL import Image

        if mask_path.exists():
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
            orientation = self.compute_orientation_metrics(mask)
            result["twombli_alignment"] = orientation["orientationj_coherency"]
        else:
            result["twombli_alignment"] = np.nan
        return result

    @staticmethod
    def _discover_orientationj_jar() -> Path | None:
        candidates = (
            Path(__file__).resolve().parents[1] / "third_party" / "twombli" / "OrientationJ_.jar",
            Path("/datastorage/li/fiji/plugins/OrientationJ/OrientationJ_.jar"),
            Path("/home/li/fiji/plugins/OrientationJ/OrientationJ_.jar"),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _discover_twombli_dir() -> Path | None:
        candidate = Path(__file__).resolve().parents[1] / "third_party" / "twombli"
        return candidate if candidate.exists() else None

    def _classpath_jars(self) -> list[Path]:
        jars: list[Path] = []
        if self.orientationj_jar:
            jars.append(Path(self.orientationj_jar))
        if self.twombli_dir:
            jars.extend(sorted(Path(self.twombli_dir).glob("*.jar*")))
        deduped: list[Path] = []
        seen: set[str] = set()
        for jar in jars:
            key = str(jar.resolve())
            if jar.exists() and key not in seen:
                seen.add(key)
                deduped.append(jar)
        return deduped

    @staticmethod
    def _to_orientationj_pixels(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.0:
            return np.clip(image, 0.0, 1.0) * 255.0
        return np.clip(image, 0.0, 255.0)
