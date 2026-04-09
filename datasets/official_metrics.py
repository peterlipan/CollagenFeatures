"""Load official Fiji-derived collagen metrics exported on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from ..utils import normalize_image_name


def load_official_metric_lookup(segmented_root: str | Path) -> pd.DataFrame:
    """Load and merge official TWOMBLI, AnaMorf, and stats outputs by image name."""
    segmented_root = Path(segmented_root)

    twombli_df = _load_twombli_results(segmented_root)
    anamorf_df = _load_anamorf_results(segmented_root)
    stats_df = _load_stats_results(segmented_root)

    merged = anamorf_df
    if not twombli_df.empty:
        merged = merged.merge(twombli_df, on="image_name", how="outer", suffixes=("_anamorf", ""))
        merged = _coalesce_columns(
            merged,
            {
                "mask_area_um2": ["mask_area_um2", "mask_area_um2_anamorf"],
                "lacunarity": ["lacunarity", "lacunarity_anamorf"],
                "fibre_length_um": ["fibre_length_um", "fibre_length_um_anamorf"],
                "endpoints": ["endpoints", "endpoints_anamorf"],
                "hyphal_growth_unit_um": ["hyphal_growth_unit_um", "hyphal_growth_unit_um_anamorf"],
                "branch_points": ["branch_points", "branch_points_anamorf"],
                "fractal_dimension_boxcount": ["fractal_dimension_boxcount", "fractal_dimension_boxcount_anamorf"],
                "curvature_30um_deg": ["curvature_30um_deg", "curvature_30um_deg_anamorf"],
            },
        )

    if not stats_df.empty:
        merged = merged.merge(stats_df, on="image_name", how="outer")

    if "image_name" in merged.columns:
        merged = merged.sort_values("image_name").reset_index(drop=True)
    return merged


def finalize_metric_columns(df: pd.DataFrame, pixel_area_um2: float = 1.0) -> pd.DataFrame:
    """Add derived stable columns after joining with image metadata."""
    result = df.copy()

    if {"mask_area_um2", "image_height_px", "image_width_px"}.issubset(result.columns):
        total_area = result["image_height_px"] * result["image_width_px"] * float(pixel_area_um2)
        total_area = total_area.replace(0, np.nan)
        result["mask_area_fraction"] = result["mask_area_um2"] / total_area

    stable_columns = [
        "image_name",
        "image_path",
        "folder_name",
        "image_height_px",
        "image_width_px",
        "orientationj_backend",
        "orientationj_orientation_deg",
        "orientationj_coherency",
        "mask_area_fraction",
        "mask_area_um2",
        "lacunarity",
        "fibre_length_um",
        "endpoints",
        "branch_points",
        "hyphal_growth_unit_um",
        "curvature_30um_deg",
        "curvature_40um_deg",
        "fractal_dimension_boxcount",
        "fibre_width_um",
        "fibre_width_sd_um",
        "length_tortuosity",
        "length_tortuosity_sd",
        "twombli_alignment",
        "high_density_matrix_fraction",
        "total_image_area_px2",
        "twombli_metrics_source",
        "anamorf_metrics_source",
        "stats_metrics_source",
    ]
    available_columns = [column for column in stable_columns if column in result.columns]
    return result[available_columns]


def _load_twombli_results(segmented_root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(segmented_root.glob("Twombli_Results*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except EmptyDataError:
            continue
        if df.empty or "Image" not in df.columns:
            continue
        df = df.rename(
            columns={
                "Image": "image_name",
                "Area (microns^2)": "mask_area_um2",
                "Lacunarity": "lacunarity",
                "Total Length (microns)": "fibre_length_um",
                "Endpoints": "endpoints",
                "HGU (microns)": "hyphal_growth_unit_um",
                "Branchpoints": "branch_points",
                "Box-Counting Fractal Dimension": "fractal_dimension_boxcount",
                "Curvature_30.0": "curvature_30um_deg",
                "Curvature_40.0": "curvature_40um_deg",
                "Alignment": "twombli_alignment",
                "% High Density Matrix": "high_density_matrix_fraction",
                "TotalImageArea": "total_image_area_px2",
            }
        )
        df["image_name"] = df["image_name"].map(normalize_image_name)
        df["twombli_metrics_source"] = csv_path.name
        frames.append(df)
    return _combine_frames(frames)


def _load_anamorf_results(segmented_root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(segmented_root.glob("*_Masks/**/results.csv")):
        try:
            df = pd.read_csv(csv_path)
        except EmptyDataError:
            continue
        if df.empty or "Image" not in df.columns:
            continue
        df = df[df["Image"].astype(str).str.contains(r"\.(?:png|tif)$", case=False, regex=True)].copy()
        if df.empty:
            continue
        df = df.rename(
            columns={
                "Image": "image_name",
                "Area (microns^2)": "mask_area_um2",
                "Lacunarity": "lacunarity",
                "Total Length (microns)": "fibre_length_um",
                "Endpoints": "endpoints",
                "HGU (microns)": "hyphal_growth_unit_um",
                "Branchpoints": "branch_points",
                "Box-Counting Fractal Dimension": "fractal_dimension_boxcount",
                "Curvature_30.0": "curvature_30um_deg",
                "Curvature_40.0": "curvature_40um_deg",
            }
        )
        df["image_name"] = df["image_name"].map(normalize_image_name)
        df["anamorf_metrics_source"] = str(csv_path.relative_to(segmented_root))
        frames.append(df)
    return _combine_frames(frames)


def _load_stats_results(segmented_root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(segmented_root.glob("*_stats.csv")):
        try:
            df = pd.read_csv(csv_path)
        except EmptyDataError:
            continue
        if df.empty or "Image" not in df.columns:
            continue
        df.columns = [column.strip().lstrip("\ufeff") for column in df.columns]
        if "SD_Tortuosity" not in df.columns:
            df["SD_Tortuosity"] = np.nan
        df = df.rename(
            columns={
                "Image": "image_name",
                "Mean_Radius": "fibre_width_um",
                "SD_Radius": "fibre_width_sd_um",
                "Mean_Tortuosity": "length_tortuosity",
                "SD_Tortuosity": "length_tortuosity_sd",
            }
        )
        df["image_name"] = df["image_name"].map(normalize_image_name)
        df["stats_metrics_source"] = csv_path.name
        frames.append(df)
    return _combine_frames(frames)


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["image_name"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=["image_name"], keep="first")


def _coalesce_columns(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    for target, sources in mapping.items():
        available = [column for column in sources if column in df.columns]
        if not available:
            continue
        series = df[available[0]]
        for column in available[1:]:
            series = series.combine_first(df[column])
        df[target] = series
        for column in available:
            if column != target and column in df.columns:
                df = df.drop(columns=column)
    return df
