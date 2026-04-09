"""Main batch entry point for the red-channel workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipelines import build_red_channel_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = Path("/datastorage/li/CollagenRawImages")
SEGMENTED_ROOT = Path("/datastorage/li/Collagen/Segmented")
OUTPUT_CSV = PROJECT_ROOT / "results" / "red_channel_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="Root folder containing raw image folders.")
    parser.add_argument(
        "--segmented-root",
        type=Path,
        default=SEGMENTED_ROOT,
        help="Root folder containing official segmented exports.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Path for the merged output CSV.",
    )
    args = parser.parse_args()

    df = build_red_channel_metrics(
        raw_root=args.raw_root,
        segmented_root=args.segmented_root,
        output_csv=args.output_csv,
    )
    print(f"Saved {len(df)} image metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
