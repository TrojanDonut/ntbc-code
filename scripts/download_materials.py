"""CLI: download a curated set of CC0 PBR materials from ambientCG."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.download import DEFAULT_MATERIALS, download_materials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PBR materials from ambientCG into a local directory."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/materials"),
        help="Output directory (one subfolder per material).",
    )
    parser.add_argument(
        "--materials", nargs="*", default=None,
        help="Optional ambientCG material names. Defaults to a curated set.",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Limit to first N default materials (ignored if --materials is set).",
    )
    parser.add_argument(
        "--resolution", default="1K-PNG",
        help="ambientCG resolution suffix (e.g. 1K-PNG, 2K-PNG).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if files already exist.",
    )
    args = parser.parse_args()

    if args.materials:
        names = list(args.materials)
    else:
        names = list(DEFAULT_MATERIALS)
        if args.count is not None:
            names = names[: args.count]

    results = download_materials(
        names, args.out, resolution=args.resolution, overwrite=args.overwrite
    )

    n_ok = sum(1 for r in results if r.error is None)
    print(f"Downloaded {n_ok}/{len(results)} materials to {args.out}")
    for r in results:
        if r.error:
            print(f"  FAILED: {r.name}: {r.error}")
        elif r.skipped:
            print(f"  skipped (already present): {r.name} -> {r.path}")
        else:
            print(f"  ok: {r.name} -> {r.path}")


if __name__ == "__main__":
    main()
