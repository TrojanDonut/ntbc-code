"""Reproduction instructions and optional runners for the NTBC seminar project.

Assignment requirements are satisfied by training one material (joint layers,
weights-only on disk) and evaluating against BC1/BC4 (compression, quality,
encoding time). Everything else is optional scale-up or report packaging.

Examples:
  python -m scripts.reproduce --help
  python -m scripts.reproduce --minimal          # print minimal commands
  python -m scripts.reproduce --minimal --run  # execute minimal pipeline
  python -m scripts.reproduce --full             # print full-study commands
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATERIAL = "MetalPlates013"
DEFAULT_CONFIG = "configs/storage_demo.yaml"
DEFAULT_RES = 1024
DEFAULT_ITERS = 10000
DEFAULT_RUN = "demo"


def _py() -> str:
    return sys.executable


def _minimal_commands(
    material: str = DEFAULT_MATERIAL,
    run_name: str = DEFAULT_RUN,
    resolution: int = DEFAULT_RES,
    iters: int = DEFAULT_ITERS,
) -> str:
    mat_dir = f"data/materials/{material}"
    run_dir = f"results/{material}/{run_name}"
    return f"""\
# Environment (once)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m scripts.download_materials --out data/materials --materials {material}

# Minimal assignment pipeline: train NTBC, evaluate vs BC1/BC4
python3 -m scripts.train --config {DEFAULT_CONFIG} \\
  --material-dir {mat_dir} --run-name {run_name} \\
  --resolution {resolution} --iters {iters}
python3 -m scripts.eval --run {run_dir}

# Optional qualitative output (not required for metrics)
python3 -m scripts.render_sphere --run {run_dir}
"""


def _full_commands() -> str:
    return """\
# Full study (12 materials, ~114 runs; multi-hour on CPU)
bash scripts/run_cpu_long_batch.sh
bash scripts/run_cpu_long_batch.sh \\
  --manifest scripts/cpu_long_batch2_manifest.json
python3 -m scripts.aggregate_all_results

# Report PDF and figures
python3 -m scripts.build_report_figures
make -C docs all
# or: bash scripts/build_report.sh

# Storage-demo headline runs (paper crossover regime)
python3 -m scripts.run_storage_demo --resolution 1024 --iters 10000 \\
  --material MetalPlates013
python3 -m scripts.run_storage_demo --resolution 1024 --iters 10000 \\
  --material Carpet015
python3 -m scripts.run_storage_demo --resolution 2048 --iters 5000 \\
  --material MetalPlates013
"""


def _run_cmd(argv: list[str], cwd: Path) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def _run_minimal(
    material: str,
    run_name: str,
    resolution: int,
    iters: int,
    skip_download: bool,
) -> None:
    py = _py()
    mat_dir = ROOT / "data" / "materials" / material
    if not skip_download:
        _run_cmd(
            [py, "-m", "scripts.download_materials", "--out", "data/materials", "--materials", material],
            ROOT,
        )
    if not mat_dir.is_dir():
        raise SystemExit(f"Missing material directory: {mat_dir}")

    _run_cmd(
        [
            py,
            "-m",
            "scripts.train",
            "--config",
            DEFAULT_CONFIG,
            "--material-dir",
            str(mat_dir),
            "--run-name",
            run_name,
            "--resolution",
            str(resolution),
            "--iters",
            str(iters),
        ],
        ROOT,
    )
    run_dir = ROOT / "results" / material / run_name
    _run_cmd([py, "-m", "scripts.eval", "--run", str(run_dir)], ROOT)
    print(f"\nDone. Metrics: {run_dir}/report.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "NTBC reproduction helper. The minimal path (train + eval on one "
            "material) satisfies every assignment requirement; use --full to "
            "print commands for the complete study and report build."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Minimal commands:\n"
            + _minimal_commands()
            + "\nFull study / report:\n"
            + _full_commands()
        ),
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Print the minimal train+eval commands.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print commands for the full experiment sweep and report PDF.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="With --minimal: download (unless --skip-download), train, and eval.",
    )
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RES)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="With --run: skip ambientCG download.",
    )
    args = parser.parse_args()

    if args.full:
        print(_full_commands(), end="")
    elif args.minimal:
        if args.run:
            _run_minimal(
                args.material,
                args.run_name,
                args.resolution,
                args.iters,
                args.skip_download,
            )
        else:
            print(
                _minimal_commands(args.material, args.run_name, args.resolution, args.iters),
                end="",
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
