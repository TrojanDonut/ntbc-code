"""CLI: evaluate a trained NTBC run vs the BC1/BC4 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.report import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an NTBC training run.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory.")
    args = parser.parse_args()

    report = evaluate_run(args.run)
    print(f"Evaluation done. Report saved to {args.run}/report.md")
    print(f"  Avg PSNR baseline: {sum(l.psnr_baseline for l in report.layers) / len(report.layers):.2f} dB")
    print(f"  Avg PSNR NTBC:     {sum(l.psnr_ntbc for l in report.layers) / len(report.layers):.2f} dB")
    print(f"  Storage NTBC / BC: {report.storage_ntbc_bytes / max(report.storage_baseline_bytes, 1):.2f}x")


if __name__ == "__main__":
    main()
