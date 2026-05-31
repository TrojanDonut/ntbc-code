"""CLI: train an NTBC model on one material from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train.config import TrainConfig
from src.train.loop import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NTBC on a single material.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument(
        "--material-dir", type=Path, default=None,
        help="Override material_dir from the YAML config.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Override run_name from the YAML config.",
    )
    parser.add_argument("--iters", type=int, default=None, help="Override iters.")
    parser.add_argument("--resolution", type=int, default=None, help="Override resolution.")
    parser.add_argument(
        "--blocks-per-iter", type=int, default=None, help="Override blocks_per_iter.",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.material_dir is not None:
        cfg.material_dir = str(args.material_dir)
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.iters is not None:
        cfg.iters = args.iters
    if args.resolution is not None:
        cfg.resolution = args.resolution
    if args.blocks_per_iter is not None:
        cfg.blocks_per_iter = args.blocks_per_iter

    run_dir = train_from_config(cfg)
    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
