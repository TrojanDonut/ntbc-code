"""Smoke tests for the training loop and config serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data.preprocess import make_synthetic_material
from src.models.hash_grid import MultiResGridConfig
from src.models.ntbc import NTBCConfig
from src.train.config import TrainConfig
from src.train.loop import train, train_from_config


def _small_cfg(iters: int = 50) -> TrainConfig:
    return TrainConfig(
        iters=iters,
        blocks_per_iter=16,
        log_every=10,
        seed=0,
        model=NTBCConfig(
            color_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=16, n_features=2),
            endpoint_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=8, n_features=2),
            mlp_hidden=16,
            mlp_layers=2,
        ),
    )


def test_training_reduces_loss():
    torch.set_num_threads(1)
    mat = make_synthetic_material(resolution=16, seed=0)
    cfg = _small_cfg(iters=60)
    model, history = train(mat, cfg)
    assert len(history) >= 2
    assert history[-1].decoded < history[0].decoded
    assert torch.all(torch.tensor([s.total for s in history]).isfinite())


def test_config_roundtrip(tmp_path):
    cfg = _small_cfg()
    p = tmp_path / "cfg.yaml"
    cfg.save_yaml(p)
    cfg2 = TrainConfig.from_yaml(p)
    assert cfg2.iters == cfg.iters
    assert cfg2.blocks_per_iter == cfg.blocks_per_iter
    assert cfg2.model.color_grid.n_levels == cfg.model.color_grid.n_levels


def test_train_from_config_writes_artifacts(tmp_path):
    torch.set_num_threads(1)
    # Synthetic material on disk: write a tiny folder of fake PNGs.
    from PIL import Image
    import numpy as np
    mat_dir = tmp_path / "FakeMat"
    mat_dir.mkdir()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    gray = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    Image.fromarray(rgb).save(mat_dir / "FakeMat_1K-PNG_Color.png")
    Image.fromarray(gray, mode="L").save(mat_dir / "FakeMat_1K-PNG_Roughness.png")

    cfg = _small_cfg(iters=20)
    cfg.material_dir = str(mat_dir)
    cfg.output_dir = str(tmp_path / "out")
    cfg.resolution = 16
    cfg.run_name = "smoke"
    cfg.log_every = 5
    run_dir = train_from_config(cfg)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "history.json").exists()
    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["material"] == "FakeMat"
    assert summary["iters"] == 20
