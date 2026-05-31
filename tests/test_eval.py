"""End-to-end smoke test of the evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.eval.report import evaluate_run
from src.models.hash_grid import MultiResGridConfig
from src.models.ntbc import NTBCConfig
from src.train.config import TrainConfig
from src.train.loop import train_from_config


def _seed_material_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    gray = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    Image.fromarray(rgb).save(base / "FakeMat_1K-PNG_Color.png")
    Image.fromarray(rgb).save(base / "FakeMat_1K-PNG_NormalGL.png")
    Image.fromarray(gray, mode="L").save(base / "FakeMat_1K-PNG_Roughness.png")
    Image.fromarray(gray, mode="L").save(base / "FakeMat_1K-PNG_Displacement.png")
    return base


def test_evaluate_run_writes_all_artifacts(tmp_path):
    torch.set_num_threads(1)
    mat_dir = _seed_material_dir(tmp_path / "FakeMat")
    cfg = TrainConfig(
        material_dir=str(mat_dir),
        output_dir=str(tmp_path / "out"),
        run_name="smoke",
        resolution=32,
        iters=40,
        blocks_per_iter=16,
        log_every=10,
        seed=0,
        model=NTBCConfig(
            color_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=32, n_features=2),
            endpoint_grid=MultiResGridConfig(n_levels=3, coarsest=4, finest=8, n_features=2),
            mlp_hidden=16,
            mlp_layers=2,
        ),
    )
    run_dir = train_from_config(cfg)
    report = evaluate_run(run_dir)
    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "crops").is_dir()
    assert len(list((run_dir / "crops").glob("*.png"))) == 4
    js = json.loads((run_dir / "report.json").read_text())
    assert js["material"] == "FakeMat"
    assert len(js["layers"]) == 4
    for layer in js["layers"]:
        assert layer["psnr_baseline"] > 0
        assert layer["psnr_ntbc"] > 0
        assert 0.0 <= layer["ssim_baseline"] <= 1.0
        assert 0.0 <= layer["ssim_ntbc"] <= 1.0
