"""Run configuration: YAML-backed dataclass for training NTBC on one material."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..models.hash_grid import MultiResGridConfig
from ..models.ntbc import NTBCConfig


@dataclass
class TrainConfig:
    """Top-level training configuration for a single NTBC run."""

    material_dir: str = "data/materials/MetalPlates013"
    output_dir: str = "results"
    run_name: str | None = None  # default: derived from material name + timestamp
    resolution: int = 512
    iters: int = 5000
    blocks_per_iter: int = 256
    learning_rate: float = 1e-2
    qat_start_frac: float = 0.8
    color_loss_weight: float = 1.0
    aux_color_loss_weight: float = 0.1
    aux_endpoint_loss_weight: float = 0.1
    seed: int = 0
    log_every: int = 100
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    model: NTBCConfig = field(default_factory=NTBCConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        model_data = data.pop("model", None)
        if model_data is not None:
            color_grid_data = model_data.pop("color_grid", None)
            endpoint_grid_data = model_data.pop("endpoint_grid", None)
            model_cfg = NTBCConfig(**model_data)
            if color_grid_data is not None:
                model_cfg.color_grid = MultiResGridConfig(**color_grid_data)
            if endpoint_grid_data is not None:
                model_cfg.endpoint_grid = MultiResGridConfig(**endpoint_grid_data)
            data["model"] = model_cfg
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
