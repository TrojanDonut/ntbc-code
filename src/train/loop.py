"""NTBC training loop.

Implements the joint multi-texture training described in Fujieda & Harada
(2024) Section 3.2:

- Main loss: L2 between the decoded BC1/BC4 colors (after argmin index lookup)
  and the reference uncompressed textures.
- Auxiliary color loss: L2 between the predicted ``c_hat`` and the reference
  texture, providing direct gradient to the color network.
- Auxiliary endpoint loss: L2 between predicted endpoints and *reference*
  endpoints obtained by running the pure-Python BC1/BC4 encoder on the
  reference textures (matching the paper's use of Compressonator-derived
  endpoint targets).

Two-phase schedule: the first ``qat_start_frac`` of iterations train with
floating-point grid features. After that, QAT (8-bit fake-quantization) is
enabled for the grid features. We don't freeze the color network in the second
phase -- our experiments at 512x512 do better when both networks keep learning
under QAT than when only the endpoint network is fine-tuned.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..codecs.bc.bc1 import bc1_encode
from ..codecs.bc.bc4 import bc4_encode
from ..data.material import Material
from ..data.preprocess import load_material_from_dir
from ..models.coords import sample_block_batch
from ..models.ntbc import NTBC, NTBCConfig
from .config import TrainConfig


@dataclass
class IterStats:
    step: int
    total: float
    decoded: float
    aux_color: float
    aux_endpoint: float


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_reference_endpoints(material: Material) -> tuple[torch.Tensor, torch.Tensor]:
    """Run BC1 and BC4 on the reference material to get per-block endpoints.

    Returns ``(bc1_endpoints, bc4_endpoints)`` in ``[0, 1]`` floats with shapes
    ``(M, n_bc1, 2, 3)`` and ``(M, n_bc4, 2, 1)``. These mirror the NTBC paper's
    use of Compressonator-derived reference endpoints as an auxiliary supervision
    signal.
    """
    h = w = material.resolution
    n_blocks = (h // 4) * (w // 4)
    bc1_layers = material.bc1_layers
    bc4_layers = material.bc4_layers
    bc1_eps = []
    for layer in bc1_layers:
        rgb = (layer.data.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        enc = bc1_encode(rgb)
        eps = enc.endpoints.astype(np.float32) / 255.0  # (n_blocks, 2, 3)
        bc1_eps.append(torch.from_numpy(eps))
    bc1_tensor = (
        torch.stack(bc1_eps, dim=1) if bc1_eps else torch.zeros(n_blocks, 0, 2, 3)
    )

    bc4_eps = []
    for layer in bc4_layers:
        gray = (layer.data[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        enc = bc4_encode(gray)
        eps = enc.endpoints.astype(np.float32) / 255.0  # (n_blocks, 2)
        bc4_eps.append(torch.from_numpy(eps).unsqueeze(-1))  # (n_blocks, 2, 1)
    bc4_tensor = (
        torch.stack(bc4_eps, dim=1) if bc4_eps else torch.zeros(n_blocks, 0, 2, 1)
    )

    return bc1_tensor, bc4_tensor


def _gather_reference_colors(
    material: Material, texel_yx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather reference BC1 (3-ch) and BC4 (1-ch) colors at the given texel
    pixel coordinates ``(N, 2)`` of (y, x) ints.

    Returns ``(bc1_target, bc4_target)`` with shapes ``(N, n_bc1 * 3)`` and
    ``(N, n_bc4 * 1)`` (empty channels if a material has none).
    """
    y = texel_yx[:, 0].long()
    x = texel_yx[:, 1].long()
    bc1_parts = []
    for layer in material.bc1_layers:
        bc1_parts.append(layer.data[:, y, x].T)  # (N, 3)
    bc4_parts = []
    for layer in material.bc4_layers:
        bc4_parts.append(layer.data[:, y, x].T)  # (N, 1)
    bc1_target = (
        torch.cat(bc1_parts, dim=-1)
        if bc1_parts
        else torch.zeros(texel_yx.shape[0], 0)
    )
    bc4_target = (
        torch.cat(bc4_parts, dim=-1)
        if bc4_parts
        else torch.zeros(texel_yx.shape[0], 0)
    )
    return bc1_target, bc4_target


def _sample_with_yx(
    h: int, w: int, n_blocks: int, block: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like :func:`sample_block_batch` but also returns per-texel (y, x) integer
    coordinates so we can gather reference values."""
    hb, wb = h // block, w // block
    block_ids = torch.randint(0, hb * wb, (n_blocks,), dtype=torch.int64)
    by = block_ids // wb
    bx = block_ids % wb
    st = torch.stack([(bx + 0.5) / wb, (by + 0.5) / hb], dim=-1).float()
    dy, dx = torch.meshgrid(torch.arange(block), torch.arange(block), indexing="ij")
    dy = dy.reshape(-1)
    dx = dx.reshape(-1)
    ty = by[:, None] * block + dy[None, :]
    tx = bx[:, None] * block + dx[None, :]
    uv = torch.stack([(tx.float() + 0.5) / w, (ty.float() + 0.5) / h], dim=-1).reshape(-1, 2)
    texel_yx = torch.stack([ty.reshape(-1), tx.reshape(-1)], dim=-1)
    texel_to_block = torch.arange(n_blocks).repeat_interleave(block * block)
    return uv, st, texel_to_block, texel_yx


def train(material: Material, cfg: TrainConfig) -> tuple[NTBC, list[IterStats]]:
    """Train an NTBC model on a material in-memory. Returns ``(model, history)``."""
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    model = NTBC(material, cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    ref_bc1_eps, ref_bc4_eps = _compute_reference_endpoints(material)
    ref_bc1_eps = ref_bc1_eps.to(device)
    ref_bc4_eps = ref_bc4_eps.to(device)
    n_bc1, n_bc4 = model.n_bc1, model.n_bc4
    h = w = material.resolution
    history: list[IterStats] = []
    qat_step = int(cfg.iters * cfg.qat_start_frac)

    for step in range(cfg.iters):
        if step == qat_step and qat_step < cfg.iters:
            model.color_grid.enable_qat(True)
            model.endpoint_grid.enable_qat(True)

        uv, st, t2b, texel_yx = _sample_with_yx(h, w, cfg.blocks_per_iter)
        uv = uv.to(device)
        st = st.to(device)
        t2b = t2b.to(device)
        bc1_target, bc4_target = _gather_reference_colors(material, texel_yx)
        bc1_target = bc1_target.to(device)
        bc4_target = bc4_target.to(device)

        out = model(uv, st, t2b)

        loss_decoded = torch.zeros((), device=device)
        if n_bc1:
            loss_decoded = loss_decoded + F.mse_loss(out.bc1_decoded, bc1_target)
        if n_bc4:
            loss_decoded = loss_decoded + F.mse_loss(out.bc4_decoded, bc4_target)

        # Auxiliary color loss directly on c_hat to provide gradient bypassing argmin.
        loss_aux_color = torch.zeros((), device=device)
        bc1_color_pred, bc4_color_pred = model.split_colors(out.color_pred)
        if n_bc1 and bc1_color_pred is not None:
            loss_aux_color = loss_aux_color + F.mse_loss(bc1_color_pred, bc1_target)
        if n_bc4 and bc4_color_pred is not None:
            loss_aux_color = loss_aux_color + F.mse_loss(bc4_color_pred, bc4_target)

        # Auxiliary endpoint loss vs reference encoder endpoints (per the paper).
        loss_aux_endpoint = torch.zeros((), device=device)
        block_ids = t2b[::16]  # one block per group of 16 texels
        if n_bc1 and out.bc1_endpoints is not None:
            target_eps = ref_bc1_eps[block_ids]  # (M, n_bc1, 2, 3)
            loss_aux_endpoint = loss_aux_endpoint + F.mse_loss(out.bc1_endpoints, target_eps)
        if n_bc4 and out.bc4_endpoints is not None:
            target_eps = ref_bc4_eps[block_ids]  # (M, n_bc4, 2, 1)
            loss_aux_endpoint = loss_aux_endpoint + F.mse_loss(out.bc4_endpoints, target_eps)

        loss = (
            cfg.color_loss_weight * loss_decoded
            + cfg.aux_color_loss_weight * loss_aux_color
            + cfg.aux_endpoint_loss_weight * loss_aux_endpoint
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0 or step == cfg.iters - 1:
            history.append(
                IterStats(
                    step=step,
                    total=float(loss.detach().cpu()),
                    decoded=float(loss_decoded.detach().cpu()),
                    aux_color=float(loss_aux_color.detach().cpu()),
                    aux_endpoint=float(loss_aux_endpoint.detach().cpu()),
                )
            )
    return model, history


def train_from_config(cfg: TrainConfig) -> Path:
    """Load the material, train, and save model + history into a run directory.

    Returns the run directory path.
    """
    material = load_material_from_dir(Path(cfg.material_dir), resolution=cfg.resolution)
    name = cfg.run_name or f"{material.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.output_dir) / material.name / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(run_dir / "config.yaml")

    start = time.time()
    model, history = train(material, cfg)
    elapsed = time.time() - start

    state = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "material_name": material.name,
        "n_bc1": model.n_bc1,
        "n_bc4": model.n_bc4,
        "resolution": material.resolution,
        "elapsed_seconds": elapsed,
    }
    torch.save(state, run_dir / "model.pt")

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in history], f, indent=2)

    summary = {
        "material": material.name,
        "iters": cfg.iters,
        "blocks_per_iter": cfg.blocks_per_iter,
        "elapsed_seconds": elapsed,
        "iter_per_sec": cfg.iters / max(elapsed, 1e-6),
        "final_total_loss": history[-1].total if history else math.nan,
        "storage_bytes": model.storage_bytes(),
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return run_dir
